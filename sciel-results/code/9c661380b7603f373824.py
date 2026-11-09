"""Product-formula coefficient optimizer for 6-site Heisenberg chain splitting."""
import numpy as np
import time
import sys

# Deadline guard: evaluator kills after 150s wall-clock; use 130s as safety margin.
_DEADLINE = 130.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain.

    Sites 0..5, open boundary. Bonds: (0,1),(1,2),(2,3),(3,4),(4,5).
    Odd bonds (0-indexed): bonds 1,3 -> (1,2),(3,4)
    Even bonds (0-indexed): bonds 0,2,4 -> (0,1),(2,3),(4,5)

    XXZ anisotropy delta=0.8, field=0.3
    """
    N = 6
    delta = 0.8
    field = 0.3

    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def site_op(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)

    def bond_op(i, j):
        """XXZ bond between sites i and j: Sz^i Sz^j + delta/2 (Sx^i Sx^j + Sy^i Sy^j)"""
        ops_szs = [I2] * N
        ops_szs[i] = sz
        ops_szs[j] = sz
        SzSz = kron_all(ops_szs)

        ops_sxs = [I2] * N
        ops_sxs[i] = sx
        ops_sxs[j] = sx
        SxSx = kron_all(ops_sxs)

        ops_sys = [I2] * N
        ops_sys[i] = sy
        ops_sys[j] = sy
        SySy = kron_all(ops_sys)

        return SzSz + delta / 2.0 * (SxSx + SySy)

    bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    # Even bonds: indices 0, 2, 4 -> (0,1), (2,3), (4,5) -> B
    # Odd bonds: indices 1, 3 -> (1,2), (3,4) -> A

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)

    for idx, (i, j) in enumerate(bonds):
        H_bond = bond_op(i, j)
        if idx % 2 == 0:  # even bond -> B
            B += H_bond
        else:  # odd bond -> A
            A += H_bond

    # Uniform field split: half into A, half into B
    # Field term: -h * sum_i Sz^i
    H_field = np.zeros((2**N, 2**N), dtype=complex)
    for s in range(N):
        H_field += site_op(s, sz)

    half_field = -field * 0.5 * H_field
    A += half_field
    B += half_field

    return A, B


def _expm(mat):
    """Compute matrix exponential using scipy if available, else fallback."""
    try:
        from scipy.linalg import expm
        return expm(mat)
    except ImportError:
        # Eigen-decomposition fallback
        eigvals, eigvecs = np.linalg.eig(mat)
        return eigvecs @ np.diag(np.exp(eigvals)) @ np.linalg.inv(eigvecs)


def _product_formula(A, B, a, b, t):
    """Compute U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) for k=1..12.
    
    The product is ordered such that k=1 is applied first (rightmost in matrix mult).
    So U = E_B12 E_A12 ... E_B2 E_A2 E_B1 E_A1 where E_Bk = exp(-i b_k t B), E_Ak = exp(-i a_k t A)
    In code: start with U = I, then for k from 11 down to 0: U = exp(-i b_k t B) @ exp(-i a_k t A) @ U
    OR equivalently: U = I; for k in range(12): U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
    Wait, let me think about this more carefully.
    
    "prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12"
    This means the product is: (factor_1)(factor_2)...(factor_12)
    where factor_k = exp(-i b_k t B) exp(-i a_k t A)
    
    In matrix multiplication, if we write P = F_1 F_2 ... F_12, then when acting on a state |psi>,
    F_12 acts first, then F_11, ..., then F_1. So F_1 is leftmost.
    
    So U = F_1 @ F_2 @ ... @ F_12
    where F_k = exp(-i b_k t B) @ exp(-i a_k t A)
    
    In code:
    U = I
    for k in range(12):
        U = U @ (exp(-i b_k t B) @ exp(-i a_k t A))
    """
    dim = A.shape[0]
    U = np.eye(dim, dtype=complex)
    for k in range(len(a)):
        U = U @ _expm(-1j * b[k] * t * B) @ _expm(-1j * a[k] * t * A)
    return U


def _exact_propagator(A, B, t):
    """Compute exp(-i t (A + B))."""
    return _expm(-1j * t * (A + B))


def _spectral_norm_error(A, B, a, b, t):
    """Compute ||U(t) - exp(-i t (A+B))||_2."""
    U_approx = _product_formula(A, B, a, b, t)
    U_exact = _exact_propagator(A, B, t)
    diff = U_approx - U_exact
    return np.linalg.norm(diff, 2)


def _score(A, B, a, b):
    """Mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    errors = [_spectral_norm_error(A, B, a, b, t) for t in times]
    return np.mean(errors)


def _normalize_sum(arr):
    """Normalize array so sum equals 1, keeping values in [-50, 50]."""
    s = sum(arr)
    if abs(s) < 1e-15:
        # If sum is near zero, set all to 1/len
        n = len(arr)
        return [1.0 / n] * n
    return [x / s for x in arr]


def _clip_and_normalize(arr, low=-50.0, high=50.0):
    """Clip values to [low, high] and then normalize to sum to 1."""
    clipped = [max(low, min(high, x)) for x in arr]
    s = sum(clipped)
    if abs(s) < 1e-15:
        n = len(clipped)
        return [1.0 / n] * n
    return [x / s for x in clipped]


def _gradient_free_optimize(A, B, a0, b0, max_evals=300, step_size=0.05):
    """Coordinate descent with small perturbations to optimize coefficients."""
    best_a = list(a0)
    best_b = list(b0)
    best_score = _score(A, B, best_a, best_b)
    
    n = len(a0)
    eval_count = 0
    no_improve_count = 0
    
    while eval_count < max_evals and _remaining() > 5.0:
        improved = False
        for i in range(n):
            if _remaining() < 5.0:
                break
            # Try small perturbations in a[i]
            for delta in [step_size, -step_size, step_size/2, -step_size/2]:
                eval_count += 1
                a_trial = list(best_a)
                b_trial = list(best_b)
                a_trial[i] += delta
                # Normalize to maintain sum=1
                a_trial = _clip_and_normalize(a_trial)
                s = _score(A, B, a_trial, b_trial)
                if s < best_score:
                    best_a = a_trial
                    best_score = s
                    improved = True
                    break
            
            # Try small perturbations in b[i]
            for delta in [step_size, -step_size, step_size/2, -step_size/2]:
                eval_count += 1
                a_trial = list(best_a)
                b_trial = list(best_b)
                b_trial[i] += delta
                b_trial = _clip_and_normalize(b_trial)
                s = _score(A, B, a_trial, b_trial)
                if s < best_score:
                    best_b = b_trial
                    best_score = s
                    improved = True
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                # Reduce step size
                step_size *= 0.5
                no_improve_count = 0
                if step_size < 1e-6:
                    break
        else:
            no_improve_count = 0
    
    return best_a, best_b, best_score


def coefficients(stages):
    """
    Returns coefficients for a product formula approximating exp(-it(A+B)).
    Structure: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    
    Uses an initial Strang-like splitting as starting point, then refines
    via coordinate descent if time permits.
    """
    if _remaining() < 10.0:
        # Not enough time for optimization; return a reasonable default
        n = stages
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b

    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    if _remaining() < 5.0:
        # Not enough time; return default
        n = stages
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b

    # Initial guess: symmetric Strang-like splitting
    # For 12 stages, use a pattern that approximates higher-order splitting
    n = stages
    if n == 12:
        # Symmetric coefficients that sum to 1
        # Start with uniform, then refine
        a = [1.0 / 12] * 12
        b = [1.0 / 12] * 12
    else:
        a = [1.0 / n] * n
        b = [1.0 / n] * n

    # Try a few structured initial guesses and pick the best
    candidates = []
    
    # Candidate 1: uniform
    candidates.append((list(a), list(b)))
    
    # Candidate 2: Strang-like with emphasis on middle
    if n == 12:
        # Symmetric pattern
        a2 = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08]
        b2 = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08]
        # Normalize
        a2 = _normalize_sum(a2)
        b2 = _normalize_sum(b2)
        candidates.append((a2, b2))
        
        # Candidate 3: endpoints smaller
        a3 = [0.05, 0.10, 0.12, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12, 0.10, 0.05]
        b3 = [0.05, 0.10, 0.12, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12, 0.10, 0.05]
        a3 = _normalize_sum(a3)
        b3 = _normalize_sum(b3)
        candidates.append((a3, b3))
    
    # Evaluate candidates and pick best
    best_a, best_b, best_score = candidates[0][0], candidates[0][1], _score(A, B, candidates[0][0], candidates[0][1])
    
    for a_cand, b_cand in candidates[1:]:
        if _remaining() < 5.0:
            break
        s = _score(A, B, a_cand, b_cand)
        if s < best_score:
            best_a, best_b, best_score = a_cand, b_cand, s
    
    # Refine via coordinate descent if time permits
    if _remaining() > 10.0:
        max_evals = min(200, int(_remaining() * 5))  # Rough estimate
        best_a, best_b, best_score = _gradient_free_optimize(A, B, best_a, best_b, max_evals=max_evals)
    
    # Final check: ensure sums are exactly 1 and values are finite
    best_a = _normalize_sum(best_a)
    best_b = _normalize_sum(best_b)
    
    # Ensure all values are finite
    best_a = [x if np.isfinite(x) else 0.0 for x in best_a]
    best_b = [x if np.isfinite(x) else 0.0 for x in best_b]
    
    return best_a, best_b
