"""Optimized product formula for Heisenberg chain time evolution."""
import time
import numpy as np
from scipy.optimize import minimize

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j, op1, op2):
        ops = [I2] * N
        ops[j] = op1
        ops[j+1] = op2
        return kron_all(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    for j in [0, 2, 4]:
        A += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Even bonds: (1,2), (3,4) -> B
    for j in [1, 3]:
        B += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Uniform field: half in A, half in B
    for j in range(N):
        A += 0.5 * field * op_on_site(j, sz)
        B += 0.5 * field * op_on_site(j, sz)
    
    return A, B


def _precompute_eigh(A, B, H):
    """Precompute eigendecompositions using eigh (Hermitian) for speed and stability."""
    A_evals, A_evecs = np.linalg.eigh(A)
    B_evals, B_evecs = np.linalg.eigh(B)
    H_evals, H_evecs = np.linalg.eigh(H)
    return A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs


def _expm_from_eigh(evals, evecs, scale):
    """Compute exp(scale * M) from precomputed Hermitian eigendecomposition."""
    return (evecs * np.exp(scale * evals)) @ evecs.conj().T


def _objective_fast(a, b, A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs, dim):
    """Fast objective using precomputed Hermitian eigendecompositions."""
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        exact = _expm_from_eigh(H_evals, H_evecs, -1j * t)
        
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            exp_a = _expm_from_eigh(A_evals, A_evecs, -1j * a[k] * t)
            exp_b = _expm_from_eigh(B_evals, B_evecs, -1j * b[k] * t)
            U = exp_b @ exp_a @ U
        
        diff = U - exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0] if len(s) > 0 else 0.0)
    
    return np.mean(errors)


def _get_initial_candidates(stages):
    """Generate initial candidate coefficients."""
    candidates = []
    
    # 1. Equal weights baseline
    eq = np.ones(stages) / stages
    candidates.append((eq.copy(), eq.copy()))
    
    # 2. Symmetric Suzuki-Trotter 4th order (approx)
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    
    # 6-step symmetric 4th order, extended to 12 steps
    a_half = np.array([x3/2, x1/2, x1/2, x3/2, x1/2, x1/2])
    b_half = np.array([x2/2, x2/2, 0.0,  0.0,  x2/2, x2/2])
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    a_full = a_full / np.sum(a_full)
    b_full = b_full / np.sum(b_full)
    candidates.append((a_full.copy(), b_full.copy()))
    
    # 3. Random symmetric perturbations
    np.random.seed(42)
    for _ in range(4):
        a_rand = np.random.uniform(0.05, 0.15, stages)
        a_rand = a_rand / np.sum(a_rand)
        b_rand = np.random.uniform(0.05, 0.15, stages)
        b_rand = b_rand / np.sum(b_rand)
        candidates.append((a_rand.copy(), b_rand.copy()))
        
    return candidates


def coefficients(stages):
    """
    12-stage product formula using L-BFGS-B optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Precompute Hermitian eigendecompositions
    A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs = _precompute_eigh(A, B, H)
    
    # Prepare data for closure
    data = {
        'A_evals': A_evals, 'A_evecs': A_evecs,
        'B_evals': B_evals, 'B_evecs': B_evecs,
        'H_evals': H_evals, 'H_evecs': H_evecs,
        'dim': dim
    }

    def eval_score(a, b):
        """Evaluate with bounds and sum checks."""
        if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
            return 1e10
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        return _objective_fast(a, b, A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs, dim)

    def objective(params):
        """
        Objective function for scipy.optimize.minimize.
        params: 22 floats (11 a_free, 11 b_free)
        """
        a_free = params[:11]
        b_free = params[11:22]
        
        a_last = 1.0 - np.sum(a_free)
        b_last = 1.0 - np.sum(b_free)
        
        a_full = np.concatenate([a_free, [a_last]])
        b_full = np.concatenate([b_free, [b_last]])
        
        # Check bounds for the derived last element
        if a_last < -50 or a_last > 50 or b_last < -50 or b_last > 50:
            return 1e10
            
        # Check bounds for free elements (L-BFGS-B handles this, but just in case)
        if np.any(a_free < -50) or np.any(a_free > 50) or np.any(b_free < -50) or np.any(b_free > 50):
            return 1e10

        return eval_score(a_full, b_full)

    # Generate initial candidates
    candidates = _get_initial_candidates(stages)
    
    best_a = candidates[0][0].copy()
    best_b = candidates[0][1].copy()
    best_score = eval_score(best_a, best_b)
    
    for a_c, b_c in candidates[1:]:
        s = eval_score(a_c, b_c)
        if s < best_score:
            best_a = a_c.copy()
            best_b = b_c.copy()
            best_score = s
            
    # Define bounds for the 22 free parameters
    bounds = [(-50, 50)] * 22
    
    # Multi-start L-BFGS-B
    np.random.seed(123)
    
    # Start with the best heuristic candidate
    start_params_list = []
    start_params_list.append(np.concatenate([best_a[:11], best_b[:11]]))
    
    # Add random starts
    for _ in range(9):
        if time.time() > deadline - 20:
            break
        perturb = np.random.normal(0, 0.1, 22)
        start_params_list.append(np.concatenate([best_a[:11], best_b[:11]]) + perturb)
    
    for i, x0 in enumerate(start_params_list):
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                params = res.x
                a_free = params[:11]
                b_free = params[11:22]
                a_last = 1.0 - np.sum(a_free)
                b_last = 1.0 - np.sum(b_free)
                
                a_full = np.concatenate([a_free, [a_last]])
                b_full = np.concatenate([b_free, [b_last]])
                
                # Final validity check
                if -50 <= a_last <= 50 and -50 <= b_last <= 50:
                    score = res.fun
                    if score < best_score:
                        best_a = a_full.copy()
                        best_b = b_full.copy()
                        best_score = score
                        
        except Exception:
            pass

    # Final cleanup
    best_a = np.clip(best_a, -50, 50)
    best_b = np.clip(best_b, -50, 50)
    best_a = best_a + (1.0 - np.sum(best_a)) / stages
    best_b = best_b + (1.0 - np.sum(best_b)) / stages
    
    if np.any(best_a < -50) or np.any(best_a > 50) or np.any(best_b < -50) or np.any(best_b > 50):
        best_a = np.ones(stages) / stages
        best_b = np.ones(stages) / stages
    
    return list(best_a), list(best_b)
