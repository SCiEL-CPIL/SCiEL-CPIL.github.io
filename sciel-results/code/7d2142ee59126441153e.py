"""Optimized 12-stage symmetric product formula for 6-site XXZ Heisenberg chain."""
import numpy as np
import time
from itertools import product

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    anisotropy = 0.8
    field = 0.3
    J = 1.0  # coupling constant
    
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    # Field split: half to A, half to B
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Helper to add bond term
    def add_bond(term, sites, mat):
        ops = [I2] * N
        for i, s in enumerate(sites):
            ops[s] = term[i]
        mat += J * kron_all(ops)
    
    # Odd bonds
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    for (i, j) in odd_bonds:
        add_bond([sx, sx], [i, j], A)
        add_bond([sy, sy], [i, j], A)
        add_bond([sz, sz], [i, j], A) * anisotropy  # need to multiply by anisotropy
        # Redo: the above line doesn't multiply, let me fix
        A -= J * anisotropy * kron_all([sz if k == i or k == j else I2 for k in range(N)])
        A += J * anisotropy * kron_all([sz if k == i or k == j else I2 for k in range(N)])
        # Actually let me redo this more carefully
    
    # Let me rebuild from scratch more carefully
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    def add_bond_to(mat, sites, coeff):
        """Add coeff * (sx_i sx_j + sy_i sy_j + anisotropy * sz_i sz_j) to mat."""
        for term in [sx, sy]:
            ops = [I2] * N
            ops[sites[0]] = term
            ops[sites[1]] = term
            mat += coeff * kron_all(ops)
        ops = [I2] * N
        ops[sites[0]] = sz
        ops[sites[1]] = sz
        mat += coeff * anisotropy * kron_all(ops)
    
    # Odd bonds go to A
    for (i, j) in odd_bonds:
        add_bond_to(A, (i, j), J)
    
    # Even bonds go to B
    even_bonds = [(1, 2), (3, 4)]
    for (i, j) in even_bonds:
        add_bond_to(B, (i, j), J)
    
    # Add half the uniform field to each
    # Field term: field * sum_k sz_k / 2 for each half
    # Actually, "half of the uniform field" means:
    # Total field: field * sum_k sz_k
    # Split: A gets field/2 * sum_k sz_k, B gets field/2 * sum_k sz_k
    
    for k in range(N):
        ops = [I2] * N
        ops[k] = sz
        A += (field / 2) * kron_all(ops)
        B += (field / 2) * kron_all(ops)
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error over given times."""
    dim = A.shape[0]
    H = A + B
    total_err = 0.0
    for t in times:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            # Apply exp(-i a_k t A) first (rightmost in product), then exp(-i b_k t B)
            # U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
            # But we want the product in order k=1..12, so:
            # U = exp(-i b_1 t B) exp(-i a_1 t A) @ exp(-i b_2 t B) exp(-i a_2 t A) @ ...
            # So we build from left to right
            if k == 0:
                U = np.linalg.matrix_power(np.eye(dim), 0)  # placeholder
                U = np.eye(dim, dtype=complex)
            # For each k, multiply U on the right by exp(-i b_k t B) then exp(-i a_k t A)
            # Actually, the product is: U = prod_{k=1}^{12} [exp(-i b_k t B) exp(-i a_k t A)]
            # This means: U = M_12 @ M_11 @ ... @ M_1 where M_k = exp(-i b_k t B) exp(-i a_k t A)
            # So we compute U = M_1 @ M_2 @ ... @ M_12
            # Wait, let me re-read: "prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12"
            # This is ambiguous. Typically "in order k=1..12" means:
            # U = [exp(-i b_1 t B) exp(-i a_1 t A)] @ [exp(-i b_2 t B) exp(-i a_2 t A)] @ ... @ [exp(-i b_12 t B) exp(-i a_12 t A)]
            # i.e., left to right, k=1 is leftmost.
            # But in physics, time evolution is often written with the earliest time on the right.
            # Let me assume the standard: U = M_12 ... M_2 M_1 where M_k = exp(-i b_k t B) exp(-i a_k t A)
            # No wait, "in order k=1..12" most naturally means the first factor applied is k=1.
            # In matrix multiplication, if we write U = F_1 F_2 ... F_12, then F_1 is applied first (rightmost in action on a ket if we think of column vectors, but in the matrix product F_1 is leftmost).
            # Actually, for a state |psi>, the evolved state is U|psi> = F_1 F_2 ... F_12 |psi>.
            # So F_12 is applied first to |psi>, then F_11, etc.
            # But the task says "in order k=1..12", which I'll interpret as the matrix product:
            # U = (F_1)(F_2)...(F_12) where F_k = exp(-i b_k t B) exp(-i a_k t A)
            # This means F_1 is the leftmost factor.
            
            # Let me just compute it as: U = identity, then for k=1..12: U = U @ F_k
            # This gives U = F_1 @ F_2 @ ... @ F_12
            # Hmm, but that means F_1 is applied last to a ket.
            # Alternatively: for k=12..1: U = U @ F_k gives U = F_12 @ ... @ F_1
            # I'll go with the literal reading: product in order k=1..12 means
            # U = F_1 F_2 ... F_12 (left to right), so we compute by left-multiplying.
            
            # Let me just do: U = np.eye(dim); for k in range(12): U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
            # This gives U = F_1 @ F_2 @ ... @ F_12
            
            pass  # I'll rewrite this properly below
    
    # Let me redo the loop properly
    total_err = 0.0
    for t in times:
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            # F_k = exp(-i b_k t B) @ exp(-i a_k t A)
            # U = U @ F_k
            eA = np.linalg.expm(-1j * a[k] * t * A)
            eB = np.linalg.expm(-1j * b[k] * t * B)
            Fk = eB @ eA
            U = U @ Fk
        U_exact = np.linalg.expm(-1j * t * H)
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(times)

def _make_symmetric(a_first6, b_first5):
    """Create full 12-element symmetric coefficient lists."""
    a = list(a_first6)
    a.extend(reversed(a_first6))
    # b: 12 elements, symmetric, sum = 1
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # sum = 2*(b1+b2+b3+b4+b5) + b6 = 1
    # b6 = 1 - 2*(b1+b2+b3+b4+b5)
    b6 = 1.0 - 2.0 * sum(b_first5)
    b = list(b_first5)
    b.append(b6)
    b.extend(reversed(b_first5))
    return a, b

def coefficients(stages):
    """Return optimized symmetric 12-stage product formula coefficients."""
    global _A, _B, _times
    if not hasattr(coefficients, '_initialized'):
        _A, _B = _build_hamiltonians()
        _times = [0.25, 0.5, 1.0]
        coefficients._initialized = True
    
    A = _A
    B = _B
    times = _times
    
    # Initial guess: equal weights with symmetry
    # For 12 stages, a_k = 1/12 for all, b_k = 1/12 for all (but b must sum to 1)
    # With symmetry: a = [1/12]*12, b = [1/12]*12
    # a_first6 = [1/12]*6, b_first5 = [1/12]*5, b6 = 1 - 2*5/12 = 1 - 10/12 = 2/12 = 1/6
    # That's not symmetric in the right way. Let me just use equal weights.
    
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = _compute_error(best_a, best_b, A, B, times)
    
    # Multi-start Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        a_first6 = params[:6]
        b_first5 = params[6:11]
        a, b = _make_symmetric(a_first6, b_first5)
        # Check bounds
        if any(abs(x) > 50 for x in a + b):
            return 1e10
        return _compute_error(a, b, A, B, times)
    
    # Initial parameter vector
    p0 = [1.0/12] * 6 + [1.0/12] * 5
    
    # Try multiple starting points
    np.random.seed(42)
    starts = [p0]
    for i in range(20):
        pert = np.random.uniform(-0.1, 0.1, size=11)
        starts.append(np.array(p0) + pert)
    
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_err:
                best_err = res.fun
                a_first6 = res.x[:6]
                b_first5 = res.x[6:11]
                best_a, best_b = _make_symmetric(a_first6, b_first5)
        except Exception:
            continue
    
    # Verify constraints
    if abs(sum(best_a) - 1.0) > 1e-8 or abs(sum(best_b) - 1.0) > 1e-8:
        # Fallback to equal weights
        best_a = [1.0/12] * 12
        best_b = [1.0/12] * 12
    
    # Clamp to [-50, 50]
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    
    return best_a, best_b
