"""Optimized 12-stage product formula for 6-site Heisenberg chain time evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonians():
    """Build the odd-bond (A) and even-bond (B) Hamiltonians for 6-site XXZ chain."""
    N = 6
    d = 2
    dim = d ** N  # 64
    delta = 0.8
    h = 0.3
    
    # Build Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    # Helper: build operator acting on sites i and j (0-indexed)
    def two_site_op(op1, op2, site1, site2):
        """Build op1 ⊗ op2 on sites site1, site2, identity elsewhere."""
        ops = [I2] * N
        ops[site1] = op1
        ops[site2] = op2
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B (bond (5,0) doesn't exist for open chain)
    # Wait, for 6 sites open chain: bonds are (0,1),(1,2),(2,3),(3,4),(4,5)
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    for (i, j) in odd_bonds:
        # XXZ term: (sx⊗sx + sy⊗sy)/2 + delta * sz⊗sz
        # Actually standard XXZ: (sx⊗sx + sy⊗sy + delta*sz⊗sz)/2 or without /2
        # Let's use: H_bond = (sx⊗sx + sy⊗sy)/2 + delta*sz⊗sz
        # Plus field: -h/2 * (sz on site i + sz on site j) for each bond's sites
        # But field is uniform, so each site gets -h/2*sz, split between A and B
        # Each site appears in exactly one odd bond and (for interior) one even bond
        # Site 0: only in odd bond (0,1)
        # Site 1: in odd (0,1) and even (1,2)
        # Site 2: in even (1,2) and odd (2,3)
        # Site 3: in odd (2,3) and even (3,4)
        # Site 4: in even (3,4) and odd (4,5)
        # Site 5: only in odd (4,5)
        
        bond_term = (two_site_op(sx, sx, i, j) + two_site_op(sy, sy, i, j)) / 2.0 + delta * two_site_op(sz, sz, i, j)
        A += bond_term
        # Field contribution: each site in this bond gets -h/2 * sz
        # But we need to be careful: the total field is -h * sum sz_k
        # Split: A gets field from odd-bond sites, B gets field from even-bond sites
        # Actually, let's assign: for each bond in A, add -h/2*(sz_i + sz_j) to A
        # For each bond in B, add -h/2*(sz_i + sz_j) to B
        # This way each interior site gets -h/2 from A and -h/2 from B = -h total
        # End sites get -h/2 from A only... that's wrong.
        
    # Let me reconsider. The problem says "each including half of the uniform field"
    # So A includes half the field, B includes half the field.
    # Total field: -h * sum_{k=0}^{5} sz_k
    # A gets: -h/2 * sum_{k=0}^{5} sz_k  (all sites)
    # B gets: -h/2 * sum_{k=0}^{5} sz_k  (all sites)
    
    # Let me redo: build A and B with just bond terms, then add half field to each
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for (i, j) in odd_bonds:
        bond_term = (two_site_op(sx, sx, i, j) + two_site_op(sy, sy, i, j)) / 2.0 + delta * two_site_op(sz, sz, i, j)
        A += bond_term
    
    for (i, j) in even_bonds:
        bond_term = (two_site_op(sx, sx, i, j) + two_site_op(sy, sy, i, j)) / 2.0 + delta * two_site_op(sz, sz, i, j)
        B += bond_term
    
    # Add half the uniform field to each
    field_total = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        field_total += two_site_op(sz, I2, k, k)  # This doesn't work, let me fix
    
    # Better: build single-site operator
    def single_site_op(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    field_total = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        field_total += single_site_op(sz, k)
    
    A -= (h / 2.0) * field_total
    B -= (h / 2.0) * field_total
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error for given coefficients."""
    dim = A.shape[0]
    exact = np.eye(dim, dtype=complex)
    # Precompute exact propagators
    eigvals, eigvecs = np.linalg.eigh(A + B)
    exact_prop = []
    for t in times:
        exact_prop.append(eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T)
    
    errors = []
    for t in times:
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            # exp(-i * b_k * t * B)
            if abs(b[k]) > 1e-15:
                eigv_B, eigvec_B = np.linalg.eigh(B)
                expB = eigvec_B @ np.diag(np.exp(-1j * b[k] * t * eigv_B)) @ eigvec_B.conj().T
                U = U @ expB
            if abs(a[k]) > 1e-15:
                eigv_A, eigvec_A = np.linalg.eigh(A)
                expA = eigvec_A @ np.diag(np.exp(-1j * a[k] * t * eigv_A)) @ eigvec_A.conj().T
                U = U @ expA
        err = np.linalg.norm(U - exact_prop[times.index(t)], 2)
        errors.append(err)
    return np.mean(errors)

def _objective(params, A, B, times):
    """Objective function for optimization. params = [a_1..a_12, b_1..b_12]"""
    a = params[:12]
    b = params[12:]
    # Enforce sum constraints via penalty
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    penalty = 100.0 * (sum_a - 1.0)**2 + 100.0 * (sum_b - 1.0)**2
    
    # Clip to valid range
    a_clip = np.clip(a, -50.0, 50.0)
    b_clip = np.clip(b, -50.0, 50.0)
    
    err = _compute_error(a_clip, b_clip, A, times)
    return err + penalty

def coefficients(stages):
    """Optimized 12-stage product formula coefficients."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Start with a reasonable initial guess
    # Use symmetric pattern: a = [x, y, z, z, y, x, x, y, z, z, y, x], b similar
    # Or start from the seed: Strang-like
    # For 12 stages with sum=1: try uniform first
    a0 = [1.0/12.0] * 12
    b0 = [1.0/12.0] * 12
    params0 = np.array(a0 + b0)
    
    best_params = params0.copy()
    best_err = _compute_error(a0, b0, A, times)
    
    # Try a few random restarts and Nelder-Mead optimization
    rng = np.random.RandomState(42)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if r == 0:
            params = params0.copy()
        else:
            # Random perturbation
            params = params0 + rng.randn(24) * 0.1
            # Ensure positive-ish sums
            params[:12] = np.clip(params[:12], -10, 10)
            params[12:] = np.clip(params[12:], -10, 10)
        
        # Use Nelder-Mead
        try:
            result = minimize(
                _objective, params,
                args=(A, B, times),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_err:
                best_params = result.x.copy()
                best_err = result.fun
        except Exception:
            pass
        
        if time.time() > deadline - 30.0:
            break
    
    # Extract and normalize
    a = best_params[:12]
    b = best_params[12:]
    
    # Normalize to ensure sum = 1
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-10:
        a = a / sum_a
    else:
        a = np.ones(12) / 12.0
    if abs(sum_b) > 1e-10:
        b = b / sum_b
    else:
        b = np.ones(12) / 12.0
    
    # Clip to valid range
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    # Re-normalize after clipping
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-10:
        a = a / sum_a
    if abs(sum_b) > 1e-10:
        b = b / sum_b
    
    # Final check
    a = list(np.real(a))
    b = list(np.real(b))
    
    return a, b
# EVOLVE-BLOCK-END
