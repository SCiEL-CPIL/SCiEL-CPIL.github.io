"""
Optimized product formula for Heisenberg-chain time evolution.
Uses symmetric 12-stage pattern with golden-section search over few parameters.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half uniform field) and B (even bonds + half uniform field)
    for a 6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
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
    
    def op_on_site(site, op, N):
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
        A += 0.5 * field * op_on_site(j, sz, N)
        B += 0.5 * field * op_on_site(j, sz, N)
    
    return A, B


def _expm_hermitian(H, t):
    """Compute exp(-i*t*H) for Hermitian H using eigendecomposition."""
    try:
        evals, evecs = np.linalg.eigh(H)
        return evecs @ np.diag(np.exp(-1j * t * evals)) @ evecs.conj().T
    except np.linalg.LinAlgError:
        return None


def _make_symmetric_12(a1, a2, a3, b1, b2):
    """
    Create symmetric 12-stage coefficients from 5 parameters.
    Pattern: A,B,A,B,A,B,A,B,A,B,A,B
    Symmetric: a = [a1, a2, a3, a3, a2, a1, a1, a2, a3, a3, a2, a1]
    b = [b1, b2, b2, b2, b2, b2, b2, b2, b2, b2, b2, b1]
    Wait, that's not right. Let me think more carefully.
    
    For a symmetric 12-stage product formula U = prod_k exp(-i b_k t B) exp(-i a_k t A),
    symmetry means U(t) = U(-t)^† which requires:
    a_k = a_{13-k} and b_k = b_{13-k}
    
    So: a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    That's 6+6=12 parameters. Still too many.
    
    Let me use a simpler approach: use the Strang splitting repeated 6 times,
    but with a single free parameter that modifies the weights.
    
    Actually, let me use a 3-parameter symmetric pattern:
    a = [x, y, z, z, y, x, x, y, z, z, y, x]
    b = [w, v, v, v, v, v, v, v, v, v, v, w]
    
    With constraints: 2(x+y+z) = 1, 2w + 10v = 1
    So: x+y+z = 0.5, w + 5v = 0.5
    Free parameters: x, y, z (with z = 0.5 - x - y), and v (with w = 0.5 - 5v)
    That's 3 free parameters: x, y, v
    """
    # a = [x, y, z, z, y, x, x, y, z, z, y, x] where z = 0.5 - x - y
    z = 0.5 - a1 - a2
    a = [a1, a2, z, z, a2, a1, a1, a2, z, z, a2, a1]
    
    # b = [w, v, v, v, v, v, v, v, v, v, v, w] where w = 0.5 - 5v
    w = 0.5 - 5 * b1
    b = [w, b1, b1, b1, b1, b1, b1, b1, b1, b1, b1, w]
    
    return np.array(a), np.array(b)


def _objective_3param(params, A, B, H, dim):
    """
    Objective function for 3 free parameters: (x, y, v)
    """
    x, y, v = params
    
    # Check bounds for all coefficients
    z = 0.5 - x - y
    w = 0.5 - 5 * v
    
    a = [x, y, z, z, y, x, x, y, z, z, y, x]
    b = [w, v, v, v, v, v, v, v, v, v, v, w]
    
    if any(v < -50 or v > 50 for v in a + b):
        return 1e10
    
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        exact = _expm_hermitian(H, t)
        if exact is None:
            return 1e10
        
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            exp_a = _expm_hermitian(A, a[k] * t)
            if exp_a is None:
                return 1e10
            exp_b = _expm_hermitian(B, b[k] * t)
            if exp_b is None:
                return 1e10
            U = exp_b @ exp_a @ U
        
        err = np.linalg.svd(U - exact, compute_uv=False)[0]
        errors.append(err)
    
    return np.mean(errors)


def _golden_section_search(f, x_lo, x_hi, max_iter=50, timeout=None, start_time=None):
    """Golden-section search for 1D minimization."""
    if start_time is None:
        start_time = time.time()
    
    gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
    
    x1 = x_hi - gr * (x_hi - x_lo)
    x2 = x_lo + gr * (x_hi - x_lo)
    f1 = f(x1)
    f2 = f(x2)
    
    for i in range(max_iter):
        if timeout is not None and (time.time() - start_time) > timeout:
            break
        
        if f1 < f2:
            x_hi = x2
            x2 = x1
            f2 = f1
            x1 = x_hi - gr * (x_hi - x_lo)
            f1 = f(x1)
        else:
            x_lo = x1
            x1 = x2
            f1 = f2
            x2 = x_lo + gr * (x_hi - x_lo)
            f2 = f(x2)
    
    x_opt = (x_lo + x_hi) / 2
    f_opt = min(f1, f2)
    return x_opt, f_opt


def coefficients(stages):
    """
    Return 12-stage product formula coefficients (a, b) that minimize
    the mean spectral-norm error of the time evolution approximation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = 2 ** 6
    
    # Initial guess: Strang splitting (x=0.25, y=0.25, v=0.1)
    # For Strang: a = [0.25, 0.25, 0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0.25, 0.25]
    # b = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0]
    # So x=0.25, y=0.25, z=0, v=0.1, w=0
    
    # Try a few different starting points
    candidates = [
        (0.25, 0.25, 0.1),   # Strang-like
        (0.3, 0.2, 0.1),
        (0.2, 0.3, 0.1),
        (0.35, 0.15, 0.1),
        (0.15, 0.35, 0.1),
        (0.25, 0.25, 0.05),
        (0.25, 0.25, 0.15),
        (0.4, 0.1, 0.1),
        (0.1, 0.4, 0.1),
    ]
    
    best_params = None
    best_score = 1e10
    
    # Evaluate all candidates
    for params in candidates:
        score = _objective_3param(params, A, B, H, dim)
        if score < best_score:
            best_score = score
            best_params = params
    
    # Now do local refinement around the best candidate using golden-section search
    # on each parameter
    if time.time() < deadline - 30:
        # Refine x
        def obj_x(x):
            return _objective_3param((x, best_params[1], best_params[2]), A, B, H, dim)
        
        x_lo = max(-50, best_params[0] - 0.15)
        x_hi = min(50, best_params[0] + 0.15)
        x_opt, _ = _golden_section_search(obj_x, x_lo, x_hi, max_iter=20, 
                                           timeout=deadline - 5, start_time=start_time)
        best_params = (x_opt, best_params[1], best_params[2])
        
    if time.time() < deadline - 20:
        # Refine y
        def obj_y(y):
            return _objective_3param((best_params[0], y, best_params[2]), A, B, H, dim)
        
        y_lo = max(-50, best_params[1] - 0.15)
        y_hi = min(50, best_params[1] + 0.15)
        y_opt, _ = _golden_section_search(obj_y, y_lo, y_hi, max_iter=20,
                                           timeout=deadline - 5, start_time=start_time)
        best_params = (best_params[0], y_opt, best_params[2])
        
    if time.time() < deadline - 10:
        # Refine v
        def obj_v(v):
            return _objective_3param((best_params[0], best_params[1], v), A, B, H, dim)
        
        v_lo = max(-50, best_params[2] - 0.15)
        v_hi = min(50, best_params[2] + 0.15)
        v_opt, _ = _golden_section_search(obj_v, v_lo, v_hi, max_iter=20,
                                           timeout=deadline - 5, start_time=start_time)
        best_params = (best_params[0], best_params[1], v_opt)
    
    # Generate final coefficients
    a, b = _make_symmetric_12(best_params[0], best_params[1], best_params[2], 
                               0, best_params[2])
    
    # Clamp to [-50, 50]
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Normalize to ensure sum = 1
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-10:
        a = a / sum_a
    if abs(sum_b) > 1e-10:
        b = b / sum_b
    
    return a.tolist(), b.tolist()

# EVOLVE-BLOCK-END
