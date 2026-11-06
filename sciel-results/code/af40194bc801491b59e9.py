"""
Optimized product formula for Heisenberg-chain time evolution.
Uses symmetric parameterized families with reduced free parameters,
optimized via coordinate descent with line search.
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


# Precompute eigendecompositions for fast evaluation
_A_eig = None
_B_eig = None
_H_eig = None
_A_mat = None
_B_mat = None
_H_mat = None

def _ensure_precomputed():
    global _A_eig, _B_eig, _H_eig, _A_mat, _B_mat, _H_mat
    if _A_eig is None:
        _A_mat, _B_mat = _build_hamiltonians()
        _H_mat = _A_mat + _B_mat
        # Use eigh for Hermitian matrices (faster and more stable)
        _A_eig = np.linalg.eigh(_A_mat)
        _B_eig = np.linalg.eigh(_B_mat)
        _H_eig = np.linalg.eigh(_H_mat)


def _expm_from_eigh(eigvals, eigvecs, factor):
    """Compute exp(factor * H) from precomputed eigendecomposition."""
    # eigvecs @ diag(exp(factor * eigvals)) @ eigvecs^H
    phase = np.exp(factor * eigvals)
    return (eigvecs * phase) @ eigvecs.conj().T


def _objective_fast(a, b):
    """
    Fast objective: mean spectral norm error over t = 0.25, 0.5, 1.0.
    a, b: lists of 12 floats each, summing to 1.
    """
    _ensure_precomputed()
    
    if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
        return 1e10
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    dim = 64
    
    for t in t_vals:
        # Exact propagator
        exact = _expm_from_eigh(_H_eig[0], _H_eig[1], -1j * t)
        
        # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # exp(-i a_k t A)
            exp_a = _expm_from_eigh(_A_eig[0], _A_eig[1], -1j * a[k] * t)
            # exp(-i b_k t B)
            exp_b = _expm_from_eigh(_B_eig[0], _B_eig[1], -1j * b[k] * t)
            U = exp_b @ exp_a @ U
        
        err = np.linalg.svd(U - exact, compute_uv=False)[0]
        errors.append(err)
    
    return np.mean(errors)


def _symmetric_12stage(params):
    """
    Generate a and b coefficients from a reduced parameter set.
    
    Symmetric pattern for 12 stages (palindromic):
    The sequence is symmetric: stage k mirrors stage 13-k.
    
    Pattern: A, B, A, B, A, B, A, B, A, B, A, B
    With symmetry: a[0]=a[11], a[1]=a[10], ..., a[5]=a[6]
                   b[0]=b[11], b[1]=b[10], ..., b[5]=b[6]
    
    params: [a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5]
    where a = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
    and b = [b0, b1, b2, b3, b4, b5, b5, b4, b3, b2, b1, b0]
    
    Then normalize so sum(a)=1, sum(b)=1.
    """
    a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5 = params
    
    a = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
    b = [b0, b1, b2, b3, b4, b5, b5, b4, b3, b2, b1, b0]
    
    # Normalize
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-10:
        a = [v / sum_a for v in a]
    if abs(sum_b) > 1e-10:
        b = [v / sum_b for v in b]
    
    return a, b


def _yoshida_4th_order():
    """Return Yoshida 4th-order coefficients mapped to 12-stage symmetric form."""
    c = 2.0 ** (1.0/3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - x1
    
    # 4th order symmetric: A(x3), B(x2), A(x1), B(x2), A(x3)
    # For 12 stages with pattern A,B,A,B,A,B,A,B,A,B,A,B:
    # We need to distribute. Let's use the structure:
    # a = [x3/2, x1/2, x1/2, x3/2, x3/2, x1/2, x1/2, x3/2, x3/2, x1/2, x1/2, x3/2]
    # But that's not symmetric in the right way.
    # 
    # Better: use the 5-stage 4th order pattern and repeat/pad
    # Actually, let's just use equal weights as starting point and let optimization handle it.
    # Or use a known good symmetric pattern.
    
    # Simple symmetric 4th-order-like:
    # a = [0.3, 0.1, 0.1, 0.1, 0.1, 0.3] (symmetric, sums to 1.0 for the 6 unique)
    # b = [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
    
    # Let's use the Yoshida values properly:
    # For a 12-stage symmetric scheme, we can use:
    # a_params = [x3/2, x1/2, x1/2, x3/2, 0, 0]
    # b_params = [x2/2, x2/2, 0, 0, 0, 0]
    # But this gives sum_a = x3 + x1 = 1, sum_b = x2 which is negative...
    
    # Just use a reasonable starting point:
    params = np.array([
        0.3, 0.15, 0.15, 0.15, 0.15, 0.10,  # a params
        0.2, 0.15, 0.15, 0.15, 0.15, 0.20   # b params
    ])
    return params


def _coordinate_descent(obj, x0, max_iter=500, timeout=None, start_time=None, step_init=0.1):
    """
    Coordinate descent with line search for the reduced parameter space.
    """
    if start_time is None:
        start_time = time.time()
    
    x = np.array(x0, dtype=float)
    n = len(x)
    f_best = obj(x)
    best_x = x.copy()
    step = step_init
    
    for iteration in range(max_iter):
        if timeout is not None and (time.time() - start_time) > timeout:
            break
        
        improved = False
        for i in range(n):
            if timeout is not None and (time.time() - start_time) > timeout:
                break
            
            # Try positive and negative steps
            for direction in [1, -1]:
                x_try = x.copy()
                x_try[i] += direction * step
                f_try = obj(x_try)
                
                if f_try < f_best:
                    x = x_try
                    f_best = f_try
                    best_x = x.copy()
                    improved = True
                    # Shrink step slightly
                    step *= 0.9
                else:
                    # Try smaller step
                    for sub in [0.5, 0.25, 0.1]:
                        x_try2 = x.copy()
                        x_try2[i] += direction * step * sub
                        f_try2 = obj(x_try2)
                        if f_try2 < f_best:
                            x = x_try2
                            f_best = f_try2
                            best_x = x.copy()
                            improved = True
                            break
                    else:
                        # No improvement, shrink step
                        step *= 0.8
                        break
            
            if timeout is not None and (time.time() - start_time) > timeout:
                break
        
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
    
    return best_x, f_best


def coefficients(stages):
    """
    Return 12-stage product formula coefficients (a, b) that minimize
    the mean spectral-norm error of the time evolution approximation.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Precompute
    _ensure_precomputed()
    
    # Define objective wrapper for the parameterized family
    def obj(params):
        a, b = _symmetric_12stage(params)
        return _objective_fast(a, b)
    
    # Starting point: Yoshida-like or equal weights
    x0 = np.array([
        1.0/12, 1.0/12, 1.0/12, 1.0/12, 1.0/12, 1.0/12,  # a params
        1.0/12, 1.0/12, 1.0/12, 1.0/12, 1.0/12, 1.0/12   # b params
    ])
    
    # Also try Yoshida-inspired starting point
    c = 2.0 ** (1.0/3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - x1
    
    # For symmetric 12-stage: a = [a0,a1,a2,a3,a4,a5,a5,a4,a3,a2,a1,a0]
    # Sum = 2*(a0+a1+a2+a3+a4+a5) = 1, so a0+...+a5 = 0.5
    # Similarly for b.
    # Yoshida 4th order: A weights [x3, x1, x3], B weights [x2, x2]
    # Map to 6 unique a values: distribute x3, x1, x3 over 6 slots
    x0_yoshida = np.array([
        x3/2, x1/2, 0, 0, 0, x3/2,  # a: sum = x3 + x1/2... not quite
        x2/2, x2/2, 0, 0, 0, 0      # b: sum = x2
    ])
    # Normalize in _symmetric_12stage
    
    # Evaluate both starting points
    f_equal = obj(x0)
    f_yoshida = obj(x0_yoshida)
    
    if f_yoshida < f_equal:
        x_start = x0_yoshida
        f_start = f_yoshida
    else:
        x_start = x0
        f_start = f_equal
    
    # Run coordinate descent
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        x_best, f_best = _coordinate_descent(
            obj, x_start, max_iter=300,
            timeout=remaining, start_time=start_time, step_init=0.05
        )
    else:
        x_best = x_start
        f_best = f_start
    
    # Also try a few random perturbations for diversification
    if time.time() < deadline - 15.0:
        np.random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
            x_pert = x_best + np.random.randn(len(x_best)) * 0.03
            x_try, f_try = _coordinate_descent(
                obj, x_pert, max_iter=100,
                timeout=min(10.0, deadline - time.time() - 5.0),
                start_time=start_time, step_init=0.03
            )
            if f_try < f_best:
                x_best = x_try
                f_best = f_try
    
    a_opt, b_opt = _symmetric_12stage(x_best)
    
    # Clamp to [-50, 50]
    a_opt = [max(-50.0, min(50.0, v)) for v in a_opt]
    b_opt = [max(-50.0, min(50.0, v)) for v in b_opt]
    
    # Final normalization
    sum_a = sum(a_opt)
    sum_b = sum(b_opt)
    if abs(sum_a) > 1e-10:
        a_opt = [v / sum_a for v in a_opt]
    if abs(sum_b) > 1e-10:
        b_opt = [v / sum_b for v in b_opt]
    
    return a_opt, b_opt

# EVOLVE-BLOCK-END
