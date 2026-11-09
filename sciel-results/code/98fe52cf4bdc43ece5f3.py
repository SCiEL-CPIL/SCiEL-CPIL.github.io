"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain."""
import numpy as np
import time
import sys

_DEADLINE = 140.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def site_op(op, site):
        ops = [I2] * N
        ops[site] = op
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def bond_op(op1, op2, site1, site2):
        ops = [I2] * N
        ops[site1] = op1
        ops[site2] = op2
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(0, N, 2):
        A += bond_op(sx, sx, i, i+1) + bond_op(sy, sy, i, i+1) + delta * bond_op(sz, sz, i, i+1)
    for i in range(N):
        A += 0.5 * field * site_op(sz, i)
    
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(1, N-1, 2):
        B += bond_op(sx, sx, i, i+1) + bond_op(sy, sy, i, i+1) + delta * bond_op(sz, sz, i, i+1)
    for i in range(N):
        B += 0.5 * field * site_op(sz, i)
    
    return A, B

def _precompute_eig(A, B):
    """Precompute eigen-decompositions."""
    w_A, v_A = np.linalg.eigh(A)
    w_B, v_B = np.linalg.eigh(B)
    H_total = A + B
    w_AB, v_AB = np.linalg.eigh(H_total)
    return w_A, v_A, w_B, v_B, w_AB, v_AB

def _expm_herm(w, v, coeff):
    """Compute exp(-i * coeff * H) given eigen-decomp."""
    phases = np.exp(-1j * coeff * w)
    return (v * phases) @ v.conj().T

def _product_formula(a, b, t, eig_data):
    """Compute product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)."""
    w_A, v_A, w_B, v_B, _, _ = eig_data
    n = len(a)
    dim = 2**6
    U = np.eye(dim, dtype=complex)
    for k in range(n):
        U_B = _expm_herm(w_B, v_B, b[k] * t)
        U = U_B @ U
        U_A = _expm_herm(w_A, v_A, a[k] * t)
        U = U_A @ U
    return U

def _exact_propagator(t, eig_data):
    """Compute exact propagator exp(-i t (A+B))."""
    w_AB, v_AB = eig_data[4], eig_data[5]
    return _expm_herm(w_AB, v_AB, t)

def _score(a, b, eig_data):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    errors = []
    for t in times:
        U = _product_formula(a, b, t, eig_data)
        U_exact = _exact_propagator(t, eig_data)
        errors.append(np.linalg.norm(U - U_exact, 2))
    return np.mean(errors)

def _normalize_sum(arr):
    """Normalize array to sum to 1, clipping to [-50, 50]."""
    arr = np.array(arr, dtype=float)
    arr = np.clip(arr, -50, 50)
    s = np.sum(arr)
    if abs(s) > 1e-15:
        arr = arr / s
    else:
        arr = np.ones_like(arr) / len(arr)
    return arr

def _expand_symmetric(params):
    """Expand 12 symmetric params to full 12-element arrays."""
    a_half = params[:6]
    b_half = params[6:12]
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    a_full = _normalize_sum(a_full)
    b_full = _normalize_sum(b_full)
    return a_full, b_full

def _objective(params, eig_data):
    """Objective function for optimization."""
    a_full, b_full = _expand_symmetric(params)
    return _score(a_full, b_full, eig_data)

def _complex_step_gradient(f, x, h=1e-20):
    """Compute gradient using complex-step differentiation."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_c = x.astype(complex).copy()
        x_c[i] += 1j * h
        val = f(x_c)
        grad[i] = np.imag(val)
    return grad

def _gradient_descent(f, x0, max_iter=200, lr_init=0.01, lr_decay=0.95, tol=1e-12):
    """Gradient descent with adaptive learning rate."""
    n = len(x0)
    x = x0.astype(float).copy()
    f_val = f(x)
    best_x = x.copy()
    best_f = f_val
    lr = lr_init
    
    for it in range(max_iter):
        if _remaining() < 8.0:
            break
        
        # Compute gradient
        try:
            grad = _complex_step_gradient(f, x, h=1e-15)
        except:
            # Fallback to finite differences
            h = 1e-7
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Update
        x_new = x - lr * grad
        
        # Evaluate new point
        f_new = f(x_new)
        
        if f_new < f_val:
            x = x_new
            f_val = f_new
            if f_val < best_f:
                best_x = x.copy()
                best_f = f_val
            lr *= 1.05  # Slightly increase if improving
        else:
            lr *= 0.5  # Reduce learning rate if not improving
        
        # Clamp lr
        lr = min(lr, 0.1)
    
    return best_x, best_f

def _random_restart(f, x0, n_restarts=5, max_iter_per_restart=80):
    """Try multiple random restarts around x0."""
    best_x = x0.copy()
    best_f = f(x0)
    
    for r in range(n_restarts):
        if _remaining() < 10.0:
            break
        
        # Perturb initial point
        scale = 0.1 * (r + 1)
        x_perturbed = x0 + scale * np.random.randn(len(x0))
        
        x_try, f_try = _gradient_descent(f, x_perturbed, max_iter=max_iter_per_restart, lr_init=0.005)
        
        if f_try < best_f:
            best_x = x_try
            best_f = f_try
    
    return best_x, best_f

def coefficients(stages):
    """Optimize product formula coefficients using gradient-based method."""
    if stages != 12:
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonians()
    eig_data = _precompute_eig(A, B)
    
    # Define objective with eig_data closure
    def obj(params):
        return _objective(params, eig_data)
    
    # Initial guess: uniform
    x0 = np.ones(12) / 12.0
    
    # Primary optimization run
    x_best, f_best = _gradient_descent(obj, x0, max_iter=150, lr_init=0.01)
    
    # Try random restarts if time permits
    if _remaining() > 20.0:
        x_best2, f_best2 = _random_restart(obj, x_best, n_restarts=3, max_iter_per_restart=50)
        if f_best2 < f_best:
            x_best = x_best2
            f_best = f_best2
    
    # Extract final coefficients
    a_half = x_best[:6]
    b_half = x_best[6:12]
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    a_full = _normalize_sum(a_full)
    b_full = _normalize_sum(b_full)
    
    return a_full.tolist(), b_full.tolist()
