import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2 ** n
    d = 0.8  # anisotropy
    h = 0.3  # field
    
    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply op on given site, identity elsewhere."""
        ops = [ident] * n
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    def bond_op(i, j):
        """XXZ bond term between sites i and j."""
        sx_i = op_on_site(sx, i)
        sx_j = op_on_site(sx, j)
        sy_i = op_on_site(sy, i)
        sy_j = op_on_site(sy, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return sx_i @ sx_j + sy_i @ sy_j + d * sz_i @ sz_j
    
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i, j in odd_bonds:
        A += bond_op(i, j)
    for i, j in even_bonds:
        B += bond_op(i, j)
    
    # Half field on each
    half_field = 0.5 * h
    for site in range(n):
        A += half_field * op_on_site(sz, site)
        B += half_field * op_on_site(sz, site)
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error."""
    dim = A.shape[0]
    H = A + B
    errors = []
    
    # Precompute exact propagators
    U_exact = {t: expm(-1j * t * H) for t in times}
    
    for t in times:
        U_approx = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            # Order: exp(-i b_k t B) exp(-i a_k t A)
            U_approx = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U_approx
        
        err = np.linalg.norm(U_approx - U_exact[t], 2)
        errors.append(err)
    return np.mean(errors)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    n = stages
    # We assume palindromic symmetry: a[k] = a[n-1-k]
    # So we only need to define a[0..n//2 - 1]
    # Sum constraint: 2 * sum(a[0..n//2 - 1]) = 1
    # Let a_free be the first n//2 - 1 coefficients.
    # a[n//2 - 1] is determined by the sum constraint.
    
    half_n = n // 2
    # Free params for a: a[0] to a[half_n - 2]
    # a[half_n - 1] = 0.5 - sum(a[0] to a[half_n - 2])
    # Same for b.
    
    def unpack(params):
        """Convert 10 free params to full a, b vectors."""
        a_free = params[:half_n - 1]
        b_free = params[half_n - 1:]
        
        a_half = np.zeros(half_n)
        b_half = np.zeros(half_n)
        
        a_half[:-1] = a_free
        a_half[-1] = 0.5 - np.sum(a_free)
        
        b_half[:-1] = b_free
        b_half[-1] = 0.5 - np.sum(b_free)
        
        a = np.zeros(n)
        b = np.zeros(n)
        
        # Fill symmetric parts
        for k in range(half_n):
            a[k] = a_half[k]
            a[n - 1 - k] = a_half[k]
            b[k] = b_half[k]
            b[n - 1 - k] = b_half[k]
            
        return a, b
    
    def objective(params):
        a, b = unpack(params)
        
        # Check bounds [-50, 50]
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
            
        return _compute_error(a, b, A, B, times)
    
    # Initial guess: uniform distribution
    # a_k = 1/12
    # a_free[0..4] = 1/12
    # a_half[5] = 0.5 - 5*(1/12) = 0.5 - 5/12 = 1/12. Correct.
    x0 = np.full(2 * (half_n - 1), 1.0 / n)
    
    best_x = x0.copy()
    best_err = objective(x0)
    
    # Optimization
    # Use Powell method, which is robust for derivative-free optimization
    # and can handle multi-modal landscapes better than Nelder-Mead in some cases.
    
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if i == 0:
            x_start = x0.copy()
        else:
            # Random perturbation
            # Scale perturbation based on the magnitude of coefficients
            # Coeffs are around 1/12 ~ 0.08. Perturb by 0.1
            x_start = x0 + rng.normal(0, 0.1, size=len(x0))
        
        try:
            res = minimize(objective, x_start, method='Powell', 
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass
        
        # If we find a very good solution, we can try to refine it further
        # But Powell is already doing that.
        
    # Final check and unpack
    a, b = unpack(best_x)
    
    # Ensure exact sum is 1 (floating point correction)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-12:
        a = a / sum_a
    if abs(sum_b) > 1e-12:
        b = b / sum_b
        
    # Clamp to [-50, 50]
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-normalize after clamping
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-12:
        a = a / sum_a
    if abs(sum_b) > 1e-12:
        b = b / sum_b
        
    return a.tolist(), b.tolist()
