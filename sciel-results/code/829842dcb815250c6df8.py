"""Optimized product-formula coefficients for six-site Heisenberg chain splitting."""
import numpy as np
from numpy.linalg import norm
import time

# Deadline guard
_DEADLINE = 135.0  # Safety margin from 150s limit
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _expm_hermitian(H, t):
    """Compute exp(-i t H) for Hermitian H."""
    D, V = np.linalg.eigh(H)
    exp_D = np.exp(-1j * t * D)
    return V @ np.diag(exp_D) @ V.conj().T

# Precomputation variables
_A = None
_B = None
_V_A = None
_D_A = None
_V_B = None
_D_B = None
_exact_propagators = {}
_t_values = [0.25, 0.5, 1.0]

def _build_hamiltonians():
    global _A, _B, _V_A, _D_A, _V_B, _D_B
    if _A is not None:
        return
    
    N = 6
    J = 1.0
    gamma_field = 0.3
    anisotropy = 0.8
    
    # Build Heisenberg XXZ chain Hamiltonian
    # H = sum J (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + anisotropy Sz_i Sz_{i+1}) - gamma_field sum Sz_i
    # Split into even and odd bonds.
    # A = odd bonds + half field
    # B = even bonds + half field
    
    # Spin operators
    sx = np.array([[0, 0.5], [0.5, 0]])
    sy = np.array([[0, -0.5j], [0.5j, 0]])
    sz = np.array([[0.5, 0], [0, -0.5]])
    I = np.eye(2)
    
    def kron_all(ops):
        res = np.array([[1]])
        for op in ops:
            res = np.kron(res, op)
        return res
        
    H_A = np.zeros((2**N, 2**N), dtype=complex)
    H_B = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(N):
        ops = [I] * N
        ops[i] = sz
        field_op = kron_all(ops)
        # Each half gets half the field
        H_A += -0.5 * gamma_field * field_op
        H_B += -0.5 * gamma_field * field_op
        
    def add_bond(H, i, j):
        ops_x = [I] * N
        ops_x[i] = sx
        ops_x[j] = sx
        H += J * kron_all(ops_x)
        
        ops_y = [I] * N
        ops_y[i] = sy
        ops_y[j] = sy
        H += J * kron_all(ops_y)
        
        ops_z = [I] * N
        ops_z[i] = sz
        ops_z[j] = sz
        H += J * anisotropy * kron_all(ops_z)
        
    # A bonds: (0,1), (2,3), (4,5)
    add_bond(H_A, 0, 1)
    add_bond(H_A, 2, 3)
    add_bond(H_A, 4, 5)
    
    # B bonds: (1,2), (3,4)
    add_bond(H_B, 1, 2)
    add_bond(H_B, 3, 4)
    
    _A = H_A
    _B = H_B
    
    # Diagonalize
    _D_A, _V_A = np.linalg.eigh(_A)
    _D_B, _V_B = np.linalg.eigh(_B)
    
    # Precompute exact propagators
    for t in _t_values:
        _exact_propagators[t] = _expm_hermitian(_A + _B, t)

def _normalize_and_clip(x, lower=-50.0, upper=50.0):
    """Clip to bounds and normalize sum to 1."""
    x = np.clip(x, lower, upper)
    s = x.sum()
    if abs(s) < 1e-15:
        # If sum is zero, return uniform
        return np.ones_like(x) / len(x)
    x = x / s
    return x

def _spectral_norm_error(a, b):
    """
    Computes mean spectral norm error for the 3 time points.
    a, b are lists of 12 floats.
    """
    if _A is None:
        _build_hamiltonians()
        
    total_err = 0.0
    dim = _A.shape[0]
    I = np.eye(dim)
    
    for t in _t_values:
        U = I.copy()
        
        for k in range(12):
            # exp(-i b_k t B)
            exp_B = np.diag(np.exp(-1j * b[k] * t * _D_B))
            M_B = _V_B @ exp_B @ _V_B.conj().T
            
            # exp(-i a_k t A)
            exp_A = np.diag(np.exp(-1j * a[k] * t * _D_A))
            M_A = _V_A @ exp_A @ _V_A.conj().T
            
            M_k = M_B @ M_A
            U = U @ M_k
            
        U_exact = _exact_propagators[t]
        err = norm(U - U_exact, 2)
        total_err += err
        
    return total_err / len(_t_values)

def _symmetric_expand(half_a, half_b):
    """
    Expand 6-element arrays to 12-element symmetric arrays.
    half_a: [a1, a2, a3, a4, a5, a6]
    full_a: [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    """
    full_a = np.concatenate([half_a, half_a[::-1]])
    full_b = np.concatenate([half_b, half_b[::-1]])
    return full_a, full_b

def _objective_symmetric(params):
    """
    Objective function for symmetric parameters.
    params: shape (12,) -> [a1..a6, b1..b6]
    """
    half_a = params[:6]
    half_b = params[6:]
    
    # Normalize halves to sum to 0.5 so full sums to 1
    # Clip first
    half_a = np.clip(half_a, -50.0, 50.0)
    half_b = np.clip(half_b, -50.0, 50.0)
    
    sum_a = np.sum(half_a)
    sum_b = np.sum(half_b)
    
    if abs(sum_a) < 1e-15:
        half_a = np.ones(6) / 12.0
    else:
        half_a = half_a * (0.5 / sum_a)
        
    if abs(sum_b) < 1e-15:
        half_b = np.ones(6) / 12.0
    else:
        half_b = half_b * (0.5 / sum_b)
        
    a, b = _symmetric_expand(half_a, half_b)
    
    # Final clip and normalize just in case
    a = _normalize_and_clip(a)
    b = _normalize_and_clip(b)
    
    return _spectral_norm_error(a, b)

def _simplex_search(objective, x0, max_iter=50, tol=1e-6, time_limit=None):
    """
    Simple Nelder-Mead implementation for 12D symmetric parameters.
    """
    n = x0.size
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = []
    fvals = []
    
    x0 = np.array(x0, dtype=float)
    simplex.append(x0.copy())
    fvals.append(objective(x0))
    
    for i in range(n):
        xi = x0.copy()
        step = 0.01
        xi[i] += step
        simplex.append(xi)
        fvals.append(objective(xi))
        
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    best_x = simplex[0].copy()
    best_f = fvals[0]
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - time_limit) > 0:
            break
            
        xw = simplex[-1]
        xc = np.mean(simplex[:-1], axis=0)
        xr = xc + alpha * (xc - xw)
        fr = objective(xr)
        
        if fr < fvals[0]:
            xe = xc + gamma * (xc - xw)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = np.mean(simplex[:-1], axis=0)
                xc = xc + 0.5 * (xr - xc)
                fc = objective(xc)
                if fc < fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            else:
                xc = np.mean(simplex[:-1], axis=0)
                xc = xc + 0.5 * (xc - xw)
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
                        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0].copy()
            
        if max(fvals) - min(fvals) < tol:
            break
            
    return best_x, best_f

def coefficients(stages):
    if _A is None:
        _build_hamiltonians()
        
    # Initial guess: Uniform coefficients (symmetric)
    # a_i = 1/12 for all i. Half sum = 6 * (1/12) = 0.5.
    half_a0 = np.ones(6) / 12.0
    half_b0 = np.ones(6) / 12.0
    x0 = np.concatenate([half_a0, half_b0])
    
    # Compute initial error
    a0_full, b0_full = _symmetric_expand(half_a0, half_b0)
    a0_full = _normalize_and_clip(a0_full)
    b0_full = _normalize_and_clip(b0_full)
    init_err = _spectral_norm_error(a0_full, b0_full)
    
    # Try Symmetric Nelder-Mead optimization if we have time
    if _remaining() > 30:
        try:
            best_x, best_f = _simplex_search(
                _objective_symmetric,
                x0,
                max_iter=100,
                tol=1e-8,
                time_limit=time.time() + _remaining() - 10
            )
            
            # Expand and normalize result
            half_a_opt = best_x[:6]
            half_b_opt = best_x[6:]
            
            # Normalize halves to sum to 0.5
            sum_a = np.sum(half_a_opt)
            sum_b = np.sum(half_b_opt)
            
            if abs(sum_a) < 1e-15:
                half_a_opt = np.ones(6) / 12.0
            else:
                half_a_opt = half_a_opt * (0.5 / sum_a)
                
            if abs(sum_b) < 1e-15:
                half_b_opt = np.ones(6) / 12.0
            else:
                half_b_opt = half_b_opt * (0.5 / sum_b)
                
            a_opt, b_opt = _symmetric_expand(half_a_opt, half_b_opt)
            a_opt = _normalize_and_clip(a_opt)
            b_opt = _normalize_and_clip(b_opt)
            
            # Verify improvement
            if best_f < init_err:
                return a_opt.tolist(), b_opt.tolist()
        except Exception:
            pass
    
    # Fallback to uniform
    return a0_full.tolist(), b0_full.tolist()
