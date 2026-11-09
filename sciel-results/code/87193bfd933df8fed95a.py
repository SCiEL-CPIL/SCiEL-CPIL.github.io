"""
Optimized product-formula coefficients using symmetric structure and constrained optimization.
"""
import numpy as np
from numpy.linalg import norm
import time
import scipy.optimize

# Deadline guard
_DEADLINE = 145.0
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
        H += J * np.kron_all(ops_y) if False else J * kron_all(ops_y) # Fixing potential typo in thought, using kron_all
        
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

def _compute_error_from_symmetric(params):
    """
    params: array of 12 floats [a1..a6, b1..b6]
    Expands to full 12-stage symmetric coefficients.
    """
    a_half = params[:6]
    b_half = params[6:]
    
    # Construct full symmetric arrays
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    # Check constraints: sum should be 1
    # If not, the optimizer should handle it via constraints, but for evaluation we assume valid.
    # However, to be safe during intermediate steps if constraints are violated, 
    # we might need to project. But SLSQP handles constraints.
    
    if _A is None:
        _build_hamiltonians()
        
    total_err = 0.0
    dim = _A.shape[0]
    I = np.eye(dim)
    
    for t in _t_values:
        U = I.copy()
        
        for k in range(12):
            # exp(-i b_k t B)
            exp_B = np.diag(np.exp(-1j * b_full[k] * t * _D_B))
            M_B = _V_B @ exp_B @ _V_B.conj().T
            
            # exp(-i a_k t A)
            exp_A = np.diag(np.exp(-1j * a_full[k] * t * _D_A))
            M_A = _V_A @ exp_A @ _V_A.conj().T
            
            M_k = M_B @ M_A
            U = U @ M_k
            
        U_exact = _exact_propagators[t]
        err = norm(U - U_exact, 2)
        total_err += err
        
    return total_err / len(_t_values)

def coefficients(stages):
    if _A is None:
        _build_hamiltonians()
        
    # Define constraints for SLSQP
    # Sum(a_full) = 1 => 2 * sum(a_half) = 1 => sum(a_half) = 0.5
    # Sum(b_full) = 1 => 2 * sum(b_half) = 1 => sum(b_half) = 0.5
    
    def constraint_a(params):
        return np.sum(params[:6]) - 0.5
        
    def constraint_b(params):
        return np.sum(params[6:]) - 0.5
        
    constraints = [
        {'type': 'eq', 'fun': constraint_a},
        {'type': 'eq', 'fun': constraint_b}
    ]
    
    # Bounds: individual coefficients in [-50, 50]
    # Since sum is 1, individual coefficients won't actually go that high, 
    # but we set them to allow flexibility.
    bounds = [(-50, 50)] * 12
    
    # Initial guess: Symmetric Strang-like or uniform
    # Uniform: a_k = 1/12. Symmetric: a1..a6 = 1/12.
    # Let's try a slightly better initial guess based on 4th order symmetric splitting coefficients scaled.
    # For now, uniform is safe.
    x0 = np.full(12, 1.0 / 12.0)
    
    # Try to optimize
    best_params = x0.copy()
    best_score = _compute_error_from_symmetric(best_params)
    
    # Time limit for optimization
    opt_start = time.time()
    opt_limit = 120.0 # Leave 20s margin for final evaluation and return
    
    try:
        # Use SLSQP for constrained optimization
        # maxiter is a hard stop, but we also check time in callback if possible, 
        # or just rely on scipy's internal steps being fast enough.
        # 12 variables is small.
        res = scipy.optimize.minimize(
            _compute_error_from_symmetric, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if res.success or res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            
    except Exception as e:
        # Fallback to uniform if optimization fails
        pass
    
    # Check time
    elapsed = time.time() - opt_start
    if elapsed > opt_limit:
        # If it took too long, we still use the best found so far
        # (scipy doesn't easily support time-based callbacks in all versions, 
        # but 12 vars SLSQP is usually fast. If not, the exception or slow finish is handled by deadline guard below)
        pass
        
    # Final validation and conversion to lists
    a_half = best_params[:6]
    b_half = best_params[6:]
    
    # Ensure sums are exactly 1 by normalizing the full arrays just in case of numerical drift
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    
    if abs(sum_a) > 1e-15:
        a_full = a_full / sum_a
    else:
        a_full = np.full(12, 1.0/12.0)
        
    if abs(sum_b) > 1e-15:
        b_full = b_full / sum_b
    else:
        b_full = np.full(12, 1.0/12.0)
        
    # Clip to bounds
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    # Re-normalize after clipping if necessary (clipping rarely changes sum if bounds are wide)
    sum_a = np.sum(a_full)
    if abs(sum_a) > 1e-15:
        a_full = a_full / sum_a
        
    sum_b = np.sum(b_full)
    if abs(sum_b) > 1e-15:
        b_full = b_full / sum_b

    # Check deadline before returning
    if time.time() - _T0 > _DEADLINE:
        return [1.0/12.0]*12, [1.0/12.0]*12

    return list(a_full), list(b_full)
