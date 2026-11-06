import time
import numpy as np

_DEADLINE = time.time() + 140  # 10s safety margin

_HAMILTONIANS = None
_EIGS = None

def _init_system():
    global _HAMILTONIANS, _EIGS
    if _HAMILTONIANS is not None:
        return
    
    N = 6
    gamma = 0.8
    h = 0.3
    
    dim = 2 ** N
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
        
    # Field terms
    for j in range(N):
        ops = [ident] * N
        ops[j] = sz
        field_term = (h / 2.0) * kron_all(ops)
        A += field_term
        B += field_term
        
    # Bond terms
    for j in range(N - 1):
        bond_ops = [ident] * N
        bond_ops[j] = sx
        bond_ops[j+1] = sx
        term_xx = kron_all(bond_ops)
        
        bond_ops = [ident] * N
        bond_ops[j] = sy
        bond_ops[j+1] = sy
        term_yy = kron_all(bond_ops)
        
        bond_ops = [ident] * N
        bond_ops[j] = sz
        bond_ops[j+1] = sz
        term_zz = gamma * kron_all(bond_ops)
        
        bond_term = term_xx + term_yy + term_zz
        
        if j % 2 == 0:
            B += bond_term
        else:
            A += bond_term
            
    _HAMILTONIANS = (A, B)
    
    # Pre-compute eigendecompositions
    wA, vA = np.linalg.eigh(A)
    wB, vB = np.linalg.eigh(B)
    H = A + B
    wH, vH = np.linalg.eigh(H)
    
    _EIGS = (wA, vA, wB, vB, wH, vH)

def _fast_mat_exp_diag(w, v, scale):
    """Compute exp(scale * M) given M = V diag(w) V^H"""
    diag_exp = np.exp(scale * w)
    return (v * diag_exp) @ v.conj().T

def _objective_symmetric(params):
    """
    Objective for symmetric 12-stage product formula.
    params: [a1..a5, b1..b5]
    a6 = 1 - sum(a1..a5)
    b6 = 1 - sum(b1..b5)
    Full a = [a1..a5, a6, a6, a5..a1]
    Full b = [b1..b5, b6, b6, b5..b1]
    """
    wA, vA, wB, vB, wH, vH = _EIGS
    
    a_free = params[:5]
    b_free = params[5:10]
    
    a6 = 1.0 - np.sum(a_free)
    b6 = 1.0 - np.sum(b_free)
    
    a_full = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
    b_full = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
    
    # Check bounds
    if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
        return 1e10
        
    t_values = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_values:
        U_exact = _fast_mat_exp_diag(wH, vH, -1j * t)
        
        U_prod = np.eye(64, dtype=complex)
        for k in range(12):
            U_B = _fast_mat_exp_diag(wB, vB, -1j * b_full[k] * t)
            U_A = _fast_mat_exp_diag(wA, vA, -1j * a_full[k] * t)
            U_prod = U_B @ U_A @ U_prod
            
        err = np.linalg.norm(U_prod - U_exact, 2)
        errors.append(err)
        
    return np.mean(errors)

def coefficients(stages):
    _init_system()
    
    # Initial guess: Uniform
    # a_i = 1/12, b_i = 1/12
    # Symmetric params: a1..a5 = 1/12, a6 = 1/12
    x0 = np.ones(10) / 12.0
    
    # Try SciPy L-BFGS-B
    try:
        from scipy.optimize import minimize
        
        deadline = time.time() + 130
        best_x = x0
        best_f = _objective_symmetric(x0)
        
        # Use L-BFGS-B with bounds
        # Bounds for a_i, b_i: [-50, 50]
        bounds = [(-50, 50)] * 10
        
        # We need to check if time is running out during optimization
        # scipy.optimize.minimize doesn't have a built-in deadline, 
        # so we might need to limit maxiter or use a callback.
        
        # For simplicity and safety, let's run a limited number of iterations
        # L-BFGS-B is very fast. 100-200 iterations should be enough.
        
        res = minimize(_objective_symmetric, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success and res.fun < best_f:
            best_x = res.x
            best_f = res.fun
            
    except Exception:
        pass
        
    # If SciPy failed or didn't improve, try Nelder-Mead on symmetric params as fallback
    # But L-BFGS-B should be superior.
    
    a_free = best_x[:5]
    b_free = best_x[5:10]
    
    a6 = 1.0 - np.sum(a_free)
    b6 = 1.0 - np.sum(b_free)
    
    a_full = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
    b_full = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
    
    # Final check on bounds
    if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
        # Fallback to uniform
        return [1.0/12.0]*12, [1.0/12.0]*12
        
    return list(a_full), list(b_full)
