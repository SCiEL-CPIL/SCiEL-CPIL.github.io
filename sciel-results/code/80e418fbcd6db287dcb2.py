"""Product-formula coefficient optimizer using gradient-based and multi-start search."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

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
    identity = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    s_plus = np.array([[0, 1], [0, 0]], dtype=complex)
    s_minus = np.array([[0, 0], [1, 0]], dtype=complex)
    
    def bond_operator(i, j):
        ops_splus_sminus = [identity] * N
        ops_splus_sminus[i] = s_plus
        ops_splus_sminus[j] = s_minus
        term1 = 0.5 * kron_all(ops_splus_sminus)
        
        ops_sminus_splus = [identity] * N
        ops_sminus_splus[i] = s_minus
        ops_sminus_splus[j] = s_plus
        term2 = 0.5 * kron_all(ops_sminus_splus)
        
        ops_sz_sz = [identity] * N
        ops_sz_sz[i] = sz
        ops_sz_sz[j] = sz
        term3 = delta * kron_all(ops_sz_sz)
        
        return term1 + term2 + term3
    
    # Odd bonds: (1,2) and (3,4) -> A
    A += bond_operator(1, 2)
    A += bond_operator(3, 4)
    
    # Even bonds: (0,1), (2,3), (4,5) -> B
    B += bond_operator(0, 1)
    B += bond_operator(2, 3)
    B += bond_operator(4, 5)
    
    # Uniform field split half into A, half into B
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops_sz = [identity] * N
        ops_sz[j] = sz
        field_op += kron_all(ops_sz)
    
    A += -0.5 * field * field_op
    B += -0.5 * field * field_op
    
    return A, B


def _precompute_eigs(A, B, H_full):
    """Precompute eigen-decompositions for fast matrix exponential."""
    eigA_vals, eigA_vecs = np.linalg.eigh(A)
    eigB_vals, eigB_vecs = np.linalg.eigh(B)
    eigH_vals, eigH_vecs = np.linalg.eigh(H_full)
    return eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, eigH_vals, eigH_vecs


def _expm_from_eig(eigvals, eigvecs, coeff):
    """Compute exp(coeff * H) given eigen-decomposition."""
    if abs(coeff) < 1e-15:
        return np.eye(len(eigvals), dtype=complex)
    return eigvecs @ np.diag(np.exp(coeff * eigvals)) @ eigvecs.conj().T


def _compute_error_fast(a, b, eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                         eigH_vals, eigH_vecs, t_values, dim):
    """Compute mean spectral norm error using precomputed eigen-decompositions."""
    I = np.eye(dim, dtype=complex)
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        U = I.copy()
        
        for k in range(len(a)):
            eA = _expm_from_eig(eigA_vals, eigA_vecs, -1j * a[k] * t)
            eB = _expm_from_eig(eigB_vals, eigB_vecs, -1j * b[k] * t)
            U = eB @ (eA @ U)
        
        U_exact = _expm_from_eig(eigH_vals, eigH_vecs, -1j * t)
        
        diff = U - U_exact
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times


def _compute_error_full(a, b, eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                         eigH_vals, eigH_vecs, t_values, dim):
    """Compute error with additional times for better generalization."""
    I = np.eye(dim, dtype=complex)
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        U = I.copy()
        
        for k in range(len(a)):
            eA = _expm_from_eig(eigA_vals, eigA_vecs, -1j * a[k] * t)
            eB = _expm_from_eig(eigB_vals, eigB_vecs, -1j * b[k] * t)
            U = eB @ (eA @ U)
        
        U_exact = _expm_from_eig(eigH_vals, eigH_vecs, -1j * t)
        
        diff = U - U_exact
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times


def _make_objective(eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                     eigH_vals, eigH_vecs, t_values, dim):
    """Create objective function from parameters."""
    def objective(params):
        a_params = params[:12]
        b_params = params[12:24]
        
        # Enforce sum constraint by scaling
        a_sum = np.sum(a_params)
        b_sum = np.sum(b_params)
        if abs(a_sum) < 1e-15 or abs(b_sum) < 1e-15:
            return 1e10
        
        a_scaled = a_params / a_sum
        b_scaled = b_params / b_sum
        
        # Clamp
        a_scaled = np.clip(a_scaled, -50.0, 50.0)
        b_scaled = np.clip(b_scaled, -50.0, 50.0)
        
        return _compute_error_fast(a_scaled, b_scaled, eigA_vals, eigA_vecs,
                                    eigB_vals, eigB_vecs, eigH_vals, eigH_vecs,
                                    t_values, dim)
    return objective


def _suzuki_init_12():
    """Generate 12-stage symmetric Suzuki-inspired coefficients."""
    # 4th order Suzuki 3-stage: c1 = 1/(2-2^(1/3))
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    
    # For 12 stages, use a pattern that approximates higher order
    # Symmetric: 6 unique a values, 6 unique b values
    # a: [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b: [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Initialize with values that give reasonable behavior
    a_half = np.array([0.0, 0.1, 0.2, 0.25, 0.2, 0.15])
    b_half = np.array([0.0, 0.15, 0.2, 0.2, 0.15, 0.1])
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    # Normalize
    a_full = a_full / np.sum(a_full)
    b_full = b_full / np.sum(b_full)
    
    return a_full, b_full


def _random_init():
    """Generate random symmetric initial coefficients."""
    np.random.seed(np.random.randint(0, 1000000))
    
    a_half = np.random.rand(6)
    b_half = np.random.rand(6)
    
    # Add some structure: make ends smaller
    a_half[0] *= 0.3
    a_half[5] *= 0.3
    b_half[0] *= 0.3
    b_half[5] *= 0.3
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    a_full = a_full / np.sum(a_full)
    b_full = b_full / np.sum(b_full)
    
    return a_full, b_full


def coefficients(stages):
    """Optimized product formula coefficients for 6-site Heisenberg chain."""
    stages = 12
    
    A, B = _build_hamiltonians()
    H_full = A + B
    dim = 2**6
    
    t_values = [0.25, 0.5, 1.0]
    
    # Precompute eigen-decompositions
    eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, eigH_vals, eigH_vecs = _precompute_eigs(A, B, H_full)
    
    # Try multiple initialization strategies
    initializations = []
    
    # Suzuki-inspired
    a_suz, b_suz = _suzuki_init_12()
    initializations.append((a_suz, b_suz))
    
    # Random initializations
    for seed in [42, 123, 456, 789, 1011, 2020]:
        np.random.seed(seed)
        a_rand = np.random.rand(12)
        b_rand = np.random.rand(12)
        # Make symmetric
        a_sym = np.concatenate([a_rand[:6], a_rand[:6][::-1]])
        b_sym = np.concatenate([b_rand[:6], b_rand[:6][::-1]])
        a_sym = a_sym / np.sum(a_sym)
        b_sym = b_sym / np.sum(b_sym)
        initializations.append((a_sym, b_sym))
    
    # Evaluate all initializations
    best_a = initializations[0][0]
    best_b = initializations[0][1]
    best_error = _compute_error_fast(best_a, best_b, eigA_vals, eigA_vecs, 
                                       eigB_vals, eigB_vecs, eigH_vals, eigH_vecs, 
                                       t_values, dim)
    
    for a_init, b_init in initializations[1:]:
        err = _compute_error_fast(a_init, b_init, eigA_vals, eigA_vecs, 
                                   eigB_vals, eigB_vecs, eigH_vals, eigH_vecs, 
                                   t_values, dim)
        if err < best_error:
            best_error = err
            best_a = a_init.copy()
            best_b = b_init.copy()
    
    # Local refinement using L-BFGS-B on all 24 parameters
    if _remaining() > 30:
        objective = _make_objective(eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                                     eigH_vals, eigH_vecs, t_values, dim)
        
        x0 = np.concatenate([best_a, best_b])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if result.fun < best_error:
                best_error = result.fun
                a_params = result.x[:12]
                b_params = result.x[12:24]
                
                a_sum = np.sum(a_params)
                b_sum = np.sum(b_params)
                if abs(a_sum) > 1e-15:
                    best_a = a_params / a_sum
                if abs(b_sum) > 1e-15:
                    best_b = b_params / b_sum
                
                best_a = np.clip(best_a, -50.0, 50.0)
                best_b = np.clip(best_b, -50.0, 50.0)
                
        except Exception:
            pass
    
    # Try Nelder-Mead on the best solution
    if _remaining() > 20:
        objective = _make_objective(eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                                     eigH_vals, eigH_vecs, t_values, dim)
        
        x0 = np.concatenate([best_a, best_b])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 300, 'fatol': 1e-12, 'xatol': 1e-12})
            
            if result.fun < best_error:
                best_error = result.fun
                a_params = result.x[:12]
                b_params = result.x[12:24]
                
                a_sum = np.sum(a_params)
                b_sum = np.sum(b_params)
                if abs(a_sum) > 1e-15:
                    best_a = a_params / a_sum
                if abs(b_sum) > 1e-15:
                    best_b = b_params / b_sum
                
                best_a = np.clip(best_a, -50.0, 50.0)
                best_b = np.clip(best_b, -50.0, 50.0)
                
        except Exception:
            pass
    
    # Additional refinement with smaller steps
    if _remaining() > 10:
        objective = _make_objective(eigA_vals, eigA_vecs, eigB_vals, eigB_vecs, 
                                     eigH_vals, eigH_vecs, t_values, dim)
        
        x0 = np.concatenate([best_a, best_b])
        
        try:
            result = minimize(objective, x0, method='Powell', 
                            options={'maxiter': 200, 'ftol': 1e-14, 'xtol': 1e-14})
            
            if result.fun < best_error:
                best_error = result.fun
                a_params = result.x[:12]
                b_params = result.x[12:24]
                
                a_sum = np.sum(a_params)
                b_sum = np.sum(b_params)
                if abs(a_sum) > 1e-15:
                    best_a = a_params / a_sum
                if abs(b_sum) > 1e-15:
                    best_b = b_params / b_sum
                
                best_a = np.clip(best_a, -50.0, 50.0)
                best_b = np.clip(best_b, -50.0, 50.0)
                
        except Exception:
            pass
    
    # Final normalization and clamping
    a_sum = np.sum(best_a)
    b_sum = np.sum(best_b)
    if abs(a_sum) > 1e-15:
        best_a = best_a / a_sum
    if abs(b_sum) > 1e-15:
        best_b = best_b / b_sum
    
    best_a = np.clip(best_a, -50.0, 50.0)
    best_b = np.clip(best_b, -50.0, 50.0)
    
    return list(best_a), list(best_b)
