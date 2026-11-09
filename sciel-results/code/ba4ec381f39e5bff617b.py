"""Product-formula coefficient optimizer using Global Stochastic Search (Simulated Annealing)."""
import numpy as np
import time
import sys

# Deadline guard: Evaluator kills at 150s. We target 135s to be safe.
_DEADLINE = 135.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    # Single-site operators
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
    
    def bond_operator(i, j):
        ops_splus_sminus = [identity] * N
        ops_sminus_splus = [identity] * N
        ops_sz_sz = [identity] * N
        
        s_plus = np.array([[0, 1], [0, 0]], dtype=complex)
        s_minus = np.array([[0, 0], [1, 0]], dtype=complex)
        
        ops_splus_sminus[i] = s_plus
        ops_splus_sminus[j] = s_minus
        term1 = 0.5 * kron_all(ops_splus_sminus)
        
        ops_sminus_splus[i] = s_minus
        ops_sminus_splus[j] = s_plus
        term2 = 0.5 * kron_all(ops_sminus_splus)
        
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
    
    # Uniform field: -field * sum sz_j, split half into A, half into B
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops_sz = [identity] * N
        ops_sz[j] = sz
        field_op += kron_all(ops_sz)
    
    A += -0.5 * field * field_op
    B += -0.5 * field * field_op
    
    return A, B

def _compute_error_frobenius(a, b, eig_A, eig_B, eig_H, t_values):
    """Compute mean Frobenius norm error. Much faster than spectral norm (no SVD)."""
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            if abs(a[k]) > 1e-15:
                exp_vec = np.exp(-1j * a[k] * t * eig_A[0])
                U = U @ (eig_A[1] @ (exp_vec[:, None] * eig_A[2]))
            if abs(b[k]) > 1e-15:
                exp_vec = np.exp(-1j * b[k] * t * eig_B[0])
                U = U @ (eig_B[1] @ (exp_vec[:, None] * eig_B[2]))
        
        exp_vec_H = np.exp(-1j * t * eig_H[0])
        U_exact = eig_H[1] @ (exp_vec_H[:, None] * eig_H[2])
        
        diff = U - U_exact
        # Frobenius norm is sqrt(sum(|diff|^2))
        error = np.linalg.norm(diff, 'fro')
        total_error += error
    
    return total_error / n_times

def _compute_error_spectral(a, b, eig_A, eig_B, eig_H, t_values):
    """Compute mean spectral norm error. Expensive (SVD)."""
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            if abs(a[k]) > 1e-15:
                exp_vec = np.exp(-1j * a[k] * t * eig_A[0])
                U = U @ (eig_A[1] @ (exp_vec[:, None] * eig_A[2]))
            if abs(b[k]) > 1e-15:
                exp_vec = np.exp(-1j * b[k] * t * eig_B[0])
                U = U @ (eig_B[1] @ (exp_vec[:, None] * eig_B[2]))
        
        exp_vec_H = np.exp(-1j * t * eig_H[0])
        U_exact = eig_H[1] @ (exp_vec_H[:, None] * eig_H[2])
        
        diff = U - U_exact
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times

def _generate_random_symmetric(start_val=0.0, spread=0.5):
    """Generate random symmetric coefficients for first half."""
    a_half = np.random.uniform(start_val - spread, start_val + spread, 6)
    b_half = np.random.uniform(start_val - spread, start_val + spread, 6)
    # Ensure they don't sum to 0
    if np.sum(a_half) < 0.1: a_half += 0.1
    if np.sum(b_half) < 0.1: b_half += 0.1
    return a_half, b_half

def coefficients(stages):
    stages = 12
    A, B = _build_hamiltonians()
    H_full = A + B
    
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_H, vecs_H = np.linalg.eigh(H_full)
    
    vecs_A_H = vecs_A.conj().T
    vecs_B_H = vecs_B.conj().T
    vecs_H_H = vecs_H.conj().T
    
    eig_A = (vals_A, vecs_A, vecs_A_H)
    eig_B = (vals_B, vecs_B, vecs_B_H)
    eig_H = (vals_H, vecs_H, vecs_H_H)
    
    t_values = [0.25, 0.5, 1.0]
    
    # Initial guess: Suzuki 4th order style
    # c1 = 1/(2 - 2^(1/3))
    z = 1.0 / (2.0 - 2.0**(1.0/3.0))
    w = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    # Symmetric 12-stage init
    a_half_init = np.array([z/4, (z+w)/4, z/4, z/4, (z+w)/4, z/4])
    b_half_init = np.array([w/3, z/3, z/3, z/3, z/3, w/3])
    
    # Normalize
    a_half_init = a_half_init / np.sum(a_half_init) * 0.5
    b_half_init = b_half_init / np.sum(b_half_init) * 0.5
    
    current_a_half = a_half_init.copy()
    current_b_half = b_half_init.copy()
    
    def get_full_vectors(a_h, b_h):
        a_full = np.concatenate([a_h, a_h[::-1]])
        b_full = np.concatenate([b_h, b_h[::-1]])
        a_full = a_full / np.sum(a_full)
        b_full = b_full / np.sum(b_full)
        # Clamp
        a_full = np.clip(a_full, -50, 50)
        b_full = np.clip(b_full, -50, 50)
        return a_full, b_full

    cur_a, cur_b = get_full_vectors(current_a_half, current_b_half)
    cur_err = _compute_error_frobenius(cur_a, cur_b, eig_A, eig_B, eig_H, t_values)
    
    best_a_half = current_a_half.copy()
    best_b_half = current_b_half.copy()
    best_err = cur_err
    
    # Simulated Annealing
    T = 0.5
    T_min = 1e-6
    alpha = 0.995
    
    while _remaining() > 20.0:
        # Perturb
        noise = 0.05 * T
        new_a_half = current_a_half + np.random.normal(0, noise, 6)
        new_b_half = current_b_half + np.random.normal(0, noise, 6)
        
        new_a, new_b = get_full_vectors(new_a_half, new_b_half)
        new_err = _compute_error_frobenius(new_a, new_b, eig_A, eig_B, eig_H, t_values)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_err - cur_err
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_a_half = new_a_half
            current_b_half = new_b_half
            cur_a, cur_b = new_a, new_b
            cur_err = new_err
            
            if cur_err < best_err:
                best_err = cur_err
                best_a_half = current_a_half.copy()
                best_b_half = current_b_half.copy()
        
        T *= alpha
        if T < T_min:
            T = 0.1 # Restart with higher temp occasionally to escape local minima
            alpha = 0.995
    
    # Final refinement using spectral norm on the best candidate
    best_a, best_b = get_full_vectors(best_a_half, best_b_half)
    final_err = _compute_error_spectral(best_a, best_b, eig_A, eig_B, eig_H, t_values)
    
    # Quick local Nelder-Mead on the best candidate using spectral norm
    if _remaining() > 10.0:
        def objective(params):
            a_h = params[:6]
            b_h = params[6:12]
            a_f, b_f = get_full_vectors(a_h, b_h)
            return _compute_error_spectral(a_f, b_f, eig_A, eig_B, eig_H, t_values)
        
        x0 = np.concatenate([best_a_half, best_b_half])
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'fatol': 1e-8, 'xatol': 1e-8})
            if result.fun < final_err:
                final_err = result.fun
                best_a_half = result.x[:6]
                best_b_half = result.x[6:12]
                best_a, best_b = get_full_vectors(best_a_half, best_b_half)
        except:
            pass

    return best_a.tolist(), best_b.tolist()
