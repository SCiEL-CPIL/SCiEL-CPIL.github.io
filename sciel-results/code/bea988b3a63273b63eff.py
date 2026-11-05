import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def coefficients(stages):
    """
    Optimized 12-stage product formula for Heisenberg chain.
    Uses a 4th-order symmetric initialization and refines with local optimization.
    """
    if stages != 12:
        # Fallback
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin for 60s budget

    # Build Hamiltonians
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    I2 = np.eye(2)
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    
    def kron_all(*ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
        
    def make_op(op_idx, op):
        ops = [I2] * N
        ops[op_idx] = op
        return kron_all(*ops)
        
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Even bonds: (0,1), (2,3), (4,5)
    for i in [0, 2, 4]:
        if i + 1 < N:
            Sx_i = make_op(i, Sx)
            Sx_ip1 = make_op(i+1, Sx)
            Sy_i = make_op(i, Sy)
            Sy_ip1 = make_op(i+1, Sy)
            Sz_i = make_op(i, Sz)
            Sz_ip1 = make_op(i+1, Sz)
            A += Sx_i @ Sx_ip1 + Sy_i @ Sy_ip1 + anisotropy * Sz_i @ Sz_ip1
            
    # Odd bonds: (1,2), (3,4)
    for i in [1, 3]:
        if i + 1 < N:
            Sx_i = make_op(i, Sx)
            Sx_ip1 = make_op(i+1, Sx)
            Sy_i = make_op(i, Sy)
            Sy_ip1 = make_op(i+1, Sy)
            Sz_i = make_op(i, Sz)
            Sz_ip1 = make_op(i+1, Sz)
            B += Sx_i @ Sx_ip1 + Sy_i @ Sy_ip1 + anisotropy * Sz_i @ Sz_ip1
            
    # Add half field to each
    field_half = field / 2.0
    for i in range(N):
        Sz_i = make_op(i, Sz)
        A += field_half * Sz_i
        B += field_half * Sz_i
        
    H = A + B
    dim = 2**N
    I_dim = np.eye(dim, dtype=complex)
    
    # Pre-compute exact propagators
    times = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in times:
        U_exacts.append(expm(-1j * t * H))
        
    def objective(params):
        if time.time() > deadline:
            return np.inf
            
        a_opt = params[:12]
        b_opt = params[12:24]
        
        # Check bounds
        if np.any(a_opt < -50) or np.any(a_opt > 50) or np.any(b_opt < -50) or np.any(b_opt > 50):
            return 1e10
            
        # Normalize sums to 1
        sum_a = np.sum(a_opt)
        sum_b = np.sum(b_opt)
        if abs(sum_a) < 1e-10 or abs(sum_b) < 1e-10:
            return 1e10
        a_norm = a_opt / sum_a
        b_norm = b_opt / sum_b
        
        errors = []
        for i, t in enumerate(times):
            U = I_dim.copy()
            for k in range(12):
                # Order: exp(-i b_k t B) exp(-i a_k t A)
                # Note: The prompt says U = prod_k exp(-i b_k t B) exp(-i a_k t A)
                # This means for k=1, we apply B_1 then A_1.
                # Matrix multiplication: U_new = exp(-i b_k t B) @ exp(-i a_k t A) @ U_old?
                # Or U_old @ ...?
                # Standard time evolution: U(t) = U_n ... U_1.
                # If we apply operators in order 1 to 12, the rightmost is applied first.
                # Prompt: "U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12"
                # Usually prod_{k=1}^N M_k means M_N ... M_1.
                # But "in order k=1..12" suggests sequential application.
                # Let's assume U = M_12 ... M_1 where M_k = exp(-i b_k t B) exp(-i a_k t A).
                # So U = (exp(-i b_12 t B) exp(-i a_12 t A)) ... (exp(-i b_1 t B) exp(-i a_1 t A))
                
                # Compute exp(-i b_k t B)
                U = expm(-1j * b_norm[k] * t * B) @ U
                U = expm(-1j * a_norm[k] * t * A) @ U
                
            diff = U - U_exacts[i]
            # Spectral norm
            sv = np.linalg.svd(diff, compute_uv=False)
            errors.append(sv[0])
            
        return np.mean(errors)
        
    # Initialization: 4th-order symmetric scheme
    # c1, c2, c3 for 4th order
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -c1 / (2.0 * (2.0**(1.0/3.0) - 1.0))
    c3 = (1.0 - 2.0 * c1) / 2.0
    
    # Pattern for 12 stages (symmetric)
    # a = [c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1] * 0.5
    # This sums to 1.
    base_a = np.array([c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1]) * 0.5
    base_b = base_a.copy()
    
    x0 = np.concatenate([base_a, base_b])
    
    # Optimize
    best_x = x0
    best_score = objective(x0)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 100, 'fatol': 1e-8, 'xatol': 1e-8, 'adaptive': True})
        if result.fun < best_score:
            best_x = result.x
            best_score = result.fun
    except Exception:
        pass
    
    a_final = best_x[:12]
    b_final = best_x[12:24]
    
    # Final normalization check
    sum_a = np.sum(a_final)
    sum_b = np.sum(b_final)
    if sum_a != 0:
        a_final = a_final / sum_a
    if sum_b != 0:
        b_final = b_final / sum_b
        
    return list(a_final), list(b_final)
