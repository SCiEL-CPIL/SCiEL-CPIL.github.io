import numpy as np
import scipy.linalg
import scipy.optimize
import time

def _build_hamiltonians():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id = np.eye(2, dtype=complex)
    
    N = 6
    gamma = 0.8
    h_field = 0.3
    
    def apply_op(op, i):
        ops = [id] * N
        ops[i] = op
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    def apply_bond(op1, op2, i, j):
        ops = [id] * N
        ops[i] = op1
        ops[j] = op2
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    H_odd = np.zeros((2**N, 2**N), dtype=complex)
    H_even = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(N-1):
        term = apply_bond(sx, sx, i, i+1) + apply_bond(sy, sy, i, i+1) + gamma * apply_bond(sz, sz, i, i+1)
        if i % 2 == 0:
            H_odd += term
        else:
            H_even += term
            
    H_field = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        H_field += apply_op(sz, i)
        
    A = H_odd + 0.5 * h_field * H_field
    B = H_even + 0.5 * h_field * H_field
    
    return A, B

def coefficients(stages):
    start_time = time.time()
    A, B = _build_hamiltonians()
    
    # Initial guess: 4th-order Suzuki repeated 4 times
    c_root = 2.0 ** (1.0/3.0)
    c1 = 1.0 / (2.0 * (2.0 - c_root))
    c3 = c1
    c2 = 1.0 - 2.0 * c1
    d1 = 1.0 / (2.0 * (c_root - 1.0))
    d2 = 1.0 - 2.0 * d1
    
    a_local = [c1, c2, c3]
    b_local = [0.0, d1, d2]
    
    a_init = []
    b_init = []
    for _ in range(stages // 3):
        a_init.extend([x * (3.0/stages) for x in a_local])
        b_init.extend([x * (3.0/stages) for x in b_local])
        
    # If stages is not multiple of 3, pad with uniform
    if len(a_init) < stages:
        rem = stages - len(a_init)
        a_init.extend([1.0/stages] * rem)
        b_init.extend([1.0/stages] * rem)
        
    a_init = np.array(a_init[:stages])
    b_init = np.array(b_init[:stages])
    
    def error(params):
        a_vars = params[:stages-1]
        b_vars = params[stages-1:]
        
        a = np.append(a_vars, 1.0 - np.sum(a_vars))
        b = np.append(b_vars, 1.0 - np.sum(b_vars))
        
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
            
        total_err = 0.0
        times = [0.25, 0.5, 1.0]
        
        for t in times:
            U_prod = np.eye(2**6, dtype=complex)
            for k in range(stages):
                U_prod = U_prod @ scipy.linalg.expm(-1j * b[k] * t * B)
                U_prod = U_prod @ scipy.linalg.expm(-1j * a[k] * t * A)
            
            U_exact = scipy.linalg.expm(-1j * t * (A + B))
            
            err = np.linalg.norm(U_prod - U_exact, 2)
            total_err += err**2
            
        return total_err / len(times)

    x0 = np.concatenate([a_init[:-1], b_init[:-1]])
    
    best_params = x0
    best_err = error(x0)
    
    try:
        res = scipy.optimize.minimize(error, x0, method='Nelder-Mead', 
                                      options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 300})
        
        if time.time() - start_time < 50:
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
    except Exception:
        pass
        
    a_vars = best_params[:stages-1]
    b_vars = best_params[stages-1:]
    
    a_final = np.append(a_vars, 1.0 - np.sum(a_vars))
    b_final = np.append(b_vars, 1.0 - np.sum(b_vars))
    
    a_final = np.clip(a_final, -50, 50)
    b_final = np.clip(b_final, -50, 50)
    
    # Re-normalize if clipped
    sum_a = np.sum(a_final)
    sum_b = np.sum(b_final)
    if abs(sum_a - 1.0) > 1e-9:
        a_final = a_final / sum_a
    if abs(sum_b - 1.0) > 1e-9:
        b_final = b_final / sum_b
        
    return a_final.tolist(), b_final.tolist()
