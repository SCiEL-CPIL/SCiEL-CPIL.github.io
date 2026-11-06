import numpy as np
from scipy.linalg import expm
import time

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin

    # Build Hamiltonians for 6-site XXZ chain
    # A: odd bonds (1-2, 3-4, 5-6) + half field
    # B: even bonds (2-3, 4-5) + half field
    # Anisotropy delta = 0.8, Field h = 0.3
    # H = sum_j delta (Sx_j Sx_j+1 + Sy_j Sy_j+1 + Sz_j Sz_j+1) - h sum_j Sz_j
    # Split: H = A + B
    # A contains bonds (0,1), (2,3), (4,5) and -h/2 sum Sz
    # B contains bonds (1,2), (3,4) and -h/2 sum Sz
    
    N = 6
    delta = 0.8
    h = 0.3
    
    # Pauli matrices
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_op(site1, site2):
        # Sx1 Sx2 + Sy1 Sy2 + delta Sz1 Sz2
        Sx1 = site_op(Sx, site1)
        Sx2 = site_op(Sx, site2)
        Sy1 = site_op(Sy, site1)
        Sy2 = site_op(Sy, site2)
        Sz1 = site_op(Sz, site1)
        Sz2 = site_op(Sz, site2)
        return Sx1 @ Sx2 + Sy1 @ Sy2 + delta * Sz1 @ Sz2
        
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Odd bonds: 0-1, 2-3, 4-5
    A += bond_op(0, 1)
    A += bond_op(2, 3)
    A += bond_op(4, 5)
    
    # Even bonds: 1-2, 3-4
    B += bond_op(1, 2)
    B += bond_op(3, 4)
    
    # Field: -h/2 sum Sz for each
    for j in range(N):
        Sz_j = site_op(Sz, j)
        A -= 0.5 * h * Sz_j
        B -= 0.5 * h * Sz_j
        
    # Exact propagator
    H = A + B
    H_herm = (H + H.conj().T) / 2 # Ensure Hermitian
    
    # Symmetric 12-stage formula
    # U = prod_{k=1}^6 exp(-i b_k t B) exp(-i a_k t A) exp(-i b_k t B) exp(-i a_k t A)
    # Wait, the prompt order is B A B A ...
    # Symmetric form: B1 A1 B2 A2 B3 A3 B4 A4 B5 A5 B6 A6 B6 A6 B5 A5 B4 A4 B3 A3 B2 A2 B1 A1
    # This corresponds to a_k = a_{13-k}, b_k = b_{13-k}
    
    def error_metric(params):
        a_half = params[:6]
        b_half = params[6:12]
        
        # Construct full a, b
        a = np.concatenate([a_half, a_half[::-1]])
        b = np.concatenate([b_half, b_half[::-1]])
        
        U = np.eye(2**N, dtype=complex)
        for k in range(12):
            U = expm(-1j * b[k] * t_eval * B_herm) @ U
            U = expm(-1j * a[k] * t_eval * A_herm) @ U
            
        U_exact = expm(-1j * t_eval * H_herm)
        
        # Spectral norm of error
        err = U - U_exact
        # Use svd for spectral norm
        s = np.linalg.svd(err, compute_uv=False)
        return s[0]
    
    t_evals = [0.25, 0.5, 1.0]
    B_herm = (B + B.conj().T) / 2
    A_herm = (A + A.conj().T) / 2
    
    # Initial guess: 4th order symmetric coefficients
    # For n=6, 4th order coefficients
    # Using known values from literature for 4th order symmetric splitting
    # a1=a6, a2=a5, a3=a4
    # b1=b6, b2=b5, b3=b4
    # Sum a = 1/2, Sum b = 1/2
    # Sum k a_k = 3/4, Sum k b_k = 3/4
    # Sum k^2 a_k = 7/12? No, let's use a simple symmetric distribution
    # a = [0.1, 0.15, 0.2, 0.2, 0.15, 0.1] -> sum 0.9? No, sum must be 0.5
    # a = [0.05, 0.075, 0.1, 0.1, 0.075, 0.05] -> sum 0.5
    # b = [0.05, 0.075, 0.1, 0.1, 0.075, 0.05] -> sum 0.5
    
    best_params = np.array([0.05, 0.075, 0.1, 0.1, 0.075, 0.05, 0.05, 0.075, 0.1, 0.1, 0.075, 0.05])
    best_err = np.inf
    
    # Simple local search
    # Since we have time, we can do a few iterations of Nelder-Mead
    from scipy.optimize import minimize
    
    def total_error(params):
        if time.time() > deadline:
            return np.inf
        errs = []
        for t in t_evals:
            a_half = params[:6]
            b_half = params[6:12]
            a = np.concatenate([a_half, a_half[::-1]])
            b = np.concatenate([b_half, b_half[::-1]])
            
            U = np.eye(2**N, dtype=complex)
            for k in range(12):
                U = expm(-1j * b[k] * t * B_herm) @ U
                U = expm(-1j * a[k] * t * A_herm) @ U
            U_exact = expm(-1j * t * H_herm)
            err = U - U_exact
            s = np.linalg.svd(err, compute_uv=False)
            errs.append(s[0])
        return np.mean(errs)
    
    # Use minimize
    try:
        res = minimize(total_error, best_params, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Ensure constraints: sum a = 1, sum b = 1
    # The optimizer might drift. We should project back or constrain.
    # For simplicity, we'll just normalize the halves
    a_half = best_params[:6]
    b_half = best_params[6:12]
    
    # Normalize to sum 0.5
    sum_a = np.sum(a_half)
    sum_b = np.sum(b_half)
    if sum_a != 0:
        a_half = a_half * (0.5 / sum_a)
    if sum_b != 0:
        b_half = b_half * (0.5 / sum_b)
        
    a = np.concatenate([a_half, a_half[::-1]])
    b = np.concatenate([b_half, b_half[::-1]])
    
    # Clip to [-50, 50]
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Final normalization check
    a = a / np.sum(a)
    b = b / np.sum(b)
    
    return a.tolist(), b.tolist()
