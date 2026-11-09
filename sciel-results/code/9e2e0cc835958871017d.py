import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p

    def verify_decomp(U, V, W, rank, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def round_to_half(arr):
        return np.round(arr * 2) / 2.0

    def round_to_quarter(arr):
        return np.round(arr * 4) / 4.0

    def get_rank_17_solution():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Index helpers
        # U[i*m + j, t] for A[i,j]
        # V[j*p + k, t] for B[j,k]
        # W[k*n + i, t] for C[k,i] (trace convention: W[k*n+i, r])
        
        # Term 0: A00*B00 -> C00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        
        # Term 1: (A00+A10)*(B00+B10) -> C10 part
        U[0, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1; V[3, 1] = 1
        W[3, 1] = 1  # C[0*2+1]=C[1]
        
        # Term 2: (A01+A11)*(B01+B11) -> C11 part  
        U[1, 2] = 1; U[4, 2] = 1
        V[1, 2] = 1; V[4, 2] = 1
        W[3, 2] = 1  # C[0*2+1]... wait let me be more careful
        
        # Let me restart with a cleaner approach based on known rank-17
        # Using the decomposition from the current program which is verified
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Term 1
        U[0][0] += 1; U[4][0] += 1
        V[0][0] += 1; V[4][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # Term 2
        U[3][1] += 1; U[4][1] += 1
        V[0][1] += 1
        W[1][1] += 1; W[3][1] -= 1
        
        # Term 3
        U[0][2] += 1
        V[1][2] += 1; V[4][2] -= 1
        W[2][2] += 1; W[3][2] += 1
        
        # Term 4
        U[4][3] += 1
        V[3][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[1][3] += 1
        
        # Term 5
        U[0][4] += 1; U[1][4] += 1
        V[4][4] += 1
        W[0][4] -= 1; W[2][4] += 1
        
        # Term 6
        U[3][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # Term 7
        U[1][6] += 1; U[4][6] -= 1
        V[3][6] += 1; V[4][6] += 1
        W[0][6] += 1
        
        # Term 8: A00B02
        U[0][7] += 1
        V[2][7] += 1
        W[4][7] += 1
        
        # Term 9: A01B12
        U[1][8] += 1
        V[5][8] += 1
        W[4][8] += 1
        
        # Term 10: A10B02
        U[3][9] += 1
        V[2][9] += 1
        W[5][9] += 1
        
        # Term 11: A11B12
        U[4][10] += 1
        V[5][10] += 1
        W[5][10] += 1
        
        # Term 12: A02B20
        U[2][11] += 1
        V[6][11] += 1
        W[0][11] += 1
        
        # Term 13: A02B21
        U[2][12] += 1
        V[7][12] += 1
        W[2][12] += 1
        
        # Term 14: A02B22
        U[2][13] += 1
        V[8][13] += 1
        W[4][13] += 1
        
        # Term 15: A12B20
        U[5][14] += 1
        V[6][14] += 1
        W[1][14] += 1
        
        # Term 16: A12B21
        U[5][15] += 1
        V[7][15] += 1
        W[3][15] += 1
        
        # Term 17: A12B22
        U[5][16] += 1
        V[8][16] += 1
        W[5][16] += 1
        
        return U, V, W

    def try_rank_16():
        if time.time() > deadline - 30:
            return None
            
        U17, V17, W17 = get_rank_17_solution()
        rank = 16
        
        best_result = None
        
        # Try dropping each column and optimizing
        for drop_idx in range(17):
            if time.time() > deadline - 15:
                break
                
            U0 = U17.copy()
            V0 = V17.copy()
            W0 = W17.copy()
            
            # Remove the column at drop_idx
            mask = [i for i in range(17) if i != drop_idx]
            U0 = U17[:, mask].copy()
            V0 = V17[:, mask].copy()
            W0 = W17[:, mask].copy()
            
            def error_func(params):
                U_flat = params[0:n*m*rank]
                V_flat = params[n*m*rank:n*m*rank+m*p*rank]
                W_flat = params[n*m*rank+m*p*rank:]
                U_ = U_flat.reshape(n*m, rank)
                V_ = V_flat.reshape(m*p, rank)
                W_ = W_flat.reshape(p*n, rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_[i*m+j, :], V_[j*p+k, :] * W_[k*n+i, :])
                            err += (s - 1.0)**2
                return err

            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Try multiple optimization methods
                for method in ['Powell', 'BFGS']:
                    if time.time() > deadline - 10:
                        break
                    
                    maxiter = 100 if method == 'Powell' else 200
                    res = minimize(error_func, params0, method=method, 
                                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10})
                    
                    if res.fun < 1e-6:
                        U_sol = res.x[0:n*m*rank].reshape(n*m, rank)
                        V_sol = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                        W_sol = res.x[n*m*rank+n*m*rank+m*p*rank:].reshape(p*n, rank)
                        
                        # Try rounding to half-integers
                        for round_fn in [round_to_half, round_to_quarter, lambda x: np.round(x)]:
                            U_rounded = round_fn(U_sol)
                            V_rounded = round_fn(V_sol)
                            W_rounded = round_fn(W_sol)
                            
                            if verify_decomp(U_rounded, V_rounded, W_rounded, rank):
                                return [U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()]
                        
                        # Check if unrounded solution is close enough
                        if verify_decomp(U_sol, V_sol, W_sol, rank, tol=1e-6):
                            # Round and check
                            U_rounded = round_to_half(U_sol)
                            V_rounded = round_to_half(V_sol)
                            W_rounded = round_to_half(W_sol)
                            if verify_decomp(U_rounded, V_rounded, W_rounded, rank):
                                return [U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()]
                
            except Exception:
                pass
        
        return None

    # Try rank 16
    result_16 = try_rank_16()
    
    if result_16:
        U, V, W = result_16
        return {"U": U, "V": V, "W": W}
        
    # Fall back to rank 17
    U17, V17, W17 = get_rank_17_solution()
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
