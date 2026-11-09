import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p

    def verify_decomp(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                # W index is k*n + i
                # U index is i*m + j
                # V index is j*p + k
                    s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def round_to_half(arr):
        return np.round(arr * 2) / 2.0

    def get_rank_17_solution():
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
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
        
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    def try_rank_16():
        if time.time() > deadline - 20:
            return None
            
        U17, V17, W17 = get_rank_17_solution()
        
        # Strategy: Project rank 17 to rank 16 and optimize.
        # We drop the last column (index 16) and optimize the rest.
        rank = 16
        U0 = U17[:, :rank].copy()
        V0 = V17[:, :rank].copy()
        W0 = W17[:, :rank].copy()
        
        def error_func(params):
            U_flat, V_flat, W_flat = params[0:n*m*rank], params[n*m*rank:n*m*rank+m*p*rank], params[n*m*rank+m*p*rank:]
            U_ = U_flat.reshape(n*m, rank)
            V_ = V_flat.reshape(m*p, rank)
            W_ = W_flat.reshape(p*n, rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_[i*m+j, :] * V_[j*p+k, :] * W_[k*n+i, :])
                        err += (s - 1.0)**2
            return err

        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # Use BFGS or Powell with tight limits
            res = minimize(error_func, params0, method='Powell', options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res.fun < 1e-8:
                U_sol = res.x[0:n*m*rank].reshape(n*m, rank)
                V_sol = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_sol = res.x[n*m*rank+n*m*rank+m*p*rank:].reshape(p*n, rank)
                
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
# EVOLVE-BLOCK-END
