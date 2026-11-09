import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Precompute indices for constraints
    # Constraint: sum_r U[i,j,r] V[j',k,r] W[k,i,r] = delta[j,j']
    # i in 0..n-1, k in 0..p-1, j in 0..m-1, j' in 0..m-1
    # Total constraints: n * p * m * m
    
    def flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten(params, R):
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        U = params[0:U_size].reshape(n, m, R)
        V = params[U_size:U_size+V_size].reshape(m, p, R)
        W = params[U_size+V_size:].reshape(p, n, R)
        return U, V, W

    def residuals(params, R):
        U, V, W = unflatten(params, R)
        res = np.zeros((n, p, m, m))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        # Sum over r
                        # U[i,j,:] * V[jp,k,:] * W[k,i,:]
                        s = np.sum(U[i, j, :] * V[jp, k, :] * W[k, i, :])
                        target = 1.0 if j == jp else 0.0
                        res[i, k, j, jp] = s - target
        return res.flatten()

    def verify(U, V, W, R):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Check constraints
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = np.sum(U[i, j, :] * V[jp, k, :] * W[k, i, :])
                        target = 1.0 if j == jp else 0.0
                        if abs(s - target) > 1e-6:
                            return False
        return True

    def round_to_half(arr):
        return np.round(arr * 2) / 2.0

    def try_rank(R):
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n, m, R) * 0.5
            V = np.random.randn(m, p, R) * 0.5
            W = np.random.randn(p, n, R) * 0.5
            
            params0 = flatten(U, V, W)
            
            try:
                res = least_squares(residuals, params0, args=(R,), method='trf', max_nfev=2000)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                U_sol, V_sol, W_sol = unflatten(res.x, R)
                
                # Round to half-integers
                U_rounded = round_to_half(U_sol)
                V_rounded = round_to_half(V_sol)
                W_rounded = round_to_half(W_sol)
                
                if verify(U_rounded, V_rounded, W_rounded, R):
                    # Return in the required format
                    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
                    U_out = U_rounded.reshape(n*m, R).tolist()
                    V_out = V_rounded.reshape(m*p, R).tolist()
                    W_out = W_rounded.reshape(p*n, R).tolist()
                    return {"U": U_out, "V": V_out, "W": W_out}
                
                # If rounded version fails, check if unrounded is "exact" enough
                # But the prompt requires integer/half-integer.
                # So we must find a solution that rounds well.
                # If this attempt fails, continue to next restart.
        
        return None

    # Try rank 16
    result_16 = try_rank(16)
    if result_16:
        return result_16

    # Fallback to rank 17 (hardcoded solution from current program)
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
    
    return {"U": U, "V": V, "W": W}
