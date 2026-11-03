"""Rank-30 decomposition for 2x3x6 matrix multiplication using two rank-15 2x3x3 blocks."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x3
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
    
    R233 = 15
    rows_u = n2 * m2  # 6
    rows_v = m2 * p2  # 9
    rows_w = p2 * n2  # 6
    
    def als_decompose(T_target, R, max_iter=300, tol=1e-12, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(rows_u, R) * 0.5
        V = rng.randn(rows_v, R) * 0.5
        W = rng.randn(rows_w, R) * 0.5
        
        for it in range(max_iter):
            # Solve for U given V, W
            Z_vw = np.zeros((R, rows_v * rows_w))
            for r in range(R):
                Z_vw[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            T_flat = T_target.reshape(rows_u, -1)
            try:
                U_sol, _, _, _ = np.linalg.lstsq(Z_vw.T, T_flat.T, rcond=None)
                U_new = U_sol.T
            except:
                return None
            
            # Solve for V given U, W
            Z_uw = np.zeros((R, rows_u * rows_w))
            for r in range(R):
                Z_uw[r, :] = np.outer(U_new[:, r], W[:, r]).flatten()
            T_flat_v = T_target.transpose(1, 0, 2).reshape(rows_v, -1)
            try:
                V_sol, _, _, _ = np.linalg.lstsq(Z_uw.T, T_flat_v.T, rcond=None)
                V_new = V_sol.T
            except:
                return None
            
            # Solve for W given U, V
            Z_uv = np.zeros((R, rows_u * rows_v))
            for r in range(R):
                Z_uv[r, :] = np.outer(U_new[:, r], V_new[:, r]).flatten()
            T_flat_w = T_target.transpose(2, 0, 1).reshape(rows_w, -1)
            try:
                W_sol, _, _, _ = np.linalg.lstsq(Z_uv.T, T_flat_w.T, rcond=None)
                W_new = W_sol.T
            except:
                return None
            
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if diff < tol:
                # Try rounding to nearest half-integer
                for mult in [2, 1]:
                    U_r = np.round(U * mult) / mult
                    V_r = np.round(V * mult) / mult
                    W_r = np.round(W * mult) / mult
                    
                    T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    if np.allclose(T_check, T_target, atol=1e-10):
                        return (U_r, V_r, W_r)
        
        return None
    
    # Try multiple random restarts to find a valid rank-15 decomposition
    best_233 = None
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        result = als_decompose(T233, R233, max_iter=200, seed=restart)
        if result is not None:
            best_233 = result
            break
    
    if best_233 is not None:
        U233, V233, W233 = best_233
        
        # Embed into 2x3x6
        R236 = 2 * R233
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Block 1: k in {0,1,2}
        U236[:, :R233] = U233
        for j in range(m2):
            for k in range(p2):
                V236[j*p + k, :R233] = V233[j*p2 + k, :]
                W236[k*n + j % n2, :R233] = W233[k*n2 + j % n2, :]
        
        # Block 2: k in {3,4,5}
        U236[:, R233:] = U233
        for j in range(m2):
            for k in range(p2):
                V236[j*p + (k+3), R233:] = V233[j*p2 + k, :]
                W236[(k+3)*n + j % n2, R233:] = W233[k*n2 + j % n2, :]
        
        # Verify the decomposition
        T_check = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        
        # Build expected tensor
        T_expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1.0
        
        if np.allclose(T_check, T_expected, atol=1e-10):
            # Convert to list of lists
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
