"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill
    
    # Target tensor T[i, j, k] = delta_{j,k}
    # Note: j in [0,3), k in [0,9). delta is 1 only if j==k (so k in [0,3)).
    T_true = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            if j < p: # j is always < 3, p is 9, so j < p is true
                T_true[i, j, j] = 1.0

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        T_decomp = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        return np.allclose(T_decomp, T_true, atol=1e-9)

    def schoolbook():
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
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    def try_rank(r, max_iters=500, restarts=5):
        """Try to find a CP decomposition of rank r using ALS."""
        best_err = float('inf')
        best_factors = None
        
        for restart in range(restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(42 + restart)
            
            # Initialize
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.random((p * n, r)) * 0.1 # Small positive init
            
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            for iteration in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = V3[j, :, :] * W3[:, i, :]  # shape (p, r)
                        # T[i,j,:] = U_vec @ M
                        try:
                            U_vec = np.linalg.lstsq(M.T, T_true[i, j, :], rcond=None)[0]
                        except:
                            continue
                        U3[i, j, :] = U_vec
                U = U3.reshape(n * m, r)
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        N = U.reshape(n, m, r)[:, j, :] * W.reshape(p, n, r)[k, :, :]  # shape (n, r)
                        T_vec = np.array([T_true[i, j, k] for i in range(n)])
                        try:
                            V_vec = np.linalg.lstsq(N.T, T_vec, rcond=None)[0]
                        except:
                            continue
                        V3[j, k, :] = V_vec
                V = V3.reshape(m * p, r)
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        P = U.reshape(n, m, r)[i, :, :] * V.reshape(m, p, r)[:, k, :]  # shape (m, r)
                        T_vec = np.array([T_true[i, j, k] for j in range(m)])
                        try:
                            W_vec = np.linalg.lstsq(P.T, T_vec, rcond=None)[0]
                        except:
                            continue
                        W3[k, i, :] = W_vec
                W = W3.reshape(p * n, r)
                
                # Check convergence
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                T_decomp = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.linalg.norm(T_decomp - T_true)
                
                if err < best_err:
                    best_err = err
                    best_factors = (U.copy(), V.copy(), W.copy())
                    
                if err < 1e-9:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist(),
                        "rank": r
                    }
            
        if best_err < 1e-9 and best_factors is not None:
            return {
                "U": best_factors[0].tolist(),
                "V": best_factors[1].tolist(),
                "W": best_factors[2].tolist(),
                "rank": r
            }
        return None
    
    best_result = schoolbook()
    
    # Try ranks: 15, 16, 17, 18
    # The tensor is effectively 2x3x3, so rank 18 is schoolbook for that part.
    # Try to find lower rank.
    for r in [15, 16, 17, 18]:
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=200, restarts=10)
        if result is not None and verify(result["U"], result["V"], result["W"]):
            best_result = result
            break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
