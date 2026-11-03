import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = None
    best_error = float('inf')
    
    for rank in range(40, 49):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts for each rank
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            for iter in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            for r in range(rank):
                                A_mat[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        u, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        B_mat = np.zeros((n, rank))
                        for i in range(n):
                            for r in range(rank):
                                B_mat[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        v, _, _, _ = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        C_mat = np.zeros((m, rank))
                        for j in range(m):
                            for r in range(rank):
                                C_mat[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        w, _, _, _ = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
            
            # Check error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (val - 1) ** 2
                        
            if error < best_error:
                best_error = error
                best_result = (U.copy(), V.copy(), W.copy(), rank)
                
            if error < 1e-10:
                # Found a solution
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}

    if best_result:
        U, V, W, rank = best_result
        # Try to round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounded solution is valid
        error_rounded = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                    error_rounded += (val - 1) ** 2
                    
        if error_rounded < 1e-10:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
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
