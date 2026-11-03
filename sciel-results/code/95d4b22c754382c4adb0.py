import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110.0 # Keep 10s margin

    def verify_exact(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def try_rank_als(rank, max_iter=200, tol=1e-6):
        if time.time() > deadline:
            return None
            
        np.random.seed(42 + rank)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize initial values to avoid scaling issues
        U = U / np.linalg.norm(U, axis=0, keepdims=True)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update W
            for k in range(p):
                for i in range(n):
                    # Variables: w_c = W[k*n+i, c]
                    # Equations for each j:
                    # sum_c (U[i*m+j, c] * V[j*p+k, c]) * w_c = 1
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j] = U[i*m + j] * V[j*p + k]
                    # Solve M w = 1
                    try:
                        w, residuals, rank_mat, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n + i] = w
                    except:
                        pass
                        
            # Update U
            for i in range(n):
                for j in range(m):
                    # Variables: u_c = U[i*m+j, c]
                    # Equations for each k:
                    # sum_c u_c * (V[j*p+k, c] * W[k*n+i, c]) = 1
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k] = V[j*p + k] * W[k*n + i]
                    try:
                        u, residuals, rank_mat, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m + j] = u
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    # Variables: v_c = V[j*p+k, c]
                    # Equations for each i:
                    # sum_c (U[i*m+j, c] * v_c * W[k*n+i, c]) = 1
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i] = U[i*m + j] * W[k*n + i]
                    try:
                        v, residuals, rank_mat, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p + k] = v
                    except:
                        pass
                        
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (s - 1.0)**2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < tol:
                break
                
        if best_err < 1e-4:
            # Try to round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            if verify_exact(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), rank):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
            # Try integers
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if verify_exact(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
            # If rounding failed, maybe the solution is not close to half-integers?
            # The problem states "integer/half-integer entries".
            # If ALS converges to a valid real solution, it might not be close to half-integers.
            # However, tensor decompositions often have rational solutions.
            # We might need to snap to the nearest half-integer more aggressively or use a different rounding strategy.
            # For now, if it doesn't snap, we return None.
            
        return None

    # Try ranks in increasing order
    for r in [16, 17, 18]:
        if time.time() > deadline - 10:
            break
        result = try_rank_als(r)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    rank = rank_schoolbook
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
