"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def make_tensor():
        T = np.ones((n, m, p))
        return T
    
    T = make_tensor()
    
    def residual(U, V, W):
        """Compute residual norm"""
        r = U.shape[1]
        result = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
        return np.linalg.norm(result - T)
    
    def try_rank(r, max_iters=100, seed=42):
        """Try to find a decomposition of rank r using alternating least squares"""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        best_res = residual(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Alternate: fix V,W; solve for each row of U
            for i in range(n):
                for j in range(m):
                    row_idx = i*m+j
                    # Equations: for each k, sum_c U[row_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    A = np.zeros((p, r))
                    b = np.ones(p)
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    try:
                        u_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[row_idx, :] = u_row
                    except:
                        pass
            
            # Fix U,W; solve for each row of V
            for j in range(m):
                for k in range(p):
                    row_idx = j*p+k
                    # Equations: for each i, sum_c U[i*m+j,c] * V[row_idx,c] * W[k*n+i,c] = 1
                    # A[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    A = np.zeros((n, r))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        v_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[row_idx, :] = v_row
                    except:
                        pass
            
            # Fix U,V; solve for each row of W
            for k in range(p):
                for i in range(n):
                    row_idx = k*n+i
                    # Equations: for each j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[row_idx,c] = 1
                    # A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        w_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[row_idx, :] = w_row
                    except:
                        pass
            
            res = residual(U, V, W)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_res < 1e-10:
                break
        
        if best_res < 1e-8:
            return best_U, best_V, best_W, best_res
        return None
    
    # Try ranks from low to high
    for r in range(35, 61):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_iters=50, seed=42+r)
        if result is not None:
            U, V, W, res = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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

