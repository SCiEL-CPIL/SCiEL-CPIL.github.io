import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    def check_decomposition(U, V, W, r, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :r], V[j*p+k, :r] * W[k*n+i, :r])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def is_half_integer(mat, tol=1e-9):
        """Check if all entries are half-integers."""
        for val in mat.flatten():
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True
    
    def refine(U, V, W, r, max_iters=20):
        for it in range(max_iters):
            if time.time() > deadline:
                break
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                    except:
                        pass
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
        return U, V, W
    
    # Try to find lower rank with half-integer entries
    for r in range(30, rank_sb):
        if time.time() > deadline:
            break
        for trial in range(3):
            if time.time() > deadline:
                break
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            U, V, W = refine(U, V, W, r)
            if check_decomposition(U, V, W, r) and is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
        if best_rank < rank_sb:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
