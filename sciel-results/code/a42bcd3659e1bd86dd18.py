import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        # Vectorized verification for speed and robustness
        # We need to check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is a tensor contraction.
        # For small sizes, a loop is fine, but let's use numpy efficiently.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Extract slices
                    u_slice = U[i*m+j, :rank]
                    v_slice = V[j*p+k, :rank]
                    w_slice = W[k*n+i, :rank]
                    val = np.dot(u_slice, v_slice * w_slice)
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def strassen_2x2x2x2(U, V, W, base_col, a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        col = base_col
        # M1
        U[a11, col] += 1; U[a22, col] += 1
        V[b11, col] += 1; V[b22, col] += 1
        W[c11, col] += 1; W[c22, col] += 1
        
        col += 1
        # M2
        U[a21, col] += 1; U[a22, col] += 1
        V[b11, col] += 1
        W[c21, col] += 1; W[c22, col] -= 1
        
        col += 1
        # M3
        U[a11, col] += 1
        V[b12, col] += 1; V[b22, col] -= 1
        W[c12, col] += 1; W[c22, col] += 1
        
        col += 1
        # M4
        U[a22, col] += 1
        V[b21, col] += 1; V[b11, col] -= 1
        W[c11, col] += 1; W[c21, col] += 1
        
        col += 1
        # M5
        U[a11, col] += 1; U[a12, col] += 1
        V[b22, col] += 1
        W[c11, col] -= 1; W[c12, col] += 1
        
        col += 1
        # M6
        U[a21, col] += 1; U[a11, col] -= 1
        V[b11, col] += 1; V[b12, col] += 1
        W[c22, col] += 1
        
        col += 1
        # M7
        U[a12, col] += 1; U[a22, col] -= 1
        V[b21, col] += 1; V[b22, col] += 1
        W[c11, col] += 1
        
        return base_col + 7

    def strassen_56():
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for v in range(4):
            for half in range(2):
                a11 = 0 * m + 2 * half
                a12 = 0 * m + 2 * half + 1
                a21 = 1 * m + 2 * half
                a22 = 1 * m + 2 * half + 1
                
                b11 = (2 * half) * p + 2 * v
                b12 = (2 * half) * p + 2 * v + 1
                b21 = (2 * half + 1) * p + 2 * v
                b22 = (2 * half + 1) * p + 2 * v + 1
                
                c11 = (2 * v) * n + 0
                c12 = (2 * v + 1) * n + 0
                c21 = (2 * v) * n + 1
                c22 = (2 * v + 1) * n + 1
                
                col = strassen_2x2x2x2(U, V, W, col, a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22)
        
        return U, V, W, rank

    U_best, V_best, W_best, rank_best = strassen_56()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
