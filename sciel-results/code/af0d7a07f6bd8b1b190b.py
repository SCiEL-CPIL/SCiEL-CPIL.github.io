import time
import numpy as np

DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _build_rank_36():
    n, m, p = 2, 4, 5
    # Strassen decomposition for 2x2x2 blocks
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U[i, j], V[j, k], W[k, i]
    strassen_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    # Total rank: 4 blocks * 7 terms + 8 terms for last column
    rank = 4 * 7 + 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Process 4 blocks of 2x2x2
    # j0 is start of j-index (cols of A, rows of B)
    # k0 is start of k-index (cols of B)
    for j0 in (0, 2):
        for k0 in (0, 2):
            for r in range(7):
                # Fill U: A[i, j] where i in 0..1, j in j0..j0+1
                for li in range(2):
                    for lj in range(2):
                        U[li * m + (j0 + lj)][col] = strassen_U[r][li * 2 + lj]
                
                # Fill V: B[j, k] where j in j0..j0+1, k in k0..k0+1
                for lj in range(2):
                    for lk in range(2):
                        V[(j0 + lj) * p + (k0 + lk)][col] = strassen_V[r][lj * 2 + lk]
                
                # Fill W: C[k, i] where k in k0..k0+1, i in 0..1
                # Note: W index is k*n + i
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li][col] = strassen_W[r][lk * 2 + li]
                
                col += 1
        
    # Process last column k=4
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # This requires 8 terms (one for each i, j pair)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return U, V, W

def _build_rank_40():
    n, m, p = 2, 4, 5
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W):
    n, m, p = 2, 4, 5
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build expected tensor
    Texp = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Build decomposed tensor
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        u = U_np[:, r].reshape(n, m)
        v = V_np[:, r].reshape(m, p)
        w = W_np[:, r].reshape(p, n)
        # T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        # Using einsum for efficiency
        T += np.einsum('ij,jk,ki->ijk', u, v, w).reshape(n * m, m * p, p * n)
        
    return np.allclose(T, Texp, atol=1e-9)

def construct():
    n, m, p = 2, 4, 5
    
    # Try Rank 36 first
    try:
        U, V, W = _build_rank_36()
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # Fallback to Rank 40
    U, V, W = _build_rank_40()
    return {"U": U, "V": V, "W": W}
