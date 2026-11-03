"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125

    result = _build_decomposition(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _build_decomposition(n, m, p):
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Partition indices
    # i: {0,1} top, {2} bottom
    # j: {0,1}, {2,3}, {4}
    # k: {0,1}, {2,3}, {4,5}
    
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Top 2 rows
    for j_block in range(3):
        j0 = j_block * 2
        j1 = min(j0 + 2, m)
        j_width = j1 - j0
        
        for k_block in range(3):
            k0 = k_block * 2
            k1 = min(k0 + 2, p)
            k_width = k1 - k0
            
            if j_width == 2 and k_width == 2:
                for c in range(7):
                    U[j0, col] = sU[0, c]
                    U[j0+1, col] = sU[1, c]
                    U[5+j0, col] = sU[2, c]
                    U[5+j0+1, col] = sU[3, c]
                    
                    V[j0*6 + k0, col] = sV[0, c]
                    V[j0*6 + k0+1, col] = sV[1, c]
                    V[(j0+1)*6 + k0, col] = sV[2, c]
                    V[(j0+1)*6 + k0+1, col] = sV[3, c]
                    
                    W[k0*3 + 0, col] = sW[0, c]
                    W[k0*3 + 1, col] = sW[1, c]
                    W[(k0+1)*3 + 0, col] = sW[2, c]
                    W[(k0+1)*3 + 1, col] = sW[3, c]
                    col += 1
            elif j_width == 1 and k_width == 2:
                for i in range(2):
                    for kk in range(2):
                        U[i*5 + j0, col] = 1
                        V[j0*6 + k0 + kk, col] = 1
                        W[(k0 + kk)*3 + i, col] = 1
                        col += 1
            else:
                raise ValueError(f"Unexpected top block: {j_width}x{k_width}")
    
    # Bottom 1 row
    for j_block in range(3):
        j0 = j_block * 2
        j1 = min(j0 + 2, m)
        j_width = j1 - j0
        
        for k_block in range(3):
            k0 = k_block * 2
            k1 = min(k0 + 2, p)
            k_width = k1 - k0
            
            if j_width == 2 and k_width == 2:
                for jj in range(2):
                    for kk in range(2):
                        U[10 + j0 + jj, col] = 1
                        V[(j0 + jj)*6 + k0 + kk, col] = 1
                        W[(k0 + kk)*3 + 2, col] = 1
                        col += 1
            elif j_width == 1 and k_width == 2:
                for kk in range(2):
                    U[10 + j0, col] = 1
                    V[j0*6 + k0 + kk, col] = 1
                    W[(k0 + kk)*3 + 2, col] = 1
                    col += 1
            else:
                raise ValueError(f"Unexpected bottom block: {j_width}x{k_width}")
    
    if col != rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
        
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

# EVOLVE-BLOCK-END
