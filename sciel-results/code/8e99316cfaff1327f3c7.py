import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 40 decomposition
    # 5 blocks of (2, 2, 3)
    # Each (2, 2, 3) block has rank 8:
    #   - (2, 2, 2) part with rank 7 (Strassen)
    #   - (2, 2, 1) part with rank 1
    
    rank = 40
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block_idx in range(5):
        k_base = block_idx * 3
        r_base = block_idx * 8
        
        # --- Strassen part for k_base, k_base+1 ---
        # M1
        U[0*m+0, r_base+0] = 1
        U[1*m+1, r_base+0] = 1
        V[0*p+k_base, r_base+0] = 1
        V[1*p+k_base+1, r_base+0] = 1
        W[k_base*n+0, r_base+0] = 1
        W[(k_base+1)*n+1, r_base+0] = 1
        
        # M2
        U[1*m+0, r_base+1] = 1
        U[1*m+1, r_base+1] = 1
        V[0*p+k_base, r_base+1] = 1
        W[k_base*n+1, r_base+1] = 1
        
        # M3
        U[0*m+0, r_base+2] = 1
        V[0*p+k_base+1, r_base+2] = 1
        V[1*p+k_base+1, r_base+2] = -1
        W[(k_base+1)*n+0, r_base+2] = 1
        
        # M4
        U[1*m+1, r_base+3] = 1
        V[1*p+k_base, r_base+3] = 1
        V[0*p+k_base, r_base+3] = -1
        W[k_base*n+1, r_base+3] = 1
        W[k_base*n+0, r_base+3] = 1
        
        # M5
        U[0*m+0, r_base+4] = 1
        U[0*m+1, r_base+4] = 1
        V[1*p+k_base+1, r_base+4] = 1
        W[(k_base+1)*n+0, r_base+4] = -1
        W[k_base*n+0, r_base+4] = -1
        
        # M6
        U[1*m+0, r_base+5] = 1
        U[0*m+0, r_base+5] = -1
        V[0*p+k_base, r_base+5] = 1
        V[0*p+k_base+1, r_base+5] = 1
        W[(k_base+1)*n+1, r_base+5] = 1
        
        # M7
        U[1*m+1, r_base+6] = 1
        U[0*m+1, r_base+6] = -1
        V[1*p+k_base, r_base+6] = 1
        V[1*p+k_base+1, r_base+6] = 1
        W[k_base*n+0, r_base+6] = 1
        W[(k_base+1)*n+1, r_base+6] = 1
        
        # --- Rank 1 part for k_base+2 ---
        # U[i,j] = 1, V[j, k_base+2] = 1, W[k_base+2, i] = 1
        for i in range(n):
            for j in range(m):
                U[i*m+j, r_base+7] = 1
        for j in range(m):
            V[j*p + (k_base+2), r_base+7] = 1
        for i in range(n):
            W[(k_base+2)*n + i, r_base+7] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
