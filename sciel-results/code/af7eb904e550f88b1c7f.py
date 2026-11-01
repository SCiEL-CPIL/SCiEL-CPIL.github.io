import time
import sys
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts a rank-51 decomposition using Strassen for the first 2 columns
    of B and schoolbook for the remaining 11. Falls back to rank-52 if
    verification fails.
    """
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Target tensor T[i, j, k] = 1 if i*m+j, j*p+k, k*n+i are the indices
    # Actually, the tensor is defined by the multiplication.
    # T[U_idx, V_idx, W_idx] = 1 if U_idx corresponds to A[i,j], V_idx to B[j,k], W_idx to C[k,i]
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # Build the target tensor (sparse representation or dense)
    # Dense tensor size: 4 x 26 x 26
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1

    def verify_decomp(U, V, W):
        """
        Verifies if U, V, W exactly decompose T.
        U: (4, rank)
        V: (26, rank)
        W: (26, rank)
        """
        rank = U.shape[1]
        # Reconstruct tensor
        T_recon = np.zeros((n*m, m*p, p*n), dtype=int)
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Outer product
            # T_recon += np.einsum('a,b,c->abc', u, v, w)
            for a in range(n*m):
                if u[a] != 0:
                    for b in range(m*p):
                        if v[b] != 0:
                            for c in range(p*n):
                                if w[c] != 0:
                                    T_recon[a, b, c] += u[a] * v[b] * w[c]
        return np.array_equal(T_recon, T)

    # --- Attempt Rank 51 ---
    rank_51 = 51
    U51 = np.zeros((n*m, rank_51), dtype=int)
    V51 = np.zeros((m*p, rank_51), dtype=int)
    W51 = np.zeros((p*n, rank_51), dtype=int)
    
    # Strassen for columns 0, 1
    # A indices: 0: a, 1: b, 2: c, 3: d
    # B indices for cols 0,1:
    # e: B[0,0] -> 0
    # f: B[0,1] -> 1
    # g: B[1,0] -> 13
    # h: B[1,1] -> 14
    # C indices for cols 0,1:
    # C[0,0] -> 0
    # C[0,1] -> 2
    # C[1,0] -> 1
    # C[1,1] -> 3
    
    # M1: (a+d)(e+h)
    U51[0, 0] = 1; U51[3, 0] = 1
    V51[0, 0] = 1; V51[14, 0] = 1
    W51[0, 0] = 1; W51[3, 0] = 1 # C00, C11
    
    # M2: (c+d)e
    U51[2, 1] = 1; U51[3, 1] = 1
    V51[0, 1] = 1
    W51[1, 1] = 1; W51[3, 1] = -1 # C10, C11
    
    # M3: a(f-h)
    U51[0, 2] = 1
    V51[1, 2] = 1; V51[14, 2] = -1
    W51[2, 2] = 1; W51[3, 2] = -1 # C01, C11
    
    # M4: d(g-f)
    U51[3, 3] = 1
    V51[13, 3] = 1; V51[1, 3] = -1
    W51[0, 3] = 1; W51[1, 3] = -1 # C00, C10
    
    # M5: (a+b)h
    U51[0, 4] = 1; U51[1, 4] = 1
    V51[14, 4] = 1
    W51[0, 4] = -1; W51[2, 4] = 1 # C00, C01
    
    # M6: (c-a)(e+f)
    U51[2, 5] = 1; U51[0, 5] = -1
    V51[0, 5] = 1; V51[1, 5] = 1
    W51[3, 5] = 1 # C11
    
    # M7: (b-d)(g+h)
    U51[1, 6] = 1; U51[3, 6] = -1
    V51[13, 6] = 1; V51[14, 6] = 1
    W51[0, 6] = 1 # C00

    # Schoolbook for columns 2..12
    col_offset = 7
    for k in range(2, p):
        for i in range(n):
            for j in range(m):
                col = col_offset + (k-2)*4 + i*2 + j
                U51[i*m+j, col] = 1
                V51[j*p+k, col] = 1
                W51[k*n+i, col] = 1

    if verify_decomp(U51, V51, W51):
        return {"U": U51.tolist(), "V": V51.tolist(), "W": W51.tolist()}

    # --- Fallback Rank 52 ---
    rank_52 = 52
    U52 = np.zeros((n*m, rank_52), dtype=int)
    V52 = np.zeros((m*p, rank_52), dtype=int)
    W52 = np.zeros((p*n, rank_52), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U52[i*m+j, col] = 1
                V52[j*p+k, col] = 1
                W52[k*n+i, col] = 1
                col += 1
                
    return {"U": U52.tolist(), "V": V52.tolist(), "W": W52.tolist()}
# EVOLVE-BLOCK-END
