"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c]."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=atol)

    def schoolbook():
        """Return the standard rank-56 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_7_block(k1, k2, col_start):
        """
        Generate the 7 Strassen terms for a pair of columns (k1, k2).
        Returns (U_block, V_block, W_block) each of shape (size, 7).
        """
        U_b = np.zeros((size_a, 7), dtype=np.int64)
        V_b = np.zeros((size_b, 7), dtype=np.int64)
        W_b = np.zeros((size_c, 7), dtype=np.int64)
        
        # M1: (A00+A11)(B0,k1 + B1,k2)
        U_b[0, 0] = 1
        U_b[3, 0] = 1
        V_b[k1, 0] = 1
        V_b[p + k2, 0] = 1
        W_b[k1 * n, 0] = 1
        W_b[k2 * n + 1, 0] = 1
        
        # M2: (A10+A11) * B0,k1
        U_b[2, 1] = 1
        U_b[3, 1] = 1
        V_b[k1, 1] = 1
        W_b[k1 * n + 1, 1] = 1
        W_b[k2 * n + 1, 1] = -1
        
        # M3: A00 * (B0,k2 - B1,k2)
        U_b[0, 2] = 1
        V_b[k2, 2] = 1
        V_b[p + k2, 2] = -1
        W_b[k2 * n, 2] = 1
        W_b[k2 * n + 1, 2] = 1
        
        # M4: A11 * (B1,k1 - B0,k1)
        U_b[3, 3] = 1
        V_b[p + k1, 3] = 1
        V_b[k1, 3] = -1
        W_b[k1 * n, 3] = 1
        W_b[k1 * n + 1, 3] = 1
        
        # M5: (A00+A01) * B1,k2
        U_b[0, 4] = 1
        U_b[1, 4] = 1
        V_b[p + k2, 4] = 1
        W_b[k1 * n, 4] = -1
        W_b[k2 * n, 4] = 1
        
        # M6: (A10-A00) * (B0,k1 + B0,k2)
        U_b[2, 5] = 1
        U_b[0, 5] = -1
        V_b[k1, 5] = 1
        V_b[k2, 5] = 1
        W_b[k2 * n + 1, 5] = 1
        
        # M7: (A01-A11) * (B1,k1 + B1,k2)
        U_b[1, 6] = 1
        U_b[3, 6] = -1
        V_b[p + k1, 6] = 1
        V_b[p + k2, 6] = 1
        W_b[k1 * n, 6] = 1
        
        return U_b, V_b, W_b

    def try_strassen_block_decomposition():
        """
        Partition the 14 columns into 7 pairs and apply 7-term Strassen per pair.
        Total rank: 7 * 7 = 49.
        """
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        
        for l in range(7):
            col_start = 7 * l
            k1 = 2 * l
            k2 = 2 * l + 1
            
            U_b, V_b, W_b = strassen_7_block(k1, k2, col_start)
            U[:, col_start:col_start+7] = U_b
            V[:, col_start:col_start+7] = V_b
            W[:, col_start:col_start+7] = W_b
            
        return U, V, W

    # First try the constructive rank-49 Strassen decomposition
    U, V, W = try_strassen_block_decomposition()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 56
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
