"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 14
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

    def strassen_block_decomposition():
        """
        Rank-49 decomposition using Strassen's algorithm on 2x2 blocks.
        
        Partition the 14 columns of B into 7 pairs of 2 columns.
        Each pair forms a 2x2 sub-matrix, and we apply 7-term Strassen.
        Total rank: 7 blocks * 7 terms = 49.
        
        For each block l (columns 2l, 2l+1 of B):
        A = [[A00, A01], [A10, A11]] (the full 2x2 input)
        B_l = [[B0,2l, B0,2l+1], [B1,2l, B1,2l+1]]
        C_l = A * B_l (2x2 output)
        
        Strassen terms:
        M1 = (A00+A11)(B0,2l + B1,2l+1)
        M2 = (A10+A11) * B0,2l
        M3 = A00 * (B0,2l+1 - B1,2l+1)
        M4 = A11 * (B1,2l - B0,2l)
        M5 = (A00+A01) * B1,2l+1
        M6 = (A10-A00) * (B0,2l + B0,2l+1)
        M7 = (A01-A11) * (B1,2l + B1,2l+1)
        
        C_l[0,0] = M1 + M4 - M5 + M7
        C_l[0,1] = M3 + M5
        C_l[1,0] = M2 + M4
        C_l[1,1] = M1 - M2 + M3 + M6
        """
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        
        for l in range(7):
            col_start = 7 * l
            k1 = 2 * l      # first column of block
            k2 = 2 * l + 1  # second column of block
            
            # V indices: row j of B, column k -> j*p + k
            # B[0,k1] -> 0*p + k1 = k1
            # B[0,k2] -> 0*p + k2 = k2
            # B[1,k1] -> 1*p + k1 = p + k1
            # B[1,k2] -> 1*p + k2 = p + k2
            
            # W indices: C[k*n + i] where k is column, i is row
            # C[0,k1] -> k1*n + 0 = 2*k1
            # C[0,k2] -> k2*n + 0 = 2*k2
            # C[1,k1] -> k1*n + 1 = 2*k1+1
            # C[1,k2] -> k2*n + 1 = 2*k2+1
            
            w_c0k1 = k1 * n      # C[0, k1]
            w_c0k2 = k2 * n      # C[0, k2]
            w_c1k1 = k1 * n + 1  # C[1, k1]
            w_c1k2 = k2 * n + 1  # C[1, k2]
            
            # M1: (A00+A11) * (B0,k1 + B1,k2)
            # U: A00=1, A11=1
            U[0, col_start] = 1    # A00
            U[3, col_start] = 1    # A11
            # V: B0,k1=1, B1,k2=1
            V[k1, col_start] = 1
            V[p + k2, col_start] = 1
            # W: C0,k1=1, C1,k2=1
            W[w_c0k1, col_start] = 1
            W[w_c1k2, col_start] = 1
            
            # M2: (A10+A11) * B0,k1
            U[2, col_start+1] = 1   # A10
            U[3, col_start+1] = 1   # A11
            V[k1, col_start+1] = 1  # B0,k1
            # W: C1,k1=1, C1,k2=-1
            W[w_c1k1, col_start+1] = 1
            W[w_c1k2, col_start+1] = -1
            
            # M3: A00 * (B0,k2 - B1,k2)
            U[0, col_start+2] = 1   # A00
            V[k2, col_start+2] = 1    # B0,k2
            V[p + k2, col_start+2] = -1  # B1,k2
            # W: C0,k2=1, C1,k2=1
            W[w_c0k2, col_start+2] = 1
            W[w_c1k2, col_start+2] = 1
            
            # M4: A11 * (B1,k1 - B0,k1)
            U[3, col_start+3] = 1   # A11
            V[p + k1, col_start+3] = 1   # B1,k1
            V[k1, col_start+3] = -1       # B0,k1
            # W: C0,k1=1, C1,k1=1
            W[w_c0k1, col_start+3] = 1
            W[w_c1k1, col_start+3] = 1
            
            # M5: (A00+A01) * B1,k2
            U[0, col_start+4] = 1   # A00
            U[1, col_start+4] = 1   # A01
            V[p + k2, col_start+4] = 1  # B1,k2
            # W: C0,k1=-1, C0,k2=1
            W[w_c0k1, col_start+4] = -1
            W[w_c0k2, col_start+4] = 1
            
            # M6: (A10-A00) * (B0,k1 + B0,k2)
            U[2, col_start+5] = 1   # A10
            U[0, col_start+5] = -1  # A00
            V[k1, col_start+5] = 1  # B0,k1
            V[k2, col_start+5] = 1  # B0,k2
            # W: C1,k2=1
            W[w_c1k2, col_start+5] = 1
            
            # M7: (A01-A11) * (B1,k1 + B1,k2)
            U[1, col_start+6] = 1   # A01
            U[3, col_start+6] = -1  # A11
            V[p + k1, col_start+6] = 1  # B1,k1
            V[p + k2, col_start+6] = 1  # B1,k2
            # W: C0,k1=1
            W[w_c0k1, col_start+6] = 1
        
        return U, V, W

    def try_rank_reduction():
        """
        Attempt to find linear dependencies among the 49 rank-1 tensors
        to potentially reduce rank below 49.
        
        The tensor for 2x2*2x14 is a direct sum of 7 copies of the
        2x2*2x2 tensor. Since each 2x2*2x2 tensor has rank 7 and the
        direct sum requires at least the sum of individual ranks,
        rank 49 is provably minimal. This function checks anyway.
        """
        if time.time() > deadline:
            return None
        
        U49, V49, W49 = strassen_block_decomposition()
        
        if not verify(U49, V49, W49):
            return None
        
        # Flatten each rank-1 tensor
        flat_dim = size_a * size_b * size_c
        tensors = np.zeros((49, flat_dim), dtype=np.float64)
        
        for r in range(49):
            T_r = np.einsum('a,b,c->abc', U49[:,r], V49[:,r], W49[:,r])
            tensors[r] = T_r.flatten()
        
        # Check numerical rank via SVD
        try:
            _, s, _ = np.linalg.svd(tensors, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        
        tol = 1e-10
        num_rank = np.sum(s > tol * s[0])
        
        if num_rank < 49:
            # Linear dependencies exist - could potentially reduce rank
            # But for direct sums of full-rank tensors, this shouldn't happen
            pass
        
        return None  # Rank 49 is minimal for this tensor structure

    # Primary: rank-49 Strassen block decomposition
    U, V, W = strassen_block_decomposition()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Secondary check with looser tolerance
    if verify(U, V, W, atol=1e-6):
        # Clean up any floating point artifacts
        U_clean = np.round(U * 2) / 2
        V_clean = np.round(V * 2) / 2
        W_clean = np.round(W * 2) / 2
        if verify(U_clean, V_clean, W_clean):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 56
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
