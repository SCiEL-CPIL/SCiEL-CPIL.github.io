import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = t0 + 120.0  # 120s safety margin

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    def verify(U, V, W):
        """Verify that U @ V^T * W^T reconstructs T."""
        # U: (U_dim, rank), V: (V_dim, rank), W: (W_dim, rank)
        # T_reconstructed = sum_r U[:,r] * V[:,r] * W[:,r]
        # = (U * W^T) @ V^T? No.
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # = sum_r U[i,r] * (V[j,r] * W[k,r])
        # Let's compute via einsum: np.einsum('ir,jr,kr->ijk', U, V, W)
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_rec, T)

    def schoolbook():
        """Standard O(n*m*p) decomposition."""
        rank = n * m * p
        U = np.zeros((U_dim, rank), dtype=np.int64)
        V = np.zeros((V_dim, rank), dtype=np.int64)
        W = np.zeros((W_dim, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Start with schoolbook as baseline
    best = schoolbook()
    best_rank = n * m * p

    # Try Strassen-based decomposition for 2x2x2 blocks
    # For 3x4x6, we can partition:
    # Rows of A: 3 -> 2+1
    # Cols of A / Rows of B: 4 -> 2+2
    # Cols of B: 6 -> 2+2+2
    
    # Strassen for 2x2x2 has rank 7 (vs schoolbook 8)
    # We have 2 row-groups of A (top 2, bottom 1)
    # For top 2 rows: 2x2x2 blocks, use Strassen (rank 7 each)
    # For bottom 1 row: 1x2x2 blocks, schoolbook (rank 4 each, can't do better easily)
    
    # Top 2 rows of A, cols of A split into 2 groups of 2, cols of B split into 3 groups of 2
    # That's 2 (row groups) * 2 (col-A groups) * 3 (col-B groups) = 12 blocks of 2x2x2
    # With Strassen: 12 * 7 = 84
    # Bottom 1 row: 1 * 2 * 3 = 6 blocks of 1x2x2, schoolbook: 6 * 4 = 24
    # Total: 84 + 24 = 108. Worse than 72.
    
    # Better: For top 2 rows, we have 2x4x6.
    # Partition cols of B into 3 groups of 2.
    # For each group of 2 cols of B, we have 2x4x2.
    # Partition cols of A into 2 groups of 2.
    # For each group of 2 cols of A and 2 cols of B: 2x2x2, use Strassen (rank 7).
    # 2 (col-A groups) * 3 (col-B groups) = 6 blocks of 2x2x2 for top 2 rows.
    # 6 * 7 = 42.
    # Bottom 1 row: 1x4x6.
    # Partition cols of A into 2 groups of 2, cols of B into 3 groups of 2.
    # 2 * 3 = 6 blocks of 1x2x2, schoolbook (rank 4 each).
    # 6 * 4 = 24.
    # Total: 42 + 24 = 66. Better than 72!
    
    # Let's implement this.
    def strassen_2x2x2(U, V, W, i0, i1, j0, j1, k0, k1, col_offset):
        """Add Strassen decomposition for A[i0:i1+1, j0:j1+1] * B[j0:j1+1, k0:k1+1] -> C[i0:i1+1, k0:k1+1]"""
        # Indices in the full matrices
        # A indices: i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1
        # B indices: j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1
        # C indices: k0*n+i0, k0*n+i1, k1*n+i0, k1*n+i1
        
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        c00 = k0 * n + i0
        c01 = k0 * n + i1
        c10 = k1 * n + i0
        c11 = k1 * n + i1
        
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        # M2 = (A10+A11)B00 -> contributes to C10, C11
        # M3 = A00(B10-B00) -> contributes to C00, C10
        # M4 = A11(B01-B11) -> contributes to C01, C11
        # M5 = (A00+A01)B11 -> contributes to C01, C11
        # M6 = (A10-A00)(B00+B10) -> contributes to C00, C10
        # M7 = (A01-A11)(B10+B11) -> contributes to C01, C11
        
        # Term 1: M1
        col = col_offset
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # Term 2: M2
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += 1
        col += 1
        
        # Term 3: M3
        U[a00, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # Term 4: M4
        U[a11, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # Term 5: M5
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # Term 6: M6
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b10, col] += 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # Term 7: M7
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        return col

    def schoolbook_1x2x2(U, V, W, i, j0, j1, k0, k1, col_offset):
        """Add schoolbook decomposition for A[i, j0:j1+1] * B[j0:j1+1, k0:k1+1] -> C[i, k0:k1+1]"""
        col = col_offset
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        return col

    # Build Strassen-based decomposition
    total_rank = 6 * 7 + 6 * 4  # 42 + 24 = 66
    U = np.zeros((U_dim, total_rank), dtype=np.int64)
    V = np.zeros((V_dim, total_rank), dtype=np.int64)
    W = np.zeros((W_dim, total_rank), dtype=np.int64)
    
    col = 0
    # Top 2 rows (i=0,1), cols of A: {0,1} and {2,3}, cols of B: {0,1}, {2,3}, {4,5}
    for j0 in [0, 2]:
        j1 = j0 + 1
        for k0 in [0, 2, 4]:
            k1 = k0 + 1
            col = strassen_2x2x2(U, V, W, 0, 1, j0, j1, k0, k1, col)
    
    # Bottom 1 row (i=2), cols of A: {0,1} and {2,3}, cols of B: {0,1}, {2,3}, {4,5}
    for j0 in [0, 2]:
        j1 = j0 + 1
        for k0 in [0, 2, 4]:
            k1 = k0 + 1
            col = schoolbook_1x2x2(U, V, W, 2, j0, j1, k0, k1, col)
    
    # Verify
    if verify(U, V, W):
        best = {"U": U, "V": V, "W": W}
        best_rank = total_rank
    else:
        # Fall back to schoolbook
        best = schoolbook()
        best_rank = n * m * p

    # Convert to lists
    result = {
        "U": best["U"].tolist(),
        "V": best["V"].tolist(),
        "W": best["W"].tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
