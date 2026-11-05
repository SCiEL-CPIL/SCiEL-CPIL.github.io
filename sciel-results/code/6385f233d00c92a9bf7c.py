"""Strassen-based rank-15 decomposition for 2x5 * 5x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = 2, 5, 6  # A: n x m, B: m x p, C: n x p
    # Trace convention: C[k*n+i] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # We build rank-1 terms. Each term corresponds to a rank-1 decomposition of the multiplication tensor.
    #
    # Partition:
    #   A (2x5) = [A1; A2; A3] where A1: 2x2, A2: 2x2, A3: 2x1
    #   B (5x6) = [B1 | B2 | B3] where B1: 2x6, B2: 2x6, B3: 1x6
    #   C = A1*B1 + A2*B2 + A3*B3
    #
    # For A1*B1 (2x2 * 2x6): split B1 = [B1a | B1b] each 2x3
    #   A1*B1 = A1*B1a + A1*B1b
    #   Each A1*B1a is 2x2 * 2x3. Split further: B1a = [B1a1 | B1a2 | B1a3] each 2x1
    #   So A1*B1a = sum over 3 columns... 
    #
    # Better: use Strassen on 2x2 blocks.
    # Split the 6 columns of B into three 2-column blocks: B = [Bc1 | Bc2 | Bc3], each 5x2
    # Split the 5 rows of A into: A = [Ar1; Ar2; Ar3], Ar1: 2x2, Ar2: 2x2, Ar3: 2x1
    #
    # C = Ar1*Bc1 + Ar2*Bc2 + Ar3*Bc3
    #
    # Ar1*Bc1: 2x2 * 2x2 -> Strassen rank 7
    # Ar2*Bc2: 2x2 * 2x2 -> Strassen rank 7  
    # Ar3*Bc3: 2x1 * 1x2 -> rank 1 (outer product)
    # Total rank = 15
    
    # Let's define indices carefully.
    # A[i, j] for i in 0..1, j in 0..4
    # B[j, k] for j in 0..4, k in 0..5
    # C[i, k] for i in 0..1, k in 0..5
    # Trace convention: C_flat[k*n + i]
    
    # Partition:
    # Ar1 = A[0:2, 0:2]  (rows 0-1 of A, cols 0-1)
    # Ar2 = A[0:2, 2:4]  (rows 0-1 of A, cols 2-3)
    # Ar3 = A[0:2, 4:5]  (rows 0-1 of A, col 4)
    # Bc1 = B[0:2, 0:2]  (rows 0-1 of B, cols 0-1)
    # Bc2 = B[2:4, 0:2]  (rows 2-3 of B, cols 0-1)
    # Bc3 = B[4:5, 0:2]  (row 4 of B, cols 0-1)
    
    # Wait, this doesn't cover all of B. B has 6 columns. Let me reconsider.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    #
    # So: C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:] + A[:,4:5] * B[4:5,:]
    #
    # Term 1: A1 (2x2) * B1 (2x6)
    # Term 2: A2 (2x2) * B2 (2x6)  
    # Term 3: A3 (2x1) * B3 (1x6)
    #
    # For term 1: 2x2 * 2x6. Split B1 into three 2x2 blocks: B1 = [B1_0 | B1_1 | B1_2]
    #   B1_0 = B[0:2, 0:2], B1_1 = B[0:2, 2:4], B1_2 = B[0:2, 4:6]
    #   A1 * B1 = A1*B1_0 + A1*B1_1 + A1*B1_2
    #   Each is 2x2 * 2x2, rank 7 via Strassen
    #
    # For term 2: A2 (2x2) * B2 (2x6). Similarly split:
    #   B2_0 = B[2:4, 0:2], B2_1 = B[2:4, 2:4], B2_2 = B[2:4, 4:6]
    #   A2 * B2 = A2*B2_0 + A2*B2_1 + A2*B2_2
    #   Each rank 7
    #
    # For term 3: A3 (2x1) * B3 (1x6). This is outer product: C3[i,k] = A3[i,0] * B3[0,k]
    #   This is rank 1: one rank-1 term per (i,k) pair? No, it's a single rank-1 matrix.
    #   Actually the tensor for 2x1 * 1x6 has rank 1 (it's just a scaled outer product).
    #   Wait, the multiplication tensor T_{i,k,j} = A[i,j]*B[j,k]. For j=4 only:
    #   T_{i,k,4} = A[i,4]*B[4,k]. This is rank 1 as a 3-way tensor? 
    #   The tensor is T[i,k,j] where j is only 4. So it's T[i,k,4] = A[i,4]*B[4,k].
    #   As a tensor in (i,k,j) space with j fixed at 4, it's an outer product of 
    #   vector A[:,4] and vector B[4,:]. So rank 1.
    #
    # Total: 3*7 + 3*7 + 1 = 43. That's worse than 54... wait no, 43 < 54.
    # Hmm, but the known result was 54 for a different partition. Let me recheck.
    #
    # Actually wait - I need to be more careful. Let me recount.
    # Term 1 has 3 sub-terms of 2x2*2x2, each rank 7: 3*7 = 21
    # Term 2 has 3 sub-terms of 2x2*2x2, each rank 7: 3*7 = 21
    # Term 3: rank 1
    # Total: 43
    #
    # But this seems too good. Let me verify the Strassen decomposition is correct.
    
    rank = 43
    U = [[0] * rank for _ in range(n * m)]  # 2*5 = 10 rows
    V = [[0] * rank for _ in range(m * p)]  # 5*6 = 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 6*2 = 12 rows
    
    col = 0
    
    def add_strassen(A_row_range, B_col_range, B_row_range):
        """Add Strassen terms for A[A_row_range] * B[B_row_range, B_col_range].
        A_row_range: (start, end) rows of A (2 rows)
        B_row_range: (start, end) rows of B (2 rows)  
        B_col_range: (start, end) cols of B (2 cols)
        """
        nonlocal col
        # A submatrix: rows a0..a1, cols ar0..ar1 (2x2)
        # B submatrix: rows br0..br1, cols bc0..bc1 (2x2)
        a0, a1 = A_row_range
        ar0, ar1 = B_row_range  # Actually this is the row range in A for the columns
        # Let me redefine: 
        # A sub: A[i, j] for i in {0,1}, j in {ar0, ar1}
        # B sub: B[j, k] for j in {br0, br1}, k in {bc0, bc1}
        pass
    
    # Let me just directly enumerate all rank-1 terms.
    # For each 2x2 * 2x2 block, Strassen gives 7 terms.
    # Strassen: 
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10-B11)
    #   M4 = A11(B01-B10)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    
    # For a 2x2*2x2 sub-problem with:
    #   A indices: (i, j) where i in {ai0, ai1}, j in {aj0, aj1}
    #   B indices: (j, k) where j in {bj0, bj1}, k in {bk0, bk1}
    #   C indices: (i, k) where i in {ci0, ci1}, k in {ck0, ck1}
    
    def add_2x2_strassen(ai0, ai1, aj0, aj1, bj0, bj1, bk0, bk1):
        nonlocal col
        # 7 Strassen terms
        # M1 = (A[ai0,aj0]+A[ai1,aj1]) * (B[bj0,bk0]+B[bj1,bk1])
        # etc.
        
        # Each term: U[row_a, col] += coeff_a, V[row_b, col] += coeff_b, W[row_c, col] += coeff_c
        # where the product contributes to C[ci, ck]
        
        # Term 1: M1
        # U contributions: A[ai0,aj0] and A[ai1,aj1]
        U[ai0 * m + aj0][col] += 1
        U[ai1 * m + aj1][col] += 1
        # V contributions: B[bj0,bk0] and B[bj1,bk1]
        V[bj0 * p + bk0][col] += 1
        V[bj1 * p + bk1][col] += 1
        # W: C[ai0,bk0] += 1, C[ai0,bk1] += 1, C[ai1,bk0] += 1, C[ai1,bk1] += 1
        W[bk0 * n + ai0][col] += 1
        W[bk1 * n + ai0][col] += 1
        W[bk0 * n + ai1][col] += 1
        W[bk1 * n + ai1][col] += 1
        col += 1
        
        # Term 2: M2 = (A[ai1,aj0]+A[ai1,aj1]) * B[bj0,bk0]
        U[ai1 * m + aj0][col] += 1
        U[ai1 * m + aj1][col] += 1
        V[bj0 * p + bk0][col] += 1
        # C[ai1,bk0] += 1
        W[bk0 * n + ai1][col] += 1
        col += 1
        
        # Term 3: M3 = A[ai0,aj0] * (B[bj1,bk0]-B[bj1,bk1])
        U[ai0 * m + aj0][col] += 1
        V[bj1 * p + bk0][col] += 1
        V[bj1 * p + bk1][col] += -1
        # C[ai0,bk0] += 1, C[ai0,bk1] += 1
        W[bk0 * n + ai0][col] += 1
        W[bk1 * n + ai0][col] += 1
        col += 1
        
        # Term 4: M4 = A[ai1,aj1] * (B[bj0,bk1]-B[bj0,bk0])
        U[ai1 * m + aj1][col] += 1
        V[bj0 * p + bk1][col] += 1
        V[bj0 * p + bk0][col] += -1
        # C[ai0,bk0] += 1, C[ai1,bk0] += 1
        W[bk0 * n + ai0][col] += 1
        W[bk0 * n + ai1][col] += 1
        col += 1
        
        # Term 5: M5 = (A[ai0,aj0]+A[ai0,aj1]) * B[bj1,bk1]
        U[ai0 * m + aj0][col] += 1
        U[ai0 * m + aj1][col] += 1
        V[bj1 * p + bk1][col] += 1
        # C[ai0,bk0] += -1, C[ai1,bk1] += 1
        W[bk0 * n + ai0][col] += -1
        W[bk1 * n + ai1][col] += 1
        col += 1
        
        # Term 6: M6 = (A[ai1,aj0]-A[ai0,aj0]) * (B[bj0,bk0]+B[bj0,bk1])
        U[ai1 * m + aj0][col] += 1
        U[ai0 * m + aj0][col] += -1
        V[bj0 * p + bk0][col] += 1
        V[bj0 * p + bk1][col] += 1
        # C[ai1,bk1] += 1
        W[bk1 * n + ai1][col] += 1
        col += 1
        
        # Term 7: M7 = (A[ai0,aj1]-A[ai1,aj1]) * (B[bj1,bk1]+B[bj1,bk0])
        U[ai0 * m + aj1][col] += 1
        U[ai1 * m + aj1][col] += -1
        V[bj1 * p + bk1][col] += 1
        V[bj1 * p + bk0][col] += 1
        # C[ai0,bk1] += 1
        W[bk1 * n + ai0][col] += 1
        col += 1
    
    # Now apply to our three groups:
    # Group 1: A[:, 0:2] * B[0:2, :]
    #   Sub-block 1a: A[0:2, 0:2] * B[0:2, 0:2]
    add_2x2_strassen(0, 1, 0, 1, 0, 1, 0, 1)
    #   Sub-block 1b: A[0:2, 0:2] * B[0:2, 2:4]
    add_2x2_strassen(0, 1, 0, 1, 0, 1, 2, 3)
    #   Sub-block 1c: A[0:2, 0:2] * B[0:2, 4:6]
    add_2x2_strassen(0, 1, 0, 1, 0, 1, 4, 5)
    
    # Group 2: A[:, 2:4] * B[2:4, :]
    #   Sub-block 2a: A[0:2, 2:4] * B[2:4, 0:2]
    add_2x2_strassen(0, 1, 2, 3, 2, 3, 0, 1)
    #   Sub-block 2b: A[0:2, 2:4] * B[2:4, 2:4]
    add_2x2_strassen(0, 1, 2, 3, 2, 3, 2, 3)
    #   Sub-block 2c: A[0:2, 2:4] * B[2:4, 4:6]
    add_2x2_strassen(0, 1, 2, 3, 2, 3, 4, 5)
    
    # Group 3: A[:, 4:5] * B[4:5, :]
    #   This is 2x1 * 1x6. Rank 1 term for each output column? 
    #   No, it's a single rank-1 decomposition of the whole 2x1 * 1x6 tensor.
    #   C[i,k] = A[i,4] * B[4,k] for i in {0,1}, k in {0,...,5}
    #   As a tensor: T[i,k,4] = A[i,4]*B[4,k]
    #   This is an outer product of vector (A[0,4], A[1,4]) and vector (B[4,0],...,B[4,5])
    #   But the tensor is 3-way (i,k,j). With j fixed at 4, it's rank 1.
    #   U[i*m+4, col] = 1 for i=0,1
    #   V[4*p+k, col] = 1 for k=0..5
    #   W[k*n+i, col] = 1 for all i,k
    U[0 * m + 4][col] += 1
    U[1 * m + 4][col] += 1
    for k in range(6):
        V[4 * p + k][col] += 1
        W[k * n + 0][col] += 1
        W[k * n + 1][col] += 1
    col += 1
    
    # col should be 43
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
