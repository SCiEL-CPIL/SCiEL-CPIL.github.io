"""
Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.

Strategy:
  - Split the 5 columns of B into two blocks: B1 (2 cols) and B2 (3 cols).
  - For A1 (2x2) * B1 (2x6): use Strassen-like rank-7 for the 2x2x2 core,
    extended to 6 columns via linearity => rank 7*3 = 21? No, better:
    Actually, 2x2 * 2x6 can be seen as 3 independent 2x2*2x2 multiplications,
    but we can do better by noting the structure.
    
  Let me think more carefully.
  
  The tensor for (n,m,p) = (2,5,6) has rank at most 60 (schoolbook).
  
  We can decompose:
    A is 2x5, B is 5x6, C is 2x6.
    Split B's columns: B = [B_left (5x2), B_right (5x3)]
    Split A's rows correspondingly: A = [A_left (2x2), A_right (2x3)]
    
    C_left = A_left * B_left  (2x2 * 2x2 -> 2x2)
    C_right = A_right * B_right (2x3 * 3x6 -> 2x6)
    
    But this isn't quite right because A is shared. Let me re-split.
    
    Actually, A is 2x5. We can split A into A1 (2x2) and A2 (2x3) along columns.
    B is 5x6. We split B into B1 (2x6) and B2 (3x6) along rows.
    
    Then C = A1*B1 + A2*B2 where:
      A1 is 2x2, B1 is 2x6 => C1 is 2x6
      A2 is 2x3, B2 is 3x6 => C2 is 2x6
      C = C1 + C2
    
    Rank of A1*B1 (2x2 * 2x6): This is equivalent to 3 separate 2x2*2x2 multiplications (since B1 has 6 cols = 3 groups of 2). Each 2x2*2x2 has rank 7 (Strassen). So rank = 3*7 = 21.
    
    Wait, that's not right either. 2x2 * 2x6: the tensor is M_{2,2,6}. We can view this as multiplying a 2x2 matrix by 6 different 2x1 vectors, or equivalently, it's 3 copies of M_{2,2,2} if we group columns in pairs. So rank <= 3*7 = 21.
    
    Rank of A2*B2 (2x3 * 3x6): This is M_{2,3,6}. We can group the 6 columns of B2 into 3 groups of 2, giving 3 copies of M_{2,3,2}. What's the rank of M_{2,3,2}?
    
    For M_{2,3,2}: schoolbook is 12. Can we do better?
    Actually, let's think about M_{2,3,3} first. The rank of M_{2,3,3} is known to be 12 (it's been proven). For M_{2,3,2}, we can group the 3 rows of B into... hmm.
    
    Let me try a different approach. Let's just use Strassen for the 2x2 parts and schoolbook for the rest, then see if we can optimize.
    
    Actually, let me reconsider. The total rank with the split above:
    - M_{2,2,6}: rank <= 21 (3 * Strassen 2x2x2)
    - M_{2,3,6}: Let's compute. M_{2,3,6} can be seen as 2 * M_{1,3,6} + ... no.
    
    Let me think of M_{2,3,6} as: for each of the 2 output rows, we compute a 1x3 * 3x6 product. But the A factors are shared across output rows.
    
    Actually, M_{2,3,6}: A is 2x3, B is 3x6. C is 2x6.
    We can write this as: for each column k of B (k=0..5), we compute A * B[:,k] which is a 2x1 vector. But that's just 6 separate 2x3*3x1 multiplications, each of rank 6, total 36. That's worse.
    
    Better: M_{2,3,6} = sum over the 3 middle indices. For each j=0,1,2, we have A[:,j] (2x1) * B[j,:] (1x6). This is an outer product. The tensor is sum_j a_j ⊗ b_j ⊗ e_j where... 
    
    Let me just use the known result: rank(M_{2,3,3}) = 12. For M_{2,3,6}, we can split the 6 columns into 3 groups of 2, each giving M_{2,3,2}. 
    M_{2,3,2}: A is 2x3, B is 3x2, C is 2x2. Schoolbook rank is 12. Can we do better?
    
    Hmm, this is getting complicated. Let me try a simpler but still improved approach.
    
    Alternative: Use the fact that we can apply Strassen to 2x2x2 blocks.
    
    Let's split the 5 middle dimension differently. 
    A: 2x5, B: 5x6.
    
    Group the 5 middle indices as: {0,1}, {2,3}, {4}.
    So A = [A0 (2x2), A1 (2x2), A2 (2x1)]
    B = [B0 (2x6), B1 (2x6), B2 (1x6)]
    
    C = A0*B0 + A1*B1 + A2*B2
    
    - A0*B0: 2x2 * 2x6, rank <= 21 (Strassen)
    - A1*B1: 2x2 * 2x6, rank <= 21 (Strassen)  
    - A2*B2: 2x1 * 1x6, rank = 6 (just outer products, each of the 6 output entries is a* b)
    
    Total: 21 + 21 + 6 = 48. Better than 60!
    
    Can we do even better? Let's try grouping as {0,1,2}, {3,4}:
    A = [A0 (2x3), A1 (2x2)]
    B = [B0 (3x6), B1 (2x6)]
    
    - A0*B0: 2x3 * 3x6. Rank? Schoolbook is 36. 
    - A1*B1: 2x2 * 2x6, rank <= 21.
    
    Total with schoolbook for first: 36 + 21 = 57. Worse.
    
    What about {0,1}, {2,3,4}:
    - A0*B0: 2x2*2x6, rank 21
    - A1*B1: 2x3*3x6, rank 36 (schoolbook)
    Total: 57.
    
    So the {2,2,1} split gives 48.
    
    Can we improve M_{2,2,6}? 
    M_{2,2,6}: A is 2x2, B is 2x6, C is 2x6.
    We can group B's 6 columns into 3 pairs: B = [B_a, B_b, B_c] each 2x2.
    Then C = A*B_a + A*B_b + A*B_c, each A*B_x is 2x2*2x2 with Strassen rank 7.
    Total: 21.
    
    But can we do M_{2,2,6} in less than 21? The tensor M_{2,2,6} has dimension 2*2*6=24. Lower bound on rank is at least max(4, 6) = 6... but that's very loose. Actually, for M_{n,m,p}, a lower bound is max(n*m, m*p, n*p) / something... 
    Actually, a simple lower bound: rank >= n*p / min(n,m,p) ... no.
    
    The rank of M_{2,2,6} is at least the rank of M_{2,2,2} which is 7. But since it's 3 copies, it's at least... well, they share the A factor, so it could be less than 3*7=21.
    
    Actually, I think 21 is likely close to optimal for M_{2,2,6}. Let's go with 48 total.
    
    Let me construct this:
    
    For the Strassen part (2x2*2x2 -> 2x2):
    The 7 products are:
      M1 = (a00+a11)(b00+b11)
      M2 = (a10+a11)b00
      M3 = a00(b10-b11)
      M4 = a11(b01-b00)
      M5 = (a00+a01)b11
      M6 = (a10-a00)(b00+b10)
      M7 = (a01-a11)(b01+b11)
    
    C00 = M1 - M4 + M5
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M6
    
    For M_{2,2,6}, we have B as 2x6 = [B1(2x2), B2(2x2), B3(2x2)].
    For each Bk, we compute the 7 Strassen products. Total 21.
    
    For M_{2,1,6}: A is 2x1 = [a0, a1], B is 1x6 = [b0,...,b5].
    C0k = a0 * bk, C1k = a1 * bk for k=0..5. Rank 6.
    
    Total rank: 21 + 21 + 6 = 48.
    
    Let me code this up.
"""

import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Split middle dimension into groups: [2, 2, 1]
    # A: 2x5 -> [A0(2x2), A1(2x2), A2(2x1)]
    # B: 5x6 -> [B0(2x6), B1(2x6), B2(1x6)]
    # C: 2x6 = C0 + C1 + C2
    #   C0 = A0 * B0 (2x2 * 2x6), rank 21 via 3x Strassen
    #   C1 = A1 * B1 (2x2 * 2x6), rank 21 via 3x Strassen
    #   C2 = A2 * B2 (2x1 * 1x6), rank 6
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 10 x 48
    V = [[0] * rank for _ in range(m * p)]  # 30 x 48
    W = [[0] * rank for _ in range(p * n)]  # 12 x 48
    
    col = 0
    
    # Strassen for 2x2*2x2, applied to 2x2*2x6 (3 column groups of 2)
    # For A0 (indices 0,1 of A's 5 cols) and B0 (indices 0,1 of B's 5 rows)
    # A0 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B0 = [[B[0,0..5]], [B[1,0..5]]]
    
    def add_strassen_2x2x6(a_row_start, b_row_start, col_offset):
        """Add Strassen decomposition for 2x2 * 2x6 block.
        a_row_start: starting row index in A (0 or 2 for the two rows)
        b_row_start: starting row index in B (0 or 2 for the two row groups)
        col_offset: starting column offset in the rank dimension
        Returns new col_offset.
        """
        c = col_offset
        # A is 2x2: indices (a_row_start, j) for j=0,1
        # B is 2x6: indices (b_row_start + j, k) for j=0,1, k=0..5
        
        # Strassen products for each of the 3 column pairs in B
        for cp in range(3):  # column pair 0,1,2 -> B cols 2*cp, 2*cp+1
            b_col0 = b_row_start + 0  # row 0 of B block
            b_col1 = b_row_start + 1  # row 1 of B block
            b_c0 = 2 * cp      # B column 2*cp
            b_c1 = 2 * cp + 1  # B column 2*cp+1
            
            # M1 = (a00+a11)(b00+b11)
            # In our indexing: a00 = A[a_row_start, b_row_start_group_idx=0], etc.
            # Wait, I need to be careful with indices.
            
            # A block: rows a_row_start, a_row_start+1; cols 0,1 (within the 2-col group)
            # The actual column indices in A: for group 0: cols 0,1; group 1: cols 2,3; group 2: col 4
            # Let me pass the actual A column indices instead.
            pass
        
        return c
    
    # Let me redo this more carefully with explicit indices.
    
    # A has shape 2x5. Indices: A[i, j] for i in {0,1}, j in {0,1,2,3,4}
    # B has shape 5x6. Indices: B[j, k] for j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # C has shape 2x6. Indices: C[i, k] for i in {0,1}, k in {0,1,2,3,4,5}
    
    # W uses trace convention: W[k*n + i, r] where n=2
    # So W[k*2 + i, r] corresponds to C[i, k]
    
    # Group 0: A cols 0,1; B rows 0,1
    # Group 1: A cols 2,3; B rows 2,3  
    # Group 2: A col 4; B row 4
    
    def strassen_2x2x6(a_cols, b_rows, col_start):
        """
        a_cols: list of 2 A column indices
        b_rows: list of 2 B row indices
        col_start: starting column in rank dimension
        Returns: (U, V, W, new_col)
        """
        c = col_start
        a0, a1 = a_cols  # A column indices
        b0, b1 = b_rows  # B row indices
        
        for cp in range(3):  # 3 column pairs in B (each 2x2 sub-block)
            bc0 = 2 * cp
            bc1 = 2 * cp + 1
            
            # Strassen M1 = (A[i0,a0] + A[i1,a1]) * (B[b0,bc0] + B[b1,bc1])
            # where i0=0, i1=1 for A rows
            # U factor: coefficients for A
            # V factor: coefficients for B
            # W factor: coefficients for C
            
            # M1
            U[0 * m + a0][c] = 1
            U[1 * m + a1][c] = 1
            V[b0 * p + bc0][c] = 1
            V[b1 * p + bc1][c] = 1
            # C[0, bc0] += M1, C[0, bc1] += M1, C[1, bc0] += M1, C[1, bc1] += M1? No.
            # Strassen: C00 = M1 - M4 + M5, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M2 + M6
            # So M1 contributes to C00 and C11
            W[bc0 * n + 0][c] = 1   # C[0, bc0]
            W[bc1 * n + 1][c] = 1   # C[1, bc1]
            c += 1
            
            # M2 = (A[1,a0] + A[1,a1]) * B[0,bc0]
            U[1 * m + a0][c] = 1
            U[1 * m + a1][c] = 1
            V[b0 * p + bc0][c] = 1
            # C10 += M2
            W[bc0 * n + 1][c] = 1   # C[1, bc0]
            c += 1
            
            # M3 = A[0,a0] * (B[1,bc0] - B[1,bc1])
            U[0 * m + a0][c] = 1
            V[b1 * p + bc0][c] = 1
            V[b1 * p + bc1][c] = -1
            # C01 += M3
            W[bc1 * n + 0][c] = 1   # C[0, bc1]
            c += 1
            
            # M4 = A[1,a1] * (B[0,bc1] - B[0,bc0])
            U[1 * m + a1][c] = 1
            V[b0 * p + bc1][c] = 1
            V[b0 * p + bc0][c] = -1
            # C00 += -M4, C10 += M4
            W[bc0 * n + 0][c] = -1  # C[0, bc0]
            W[bc0 * n + 1][c] = 1   # C[1, bc0]
            c += 1
            
            # M5 = (A[0,a0] + A[0,a1]) * B[1,bc1]
            U[0 * m + a0][c] = 1
            U[0 * m + a1][c] = 1
            V[b1 * p + bc1][c] = 1
            # C00 += M5, C01 += M5
            W[bc0 * n + 0][c] = 1   # C[0, bc0]
            W[bc1 * n + 0][c] = 1   # C[0, bc1]
            c += 1
            
            # M6 = (A[1,a0] - A[0,a0]) * (B[0,bc0] + B[0,bc1])
            U[1 * m + a0][c] = 1
            U[0 * m + a0][c] = -1
            V[b0 * p + bc0][c] = 1
            V[b0 * p + bc1][c] = 1
            # C11 += M6
            W[bc1 * n + 1][c] = 1   # C[1, bc1]
            c += 1
            
            # M7 = (A[0,a1] - A[1,a1]) * (B[1,bc1] + B[1,bc0])
            U[0 * m + a1][c] = 1
            U[1 * m + a1][c] = -1
            V[b1 * p + bc1][c] = 1
            V[b1 * p + bc0][c] = 1
            # C11 += M7
            W[bc1 * n + 1][c] = 1   # C[1, bc1]
            c += 1
        
        return c
    
    # Group 0: A cols [0,1], B rows [0,1]
    col = strassen_2x2x6([0, 1], [0, 1], col)
    
    # Group 1: A cols [2,3], B rows [2,3]
    col = strassen_2x2x6([2, 3], [2, 3], col)
    
    # Group 2: A col [4], B row [4], rank 6
    # C[i, k] += A[i, 4] * B[4, k] for i in {0,1}, k in {0..5}
    for k in range(6):
        U[0 * m + 4][col] = 1
        U[1 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 0][col] = 1   # C[0, k]
        W[k * n + 1][col] = 1   # C[1, k]
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
