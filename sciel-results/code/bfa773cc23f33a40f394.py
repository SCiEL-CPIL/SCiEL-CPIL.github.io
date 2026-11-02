"""Rewritten program: block-based rank reduction using Strassen for 2x2x2 blocks."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x4 matrix.
    
    Strategy: Tile the multiplication into blocks and use Strassen's rank-7 
    algorithm for 2x2x2 sub-problems where possible, reducing total rank below
    the trivial 48.
    
    The 3x4 x 4x4 multiplication can be viewed as:
    - Split the 3 rows of A into one row and a 2x4 block
    - Split the 4 columns of B into two 2x4 blocks (or 4x2, etc.)
    
    Actually, let's think more carefully. We have A (3x4) * B (4x4) = C (3x4).
    
    We can partition A into rows: A = [A1; A2; A3] where A1 is 1x4, A2 is 1x4, A3 is 1x4.
    And B is 4x4.
    
    A better approach: partition into 2x2 blocks in the "inner" dimension.
    
    Let's partition the 4x4 inner dimension into 2+2.
    A (3x4) = [A_left (3x2) | A_right (3x2)]
    B (4x4) = [B_top (2x4); B_bottom (2x4)]
    
    C = A_left * B_top + A_right * B_bottom
    
    Now A_left is 3x2, B_top is 2x4. Their product is 3x4.
    Similarly A_right (3x2) * B_bottom (2x4) = 3x4.
    
    For each 3x2 * 2x4 = 3x4 product, we can further decompose.
    
    3x2 * 2x4: Partition the 3 rows into 1+2, and the 4 columns into 2+2.
    A (3x2) = [A1 (1x2); A2 (2x2)]
    B (2x4) = [B1 (2x2) | B2 (2x2)]
    
    C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    
    A1*B1: 1x2 * 2x2 = 1x2, rank 2 (trivial, can't do better than 2 for 1x2*2x2... actually rank of 1x2*2x2 is 2)
    A1*B2: 1x2 * 2x2 = 1x2, rank 2
    A2*B1: 2x2 * 2x2 = 2x2, rank 7 (Strassen)
    A2*B2: 2x2 * 2x2 = 2x2, rank 7 (Strassen)
    
    Total for 3x2 * 2x4: 2 + 2 + 7 + 7 = 18
    
    Two such blocks: 18 + 18 = 36
    
    Wait, let me re-examine. The trivial rank for 3x4*4x4 is 3*4*4 = 48.
    With this block decomposition we get 36. That's a score of 48/36 = 1.333.
    
    Let me verify this is correct by actually constructing the factors.
    """
    n, m, p = 3, 4, 4  # A is n x m, B is m x p, C is n x p
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # using the block decomposition.
    
    # Block structure:
    # A (3x4) partitioned as [A_left (3x2) | A_right (3x2)]
    # B (4x4) partitioned as [B_top (2x4); B_bottom (2x4)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each term is (3x2) * (2x4) = 3x4
    # For each (3x2)*(2x4):
    #   A_sub (3x2) = [A1 (1x2); A2 (2x2)]
    #   B_sub (2x4) = [B1 (2x2) | B2 (2x2)]
    #   C_sub = A1*B1 + A1*B2 + A2*B1 + A2*B2
    #   A1*B1: 1x2 * 2x2 -> rank 2
    #   A1*B2: 1x2 * 2x2 -> rank 2
    #   A2*B1: 2x2 * 2x2 -> rank 7 (Strassen)
    #   A2*B2: 2x2 * 2x2 -> rank 7 (Strassen)
    #   Total per (3x2)*(2x4): 18
    # Grand total: 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 12 x 36
    V = [[0] * rank for _ in range(m * p)]  # 16 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
    col = 0
    
    # Helper: add a trivial 1x2 * 2x2 block
    # A1 is 1x2 (rows of A: row i, cols j1,j2)
    # B1 is 2x2 (rows of B: k1,k2; cols l1,l2)
    # C is 1x2 (row i, cols l1,l2)
    def add_trivial_1x2_2x2(i, j1, j2, k1, k2, l1, l2):
        nonlocal col
        # Two terms: (i,j1)*(k1,l1) and (i,j2)*(k2,l2) ... no wait
        # 1x2 * 2x2: A1 = [a, b], B1 = [[b11, b12], [b21, b22]]
        # C = [a*b11 + b*b21, a*b12 + b*b22]
        # Tensor: sum over k of A[i,k] * B[k,l] = C[i,l]
        # For this sub-block: sum over k in {k1,k2} of A[i,k] * B[k,l]
        # = A[i,k1]*B[k1,l1] + A[i,k2]*B[k2,l1] for C[i,l1]
        # = A[i,k1]*B[k1,l2] + A[i,k2]*B[k2,l2] for C[i,l2]
        # So 4 terms, but we can do it in rank 2? No, 1x2*2x2 has rank 2.
        # Actually the rank of the 1x2*2x2 tensor is 2.
        # U[i*m + k1] and U[i*m + k2], V[k1*p + l1], V[k1*p + l2], V[k2*p + l1], V[k2*p + l2]
        # W[l1*n + i] and W[l2*n + i]
        # Rank 2: 
        # Term 1: U[i*m+k1]=1, V[k1*p+l1]=1, V[k1*p+l2]=0, V[k2*p+l1]=0, V[k2*p+l2]=0, W[l1*n+i]=1, W[l2*n+i]=0
        # Hmm, that's not right. Let me think again.
        # The tensor T[i,k,l] = delta for 1x2*2x2.
        # We need to write T = sum_r U_r * V_r * W_r
        # For 1x2*2x2: T[i,k,l] where i in {0}, k in {0,1}, l in {0,1}
        # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
        # Wait no. T[i,k,l] = 1 if the indices match the multiplication pattern.
        # For matrix mult C[i,l] = sum_k A[i,k] B[k,l]
        # The tensor is T[i,k,l] = 1 (it's always 1, the structure is fixed)
        # Actually the tensor for matrix multiplication is T[i,k,l] = 1 for all valid i,k,l
        # No wait. The tensor is defined such that <T, A, B> = C
        # C[i,l] = sum_{k} A[i,k] B[k,l]
        # So T[i,k,l] = 1 for all i,k,l in the appropriate ranges.
        # The rank is the minimum number of rank-1 tensors that sum to T.
        # For 1x2*2x2: T is 1x2x2, all entries 1.
        # Rank of all-ones tensor of shape 1x2x2 is 1! Because T = e_i * (sum_k e_k) * (sum_l e_l)
        # Wait, that's not a valid decomposition because U, V, W must have the right shapes.
        # U is 1x1, V is 2x1, W is 2x1.
        # T[i,k,l] = U[i] * V[k] * W[l]
        # If U[0]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=1, then T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1. Yes!
        # So rank of 1x2*2x2 is 1!
        
        # But wait, in our global decomposition, the indices are embedded.
        # Let me just use the schoolbook approach for these small blocks to be safe.
        # Actually, let me reconsider. For 1x2*2x2, the rank is indeed 1.
        # U[i*m+k] for k in {k1,k2}: we need U[i*m+k1] and U[i*m+k2]
        # V[k*p+l] for k in {k1,k2}, l in {l1,l2}
        # W[l*n+i] for l in {l1,l2}
        # If we set: U[i*m+k1]=1, U[i*m+k2]=1, V[k1*p+l1]=1, V[k1*p+l2]=1, V[k2*p+l1]=1, V[k2*p+l2]=1, W[l1*n+i]=1, W[l2*n+i]=1
        # Then the product gives T[i,k,l] = 1 for all k in {k1,k2}, l in {l1,l2}. Correct!
        # So rank 1 for this block.
        
        U[i * m + k1][col] = 1
        U[i * m + k2][col] = 1
        V[k1 * p + l1][col] = 1
        V[k1 * p + l2][col] = 1
        V[k2 * p + l1][col] = 1
        V[k2 * p + l2][col] = 1
        W[l1 * n + i][col] = 1
        W[l2 * n + i][col] = 1
        col += 1
    
    # Helper: add Strassen 2x2*2x2 block
    # A2 is 2x2 (rows i1,i2; cols j1,j2)
    # B2 is 2x2 (rows k1,k2; cols l1,l2)
    # C is 2x2 (rows i1,i2; cols l1,l2)
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def add_strassen_2x2_2x2(i1, i2, j1, j2, k1, k2, l1, l2):
        nonlocal col
        # A indices: (i1,j1), (i1,j2), (i2,j1), (i2,j2)
        # B indices: (k1,l1), (k1,l2), (k2,l1), (k2,l2)
        # C indices: (i1,l1), (i1,l2), (i2,l1), (i2,l2)
        
        # M1 = (A[i1,j1]+A[i2,j2])(B[k1,l1]+B[k2,l2])
        U[i1 * m + j1][col] = 1
        U[i2 * m + j2][col] = 1
        V[k1 * p + l1][col] = 1
        V[k2 * p + l2][col] = 1
        W[l1 * n + i1][col] = 1
        W[l2 * n + i1][col] = 1
        W[l1 * n + i2][col] = 1
        W[l2 * n + i2][col] = 1
        col += 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) B[k1,l1]
        U[i2 * m + j1][col] = 1
        U[i2 * m + j2][col] = 1
        V[k1 * p + l1][col] = 1
        W[l1 * n + i2][col] = 1
        col += 1
        
        # M3 = A[i1,j1] (B[k1,l2]-B[k2,l2])
        U[i1 * m + j1][col] = 1
        V[k1 * p + l2][col] = 1
        V[k2 * p + l2][col] = -1
        W[l2 * n + i1][col] = 1
        col += 1
        
        # M4 = A[i2,j2] (B[k2,l1]-B[k1,l1])
        U[i2 * m + j2][col] = 1
        V[k2 * p + l1][col] = 1
        V[k1 * p + l1][col] = -1
        W[l1 * n + i2][col] = 1
        col += 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) B[k2,l2]
        U[i1 * m + j1][col] = 1
        U[i1 * m + j2][col] = 1
        V[k2 * p + l2][col] = 1
        W[l2 * n + i2][col] = 1
        col += 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) (B[k1,l1]+B[k1,l2])
        U[i2 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[k1 * p + l1][col] = 1
        V[k1 * p + l2][col] = 1
        W[l2 * n + i2][col] = 1
        col += 1
        
        # M7 = (A[i1,j2]-A[i2,j1]) (B[k2,l1]+B[k2,l2])
        U[i1 * m + j2][col] = 1
        U[i2 * m + j1][col] = -1
        V[k2 * p + l1][col] = 1
        V[k2 * p + l2][col] = 1
        W[l1 * n + i1][col] = 1
        col += 1
    
    # Now let's assemble the full decomposition.
    # A (3x4), B (4x4), C (3x4)
    # Partition: A = [A_left (3x2) | A_right (3x2)], B = [B_top (2x4); B_bottom (2x4)]
    # A_left: cols 0,1 of A; A_right: cols 2,3 of A
    # B_top: rows 0,1 of B; B_bottom: rows 2,3 of B
    
    # Term 1: A_left (3x2) * B_top (2x4)
    # A_left rows: 0,1,2; cols: 0,1
    # B_top rows: 0,1; cols: 0,1,2,3
    # Sub-decompose: A_left = [A1 (1x2); A2 (2x2)], B_top = [B1 (2x2) | B2 (2x2)]
    # A1: row 0, cols 0,1
    # A2: rows 1,2, cols 0,1
    # B1: rows 0,1, cols 0,1
    # B2: rows 0,1, cols 2,3
    
    # A1*B1: 1x2 * 2x2, rank 1
    add_trivial_1x2_2x2(0, 0, 1, 0, 1, 0, 1)  # i=0, j1=0,j2=1, k1=0,k2=1, l1=0,l2=1
    
    # A1*B2: 1x2 * 2x2, rank 1
    add_trivial_1x2_2x2(0, 0, 1, 0, 1, 2, 3)  # i=0, j1=0,j2=1, k1=0,k2=1, l1=2,l2=3
    
    # A2*B1: 2x2 * 2x2, rank 7 (Strassen)
    add_strassen_2x2_2x2(1, 2, 0, 1, 0, 1, 0, 1)  # i1=1,i2=2, j1=0,j2=1, k1=0,k2=1, l1=0,l2=1
    
    # A2*B2: 2x2 * 2x2, rank 7 (Strassen)
    add_strassen_2x2_2x2(1, 2, 0, 1, 0, 1, 2, 3)  # i1=1,i2=2, j1=0,j2=1, k1=0,k2=1, l1=2,l2=3
    
    # Term 2: A_right (3x2) * B_bottom (2x4)
    # A_right: cols 2,3 of A; B_bottom: rows 2,3 of B
    # A_right = [A1 (1x2); A2 (2x2)], B_bottom = [B1 (2x2) | B2 (2x2)]
    # A1: row 0, cols 2,3
    # A2: rows 1,2, cols 2,3
    # B1: rows 2,3, cols 0,1
    # B2: rows 2,3, cols 2,3
    
    # A1*B1: 1x2 * 2x2, rank 1
    add_trivial_1x2_2x2(0, 2, 3, 2, 3, 0, 1)  # i=0, j1=2,j2=3, k1=2,k2=3, l1=0,l2=1
    
    # A1*B2: 1x2 * 2x2, rank 1
    add_trivial_1x2_2x2(0, 2, 3, 2, 3, 2, 3)  # i=0, j1=2,j2=3, k1=2,k2=3, l1=2,l2=3
    
    # A2*B1: 2x2 * 2x2, rank 7 (Strassen)
    add_strassen_2x2_2x2(1, 2, 2, 3, 2, 3, 0, 1)  # i1=1,i2=2, j1=2,j2=3, k1=2,k2=3, l1=0,l2=1
    
    # A2*B2: 2x2 * 2x2, rank 7 (Strassen)
    add_strassen_2x2_2x2(1, 2, 2, 3, 2, 3, 2, 3)  # i1=1,i2=2, j1=2,j2=3, k1=2,k2=3, l1=2,l2=3
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
