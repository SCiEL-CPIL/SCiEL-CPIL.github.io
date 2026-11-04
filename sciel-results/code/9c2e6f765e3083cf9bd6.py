"""Optimized tensor decomposition for 2x3 x 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-14 tensor decomposition for 2x3 x 3x6 matrix multiplication.
    
    Strategy: Partition the 3x6 matrix into blocks and use Strassen's algorithm
    for the 2x2x2 sub-problems.
    
    We partition:
    - A (2x3) = [A1 | A2] where A1 is 2x2, A2 is 2x1
    - B (3x6) = [B11; B12] where B11 is 2x6, B12 is 1x6
    
    Actually, let's think more carefully. We want to compute C = A * B where
    A is 2x3 and B is 3x6.
    
    Let's partition B as [B_top; B_bottom] where B_top is 2x6 and B_bottom is 1x6.
    And A as [A_left | A_right] where A_left is 2x2 and A_right is 2x1.
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2 times 2x6), we can use Strassen-like decomposition.
    Actually, 2x2 times 2x6 can be decomposed using the rank-7 Strassen decomposition
    for the 2x2x2 case, applied to each of the 6 columns, but that gives rank 7*6=42 which is worse.
    
    Better approach: Use the known optimal decomposition for small matrices.
    
    For 2x3 x 3x6, the tensor rank is at most 36 (schoolbook). 
    We can do better by exploiting structure.
    
    Let's use a different partitioning. Write A as a 2x3 matrix and B as 3x6.
    
    C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1; j=0,1,2; k=0..5
    
    We can group the k indices. For each pair of k values (k, k+1) for k=0,1,2,
    we have a 2x3 x 3x2 sub-problem.
    
    The rank of the 2x3 x 3x2 multiplication tensor is known to be at most 12 (schoolbook).
    But we can do better.
    
    Actually, let me think about this differently. The total rank needed is the sum
    of ranks for independent sub-problems if they are truly independent.
    
    Let's try: Partition B into three 3x2 blocks: B = [B0 | B1 | B2] where each Bi is 3x2.
    Then C = [A*B0 | A*B1 | A*B2].
    
    Each A*Bi is a 2x3 times 3x2 = 2x2 matrix product. The tensor rank for 2x3 x 3x2
    is at most 12 (schoolbook: 2*3*2=12).
    
    So total rank would be 3*12 = 36, same as schoolbook.
    
    We need to find a better decomposition. Let's look for the actual minimal rank.
    
    For 2x3 x 3x6, one approach is to use the fact that we can combine terms.
    
    Let me try a direct construction. The tensor T has T[i,j,k] = 1 if the indices are valid
    (i in 0..1, j in 0..2, k in 0..5), representing the multiplication.
    
    Actually, the tensor for matrix multiplication C = AB where A is m x n and B is n x p
    has entries T[i,j,k,l] = delta_{j,k} for the standard basis, but in the tensor network
    formulation, the multiplication tensor is sum_{i,j,k} e_i ⊗ e_j ⊗ (sum_l A_il B_lk e_l).
    
    In the standard tensor decomposition for matrix multiplication of size n x m times m x p:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k ⊗ (A_{i,j} B_{j,k})
    
    Wait, I need to be more careful. The multiplication tensor M_{n,m,p} is an element of
    (F^n)^* ⊗ (F^m)^* ⊗ (F^p) ⊗ (F^{n×p}) or equivalently in the tensor network:
    
    The tensor is sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where we contract with A and B.
    
    In the rank decomposition context, we want to write the multiplication map as a sum of
    rank-1 tensors. A rank-1 term corresponds to choosing specific indices and multiplying.
    
    The schoolbook algorithm uses n*m*p rank-1 terms: for each (i,j,k), the term is
    A_{i,j} * B_{j,k} contributing to C_{i,k}.
    
    To get a better rank, we need to find linear combinations that reduce the number of terms.
    
    For 2x2x2, Strassen shows rank 7 < 8.
    
    For our problem 2x3 x 3x6, let me try to construct a rank-14 decomposition as suggested.
    
    Partition: Write A (2x3) = [A1 A2] where A1 is 2x2 and A2 is 2x1.
    Write B (3x6) = [B1; B2] where B1 is 2x6 and B2 is 1x6.
    
    C = A1*B1 + A2*B2
    
    A1*B1 is 2x2 times 2x6. We can decompose this using Strassen for each 2x2 block of B1.
    B1 = [B1_0 B1_1 B1_2] where each B1_i is 2x2.
    A1*B1 = [A1*B1_0 | A1*B1_1 | A1*B1_2]
    Each A1*B1_i uses rank 7 (Strassen), so total 21 for this part.
    
    A2*B2 is 2x1 times 1x6 = 2x6. This is just scaling: C2 = A2[0]*B2[0,:] and C2 = A2[1]*B2[1,:]
    This uses rank 2 (one for each row of the result, or actually rank 2 since it's outer product-like).
    Actually, A2*B2 where A2 is 2x1 and B2 is 1x6: the result is 2x6.
    C[i,k] = A2[i,0] * B2[0,k]. This is a rank-2 tensor: sum over i,k of A2[i,0] B2[0,k] e_i ⊗ e_k.
    As a sum of rank-1 terms in the tensor decomposition sense, this is 2 terms (one for i=0, one for i=1)
    times the full B2 row, but in the tensor rank sense for the multiplication, it's just 2 rank-1 terms
    if we consider that for each i, the term is A2[i,0] * (B2[0,:]) which is one rank-1 term per i.
    So rank 2 for this part.
    
    Total: 21 + 2 = 23. That's better than 36 but not 14.
    
    Hmm, the suggested rank 14 might not be achievable with this simple partitioning.
    
    Let me try a different approach. Maybe we can do better by not separating so cleanly.
    
    Actually, for the 2x2x2 Strassen, the rank is 7. If we have 2x2 times 2x6, we can think of it
    as 3 separate 2x2x2 multiplications, but they share the A matrix. However, in tensor decomposition,
    sharing doesn't directly reduce rank unless we find a joint decomposition.
    
    Let me just implement a reasonable decomposition. Given the constraints, let me try to get
    rank 23 as described above, or see if I can do better.
    
    Actually, let me reconsider. The problem says "rank-minimal exact tensor decomposition". 
    The known lower bound for 2x3 x 3x6 might be higher than 14.
    
    Let me just implement the Strassen-based decomposition with rank 23 and see.
    
    Strassen for 2x2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 2x2 times 2x6, we apply Strassen to each 2x2 block of the 2x6 matrix.
    
    Let me implement this.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We'll build U, V, W matrices with integer/half-integer entries.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (18, rank)  
    # W has shape (p*n, rank) = (12, rank)
    
    # Indexing:
    # U[i*m + j, c] where i in 0..n-1, j in 0..m-1, c in 0..rank-1
    # V[j*p + k, c] where j in 0..m-1, k in 0..p-1, c in 0..rank-1
    # W[k*n + i, c] where k in 0..p-1, i in 0..n-1, c in 0..rank-1
    # (Note: W uses trace convention W[k*n + i, r])
    
    # Let's use the Strassen-based decomposition with rank 23.
    # Partition A (2x3) as [A1 | A2] where A1 is 2x2 (cols 0,1), A2 is 2x1 (col 2)
    # Partition B (3x6) as [B1; B2] where B1 is 2x6 (rows 0,1), B2 is 1x6 (row 2)
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 times 2x6):
    # B1 = [B1_0 | B1_1 | B1_2] where B1_i is 2x2
    # Apply Strassen to each: 7 terms each, total 21 terms
    
    # For A2*B2 (2x1 times 1x6):
    # C[i,k] = A2[i,0] * B2[0,k]
    # This is 2 rank-1 terms (one for i=0, one for i=1)
    # Total: 23 terms
    
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 23
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 23
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 23
    
    col = 0
    
    # Strassen decomposition for 2x2 times 2x2
    # A1 is 2x2: A1[0,0]=A[0,0], A1[0,1]=A[0,1], A1[1,0]=A[1,0], A1[1,1]=A[1,1]
    # In our indexing: A[i,j] corresponds to U[i*m+j, c]
    # So A[0,0] -> U[0*3+0, c] = U[0, c]
    # A[0,1] -> U[0*3+1, c] = U[1, c]
    # A[1,0] -> U[1*3+0, c] = U[3, c]
    # A[1,1] -> U[1*3+1, c] = U[4, c]
    # A[0,2] -> U[0*3+2, c] = U[2, c]
    # A[1,2] -> U[1*3+2, c] = U[5, c]
    
    # B1 is 2x6: B1[j,k] for j in 0,1 and k in 0..5
    # B1[j,k] -> V[j*6+k, c]
    # B1[0,0] -> V[0, c], B1[0,1] -> V[1, c], ..., B1[0,5] -> V[5, c]
    # B1[1,0] -> V[6, c], B1[1,1] -> V[7, c], ..., B1[1,5] -> V[11, c]
    # B2 is 1x6: B2[0,k] for k in 0..5
    # B2[0,k] -> V[2*6+k, c] = V[12+k, c]
    # B2[0,0] -> V[12, c], ..., B2[0,5] -> V[17, c]
    
    # C is 2x6: C[i,k] for i in 0,1 and k in 0..5
    # C[i,k] -> W[k*2+i, c] (trace convention: W[k*n+i, r])
    # C[0,0] -> W[0, c], C[0,1] -> W[2, c], ..., C[0,5] -> W[10, c]
    # C[1,0] -> W[1, c], C[1,1] -> W[3, c], ..., C[1,5] -> W[11, c]
    
    def strassen_2x2_2x2(col_start, b_col_offset):
        """
        Add Strassen terms for 2x2 times 2x2.
        b_col_offset: the starting column in B1 for this 2x2 block (0, 2, or 4)
        Returns the next column index.
        """
        nonlocal col
        # A1 indices: A1[0,0]=U[0], A1[0,1]=U[1], A1[1,0]=U[3], A1[1,1]=U[4]
        # B1 block: B1[0,0]=V[b_col_offset], B1[0,1]=V[b_col_offset+1],
        #           B1[1,0]=V[6+b_col_offset], B1[1,1]=V[6+b_col_offset+1]
        # C block: C[0,0]=W[b_col_offset*2], C[0,1]=W[b_col_offset*2+2],
        #          C[1,0]=W[b_col_offset*2+1], C[1,1]=W[b_col_offset*2+3]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][col] = 1; U[4][col] = 1
        V[b_col_offset][col] = 1; V[6+b_col_offset][col] = 1
        W[b_col_offset*2][col] = 1; W[b_col_offset*2+3][col] = 1
        W[b_col_offset*2+1][col] = -1; W[b_col_offset*2+2][col] = -1
        col += 1
        
        # M2 = (A21+A22)B11
        U[3][col] = 1; U[4][col] = 1
        V[b_col_offset][col] = 1
        W[b_col_offset*2+1][col] = 1; W[b_col_offset*2+3][col] = -1
        col += 1
        
        # M3 = A11(B21-B22)
        U[0][col] = 1
        V[6+b_col_offset][col] = 1; V[b_col_offset+1][col] = -1
        W[b_col_offset*2+2][col] = 1; W[b_col_offset*2+3][col] = 1
        col += 1
        
        # M4 = A22(B12-B11)
        U[4][col] = 1
        V[b_col_offset+1][col] = 1; V[b_col_offset][col] = -1
        W[b_col_offset*2][col] = 1; W[b_col_offset*2+1][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[0][col] = 1; U[1][col] = 1
        V[b_col_offset+1][col] = 1
        W[b_col_offset*2][col] = -1; W[b_col_offset*2+2][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[3][col] = 1; U[0][col] = -1
        V[b_col_offset][col] = 1; V[b_col_offset+1][col] = 1
        W[b_col_offset*2+3][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1][col] = 1; U[4][col] = -1
        V[6+b_col_offset][col] = 1; V[b_col_offset+1][col] = 1
        W[b_col_offset*2][col] = 1
        col += 1
    
    # Apply Strassen to each 2x2 block of B1
    strassen_2x2_2x2(0, 0)   # B1 cols 0,1 -> C cols 0,1
    strassen_2x2_2x2(0, 2)   # B1 cols 2,3 -> C cols 2,3
    strassen_2x2_2x2(0, 4)   # B1 cols 4,5 -> C cols 4,5
    
    # Now handle A2*B2 (2x1 times 1x6)
    # C[i,k] += A2[i,0] * B2[0,k]
    # A2[0,0] = U[2, c], A2[1,0] = U[5, c]
    # B2[0,k] = V[12+k, c]
    # C[0,k] = W[k*2+0, c], C[1,k] = W[k*2+1, c]
    
    # For i=0: term is A[0,2] * B2[0,k] for all k
    # This is one rank-1 term: U[2][col]=1, V[12+k][col]=1 for all k, W[k*2][col]=1 for all k
    for k in range(6):
        V[12+k][col] = 1
        W[k*2][col] = 1
    U[2][col] = 1
    col += 1
    
    # For i=1: term is A[1,2] * B2[0,k] for all k
    for k in range(6):
        V[12+k][col] = 1
        W[k*2+1][col] = 1
    U[5][col] = 1
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
