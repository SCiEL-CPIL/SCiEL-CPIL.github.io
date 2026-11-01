"""Matrix multiplication 2x2 x 2x13 - tensor decomposition search."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 2x2 matrix A by a 2x13 matrix B to get a 2x13 matrix C.
    
    We use a block-diagonal approach:
    - Split the 13 columns of B into six blocks of 2 columns and one block of 1 column.
    - For each 2-column block, use the Strassen algorithm for 2x2x2 matrix multiplication (rank 7).
    - For the 1-column block, use the schoolbook algorithm for 2x2x1 (rank 4).
    
    Total rank = 6 * 7 + 4 = 46.
    
    The tensor decomposition is given by matrices U (4 x rank), V (26 x rank), W (26 x rank)
    such that T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    where T[i,j,k] = delta_{i,k} for the matrix multiplication tensor.
    
    Wait, let's be careful with the indices. The problem states:
    U has shape (n*m, rank) = (4, rank)
    V has shape (m*p, rank) = (26, rank)
    W has shape (p*n, rank) = (26, rank)
    
    And the tensor T has indices (i, j, k) where:
    i in [0, n) = [0, 2)  -> row of A, row of C
    j in [0, m) = [0, 2)  -> column of A, row of B
    k in [0, p) = [0, 13) -> column of B, column of C
    
    The matrix multiplication tensor is:
    T[i, j, k] = 1 if we are computing C[i,k] = sum_j A[i,j] * B[j,k]
    So T[i, j, k] = delta_{i,i} * delta_{k,k} ... no.
    
    Actually, the tensor for matrix multiplication M_{n,m,p} is defined by:
    T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k].
    But C[i,k] = sum_j A[i,j] B[j,k], so each term A[i,j]B[j,k] is a separate rank-1 tensor.
    The tensor is T[i, j, k] = delta_{i,k}? No.
    
    Let's think of it as: The bilinear map (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    This is a 3-tensor T with indices (i, j, k) where T[i, j, k] = 1 if the (i,j) entry of A 
    and (j,k) entry of B contribute to the (i,k) entry of C. But since each pair (i,j) and (j,k)
    uniquely determines (i,k), the tensor is just the identity in some sense.
    
    More precisely, T[i, j, k] = 1 for all valid i, j, k? No, that would be a different tensor.
    
    The correct tensor is: T[i, j, k] = 1 if and only if the indices are consistent with the 
    matrix multiplication. But since every triple (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
    corresponds to exactly one term A[i,j]*B[j,k] contributing to C[i,k], the tensor is simply:
    T[i, j, k] = 1 for all i, j, k.
    
    Wait, that's not right either. Let's look at the standard definition.
    
    The matrix multiplication tensor M_{n,m,p} is an element of R^{n x m} * R^{m x p} * R^{n x p}
    (tensor product of the three spaces). It is defined by:
    M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A * e_j^B * e_k^C * delta_{i,k}?
    
    No. The bilinear form is: (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor is: T = sum_{i,j,k} e_i * e_j * e_k * (coefficient)
    where the coefficient is 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    Since C_{i,k} = sum_j A_{i,j} B_{j,k}, the coefficient is 1 for all i, j, k.
    
    So T[i, j, k] = 1 for all i in [0,2), j in [0,2), k in [0,13).
    
    But that can't be right because then the rank would be at least max(n*m, m*p, n*p) = max(4, 26, 26) = 26,
    but we know the schoolbook rank is 52.
    
    I think I'm confusing the tensor with the matrix. Let me reconsider.
    
    The tensor T is in R^{n*m} * R^{m*p} * R^{n*p}. The indices are:
    - First index: (i, j) flattened to i*m + j, for A
    - Second index: (j, k) flattened to j*p + k, for B
    - Third index: (i, k) flattened to i*p + k, for C
    
    The tensor is: T[(i,j), (j',k), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    Wait, that's the identity tensor. That's not right.
    
    Let's think of it differently. The matrix multiplication is a bilinear map:
    f: R^{n x m} * R^{m x p} -> R^{n x p}
    f(A, B) = A * B
    
    The tensor associated with this bilinear map is:
    T = sum_{i,j,k} (e_i * e_j) * (e_j * e_k) * (e_i * e_k)
    
    where e_i * e_j is the (i,j) entry of A, e_j * e_k is the (j,k) entry of B, and e_i * e_k is the (i,k) entry of C.
    
    So T[(i,j), (j,k), (i,k)] = 1, and 0 otherwise.
    
    In the flattened indices:
    T[i*m + j, j*p + k, k*p + i] = 1  (wait, the output index is (i,k), so it's i*p + k? No, the output is n x p, so index is i*p + k? Or i + k*n?)
    
    The problem says: "The output factor follows the official trace convention W[k*n+i,r]."
    So the output index for (i, k) is k*n + i.
    
    So T[i*m + j, j*p + k, k*n + i] = 1.
    
    Now, for the schoolbook decomposition, each term A[i,j] * B[j,k] contributes to C[i,k].
    So the rank-1 tensor for this term is:
    U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1, and 0 elsewhere.
    
    This matches the current program.
    
    Now, for the Strassen algorithm for 2x2x2:
    We have A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]].
    
    C = A * B, so:
    c00 = a00*b00 + a01*b10
    c01 = a00*b01 + a01*b11
    c10 = a10*b00 + a11*b10
    c11 = a10*b01 + a11*b11
    
    Strassen's algorithm computes 7 products:
    P1 = (a00 + a11) * (b00 + b11)
    P2 = (a10 + a11) * b00
    P3 = a00 * (b10 - b11)
    P4 = a11 * (b01 - b00)
    P5 = (a00 + a01) * b11
    P6 = (a10 - a00) * (b00 + b01)
    P7 = (a01 - a11) * (b10 + b11)
    
    And:
    c00 = P1 + P4 - P5 + P7
    c01 = P3 + P5
    c10 = P2 + P4
    c11 = P1 - P2 + P3 + P6
    
    Each P_l is a product of a linear combination of A entries and a linear combination of B entries.
    So each P_l corresponds to a rank-1 tensor in the decomposition.
    
    For a 2-column block of B, say columns k and k+1, we have:
    B_block = [[b_j,k, b_j,k+1] for j in {0,1}]
    
    The 7 products are:
    P1 = (a00 + a11) * (b0,k + b1,k+1)  -- wait, no. Strassen uses b00, b01, b10, b11 which are the entries of the 2x2 B block.
    
    Let's say the 2x2 B block is:
    B_block = [[b00, b01], [b10, b11]]
    where b00 = B[0, k], b01 = B[0, k+1], b10 = B[1, k], b11 = B[1, k+1].
    
    Then the 7 products are as above, with a00, a01, a10, a11 being the entries of A.
    
    Each product P_l = (linear combo of A) * (linear combo of B_block)
    
    The linear combo of A is a vector in R^4 (for the 4 entries of A).
    The linear combo of B_block is a vector in R^4 (for the 4 entries of B_block).
    
    So for each P_l, we have:
    U_part = the linear combo coefficients for A (length 4)
    V_part = the linear combo coefficients for B_block (length 4)
    W_part = the coefficients for how P_l contributes to the 4 entries of C_block (length 4)
    
    Where C_block is the 2x2 block of C corresponding to the 2 columns of B_block.
    
    For the 2x2x2 case, the C block is:
    C_block = [[c00, c01], [c10, c11]]
    
    And we have:
    c00 = P1 + P4 - P5 + P7
    c01 = P3 + P5
    c10 = P2 + P4
    c11 = P1 - P2 + P3 + P6
    
    So the W matrix for this block is a 4x7 matrix where W[c_index, l] = coefficient of P_l in c_index.
    
    Let's list the entries:
    For P1 (l=0): c00 += 1, c11 += 1
    For P2 (l=1): c10 += 1, c11 -= 1
    For P3 (l=2): c01 += 1, c11 += 1
    For P4 (l=3): c00 += 1, c10 += 1
    For P5 (l=4): c00 -= 1, c01 += 1
    For P6 (l=5): c11 += 1
    For P7 (l=6): c00 += 1
    
    So W_block = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ]
    
    Now, the U and V parts for each P_l:
    
    P1 = (a00 + a11) * (b00 + b11)
    U_part1 = [1, 0, 0, 1]  # a00 + a11
    V_part1 = [1, 0, 0, 1]  # b00 + b11
    
    P2 = (a10 + a11) * b00
    U_part2 = [0, 0, 1, 1]  # a10 + a11
    V_part2 = [1, 0, 0, 0]  # b00
    
    P3 = a00 * (b10 - b11)
    U_part3 = [1, 0, 0, 0]  # a00
    V_part3 = [0, 0, 1, -1] # b10 - b11
    
    P4 = a11 * (b01 - b00)
    U_part4 = [0, 0, 0, 1]  # a11
    V_part4 = [-1, 1, 0, 0] # b01 - b00
    
    P5 = (a00 + a01) * b11
    U_part5 = [1, 1, 0, 0]  # a00 + a01
    V_part5 = [0, 0, 0, 1]  # b11
    
    P6 = (a10 - a00) * (b00 + b01)
    U_part6 = [-1, 0, 1, 0] # a10 - a00
    V_part6 = [1, 1, 0, 0]  # b00 + b01
    
    P7 = (a01 - a11) * (b10 + b11)
    U_part7 = [0, 1, 0, -1] # a01 - a11
    V_part7 = [0, 0, 1, 1]  # b10 + b11
    
    So for a 2-column block, the U_block is a 4x7 matrix:
    U_block = [
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ]
    
    V_block is a 4x7 matrix (for the 4 entries of the B block):
    V_block = [
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 0, 1, 0, 1, 0],   # b01
        [0, 0, 1, 0, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ]
    
    Now, for the full problem, we have 6 such 2-column blocks and 1 1-column block.
    
    For the 1-column block, we use schoolbook:
    A = [[a00, a01], [a10, a11]], B_col = [b0, b1], C_col = [c0, c1]
    c0 = a00*b0 + a01*b1
    c1 = a10*b0 + a11*b1
    
    So 4 products:
    P1 = a00 * b0
    P2 = a01 * b1
    P3 = a10 * b0
    P4 = a11 * b1
    
    U_block1 = [
        [1, 0, 0, 0],  # a00
        [0, 1, 0, 0],  # a01
        [0, 0, 1, 0],  # a10
        [0, 0, 0, 1]   # a11
    ]
    
    V_block1 = [
        [1, 0, 1, 0],  # b0
        [0, 1, 0, 1]   # b1
    ]
    
    W_block1 = [
        [1, 1, 0, 0],  # c0
        [0, 0, 1, 1]   # c1
    ]
    
    Now, we need to assemble the full U, V, W matrices.
    
    U has shape (4, total_rank) where total_rank = 6*7 + 4 = 46.
    V has shape (26, total_rank).
    W has shape (26, total_rank).
    
    For U: Each block contributes 7 or 4 columns. The U_block for each block is 4x7 or 4x4.
    We just concatenate them horizontally.
    
    For V: Each 2-column block contributes 7 columns, and the V_block is 4x7. But V has 26 rows (2*13).
    For a 2-column block at columns k, k+1, the V_block entries correspond to B[0,k], B[0,k+1], B[1,k], B[1,k+1].
    So we need to place the 4 rows of V_block into the correct rows of V.
    
    Similarly for W: Each 2-column block contributes 7 columns, and the W_block is 4x7. W has 26 rows (13*2).
    The W_block entries correspond to C[0,k], C[0,k+1], C[1,k], C[1,k+1].
    So we need to place the 4 rows of W_block into the correct rows of W.
    
    Let's define the row indices:
    For V: row index for B[j, k] is j*13 + k.
    For W: row index for C[i, k] is k*2 + i.
    
    Now, let's code this up.
    
    """
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Strassen blocks for 2x2x2
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook block for 2x2x1
    U_schoolbook = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_schoolbook = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_schoolbook = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Number of 2-column blocks and 1-column block
    num_2col_blocks = 6
    num_1col_blocks = 1
    
    total_rank = num_2col_blocks * 7 + num_1col_blocks * 4
    
    # Initialize U, V, W
    U = [[0] * total_rank for _ in range(4)]
    V = [[0] * total_rank for _ in range(26)]
    W = [[0] * total_rank for _ in range(26)]
    
    col_offset = 0
    
    # Process 2-column blocks
    for block_idx in range(num_2col_blocks):
        k_start = block_idx * 2
        k_end = k_start + 2
        
        # U: just concatenate
        for i in range(4):
            for c in range(7):
                U[i][col_offset + c] = U_strassen[i][c]
        
        # V: place into correct rows
        # V_strassen has 4 rows: b00, b01, b10, b11
        # b00 = B[0, k_start], b01 = B[0, k_end], b10 = B[1, k_start], b11 = B[1, k_end]
        v_rows = [
            0 * 13 + k_start,  # b00
            0 * 13 + k_end,    # b01
            1 * 13 + k_start,  # b10
            1 * 13 + k_end     # b11
        ]
        for v_row_idx, v_row in enumerate(v_rows):
            for c in range(7):
                V[v_row][col_offset + c] = V_strassen[v_row_idx][c]
        
        # W: place into correct rows
        # W_strassen has 4 rows: c00, c01, c10, c11
        # c00 = C[0, k_start], c01 = C[0, k_end], c10 = C[1, k_start], c11 = C[1, k_end]
        w_rows = [
            k_start * 2 + 0,  # c00
            k_end * 2 + 0,    # c01
            k_start * 2 + 1,  # c10
            k_end * 2 + 1     # c11
        ]
        for w_row_idx, w_row in enumerate(w_rows):
            for c in range(7):
                W[w_row][col_offset + c] = W_strassen[w_row_idx][c]
        
        col_offset += 7
    
    # Process 1-column block
    k_start = num_2col_blocks * 2  # 12
    
    # U: just concatenate
    for i in range(4):
        for c in range(4):
            U[i][col_offset + c] = U_schoolbook[i][c]
    
    # V: place into correct rows
    # V_schoolbook has 2 rows: b0, b1
    # b0 = B[0, k_start], b1 = B[1, k_start]
    v_rows = [
        0 * 13 + k_start,  # b0
        1 * 13 + k_start   # b1
    ]
    for v_row_idx, v_row in enumerate(v_rows):
        for c in range(4):
            V[v_row][col_offset + c] = V_schoolbook[v_row_idx][c]
    
    # W: place into correct rows
    # W_schoolbook has 2 rows: c0, c1
    # c0 = C[0, k_start], c1 = C[1, k_start]
    w_rows = [
        k_start * 2 + 0,  # c0
        k_start * 2 + 1   # c1
    ]
    for w_row_idx, w_row in enumerate(w_rows):
        for c in range(4):
            W[w_row][col_offset + c] = W_schoolbook[w_row_idx][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
