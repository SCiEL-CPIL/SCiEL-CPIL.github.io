"""Program for 3x4x7 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _laderman_3x3x3():
    """
    Laderman's rank-7 decomposition for 3x3x3 matrix multiplication.
    
    C = A * B where A, B are 3x3, C is 3x3.
    U has shape (9, 7), V has shape (9, 7), W has shape (9, 7).
    Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    # Laderman's 7 terms for 3x3x3
    # Using the standard Laderman decomposition
    # Terms are indexed 0..6
    
    U = [[0]*7 for _ in range(9)]
    V = [[0]*7 for _ in range(9)]
    W = [[0]*7 for _ in range(9)]
    
    # Laderman's decomposition:
    # C_11 = A_11*B_11 - A_12*B_21 + A_13*B_31
    # C_12 = A_11*B_12 - A_12*B_22 + A_13*B_32
    # C_13 = A_11*B_13 - A_12*B_23 + A_13*B_33
    # C_21 = A_21*B_11 - A_22*B_21 + A_23*B_31
    # C_22 = A_21*B_12 - A_22*B_22 + A_23*B_32
    # C_23 = A_21*B_13 - A_22*B_23 + A_23*B_33
    # C_31 = A_31*B_11 - A_32*B_21 + A_33*B_31
    # C_32 = A_31*B_12 - A_32*B_22 + A_33*B_32
    # C_33 = A_31*B_13 - A_32*B_23 + A_33*B_33
    #
    # This is just schoolbook (9 terms). Let me use the actual Laderman:
    #
    # Laderman (1976) rank-7 for 3x3:
    # Let me use the known explicit form.
    
    # Actually, let me use a verified Laderman decomposition.
    # The standard Laderman decomposition for 3x3x3:
    
    # Term 0: A_11 * B_11
    U[0*3+0][0] = 1  # A_11
    V[0*3+0][0] = 1  # B_11
    W[0*3+0][0] = 1  # C_11
    
    # Term 1: A_11 * (B_12 - B_21)
    U[0*3+0][1] = 1  # A_11
    V[0*3+1][1] = 1  # B_12
    V[1*3+0][1] = -1 # -B_21
    W[1*3+0][1] = 1  # C_21
    W[0*3+1][1] = -1 # -C_12
    
    # Term 2: A_11 * B_13
    U[0*3+0][2] = 1  # A_11
    V[0*3+2][2] = 1  # B_13
    W[2*3+0][2] = 1  # C_31
    
    # Term 3: (A_21 - A_12) * B_11
    U[1*3+0][3] = 1  # A_21
    U[0*3+1][3] = -1 # -A_12
    V[0*3+0][3] = 1  # B_11
    W[0*3+1][3] = 1  # C_12
    W[1*3+0][3] = -1 # -C_21
    
    # Term 4: (A_21 - A_12) * B_12
    U[1*3+0][4] = 1  # A_21
    U[0*3+1][4] = -1 # -A_12
    V[0*3+1][4] = 1  # B_12
    W[1*3+1][4] = 1  # C_22
    W[0*3+2][4] = -1 # -C_13
    
    # Term 5: (A_21 - A_12) * B_13
    U[1*3+0][5] = 1  # A_21
    U[0*3+1][5] = -1 # -A_12
    V[0*3+2][5] = 1  # B_13
    W[2*3+1][5] = 1  # C_32
    W[0*3+3-1][5] = -1 # -C_13... wait, C_13 is at index 0*3+2
    
    # I'm making errors. Let me just use schoolbook for the 3x3x3 part to be safe.
    
    # Reset and use schoolbook
    for i in range(9):
        for j in range(7):
            U[i][j] = 0
            V[i][j] = 0
            W[i][j] = 0
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return U, V, W


def _build_3x4x3_decomp():
    """
    Build a rank-12 decomposition for 3x4x3 matrix multiplication.
    
    C = A * B where A is 3x4, B is 4x3, C is 3x3.
    
    Split A = [A1 | A2] where A1 is 3x3, A2 is 3x1.
    C = A1 * B + A2 * B
    
    A1 * B: 3x3 * 4x3. Hmm, this doesn't directly work.
    
    Actually, let me think differently.
    A is 3x4, B is 4x3.
    C_ij = sum_{l=0}^{3} A_il * B_lj
    
    We can write this as:
    C = A1 * B1 + A2 * B2 + A3 * B3 + A4 * B4
    where Al is the l-th column of A (3x1) and Bl is the l-th row of B (1x3).
    
    Each Al * Bl is an outer product, rank 1.
    So total rank = 4 * 1 = 4? No, that's not right either.
    
    Actually, C_ij = sum_l A_il * B_lj.
    This is a sum of 4 outer products of vectors (A_col_l) and (B_row_l).
    So the tensor has rank at most 4.
    
    Wait, that can't be right. The rank of the 3x4x3 matrix multiplication tensor...
    
    Let me reconsider. The tensor T_{ijk} = delta_{jl} where we sum over l.
    No, T_{i,j,k} where i is row of C (0..2), j is... 
    
    Actually, for matrix multiplication A (n x m) * B (m x p) = C (n x p):
    C_{i,k} = sum_j A_{i,j} * B_{j,k}
    
    The tensor is T_{i,j,k} = 1 if we're computing C_{i,k} from A_{i,j} and B_{j,k}.
    
    For 3x4x3: n=3, m=4, p=3.
    C_{i,k} = sum_{j=0}^{3} A_{i,j} * B_{j,k}
    
    The tensor has indices (i, j, k) with i in 0..2, j in 0..3, k in 0..2.
    T_{i,j,k} = 1 for all valid (i,j,k).
    
    Rank of this tensor: we can write it as sum over j of (e_i * e_j * e_k) terms? No.
    
    Actually, the tensor is sum_{j=0}^{3} (sum_i e_i) (e_j) (sum_k e_k)
    = (sum_i e_i) (sum_j e_j) (sum_k e_k)
    
    So it's a rank-1 tensor? That can't be right for matrix multiplication.
    
    Wait, I'm confusing myself. The matrix multiplication tensor for n x m x p is:
    T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the coefficient is 1 if j matches.
    
    More precisely: T_{i,j,k} = 1 if the indices are valid, and the multiplication is
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    The tensor is: T = sum_{j=0}^{m-1} (sum_{i=0}^{n-1} e_i) ⊗ e_j ⊗ (sum_{k=0}^{p-1} e_k)
    
    No, that's not right. Let me be more careful.
    
    The matrix multiplication tensor M_{n,m,p} is defined as:
    M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k * delta_{j,j}
    
    Wait, that's just sum_{i,j,k} e_i ⊗ e_j ⊗ e_k which is rank 1 (it's the outer product
    of sum_i e_i, sum_j e_j, sum_k e_k).
    
    I think I'm confusing the tensor with something else. Let me reconsider.
    
    The matrix multiplication tensor for computing C = AB where A is n x m and B is m x p:
    The tensor has entries T_{(i,j), (j',k'), (i',k)} = 1 if i=i' and j=j' and k=k'.
    
    No, the standard definition: the tensor T in U ⊗ V ⊗ W where U = R^{n*m}, V = R^{m*p}, W = R^{n*p}.
    T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    So T has rank at most n*m*p (schoolbook), and we want to find a lower rank decomposition.
    
    For 3x4x3: T = sum_{i=0}^{2} sum_{j=0}^{3} sum_{k=0}^{2} e_{i*4+j} ⊗ e_{j*3+k} ⊗ e_{k*3+i}
    
    This has 3*4*3 = 36 terms in schoolbook.
    
    Can we do better? The rank of this tensor...
    
    Actually, I recall that for matrix multiplication A (n x m) * B (m x p), the tensor rank
    is related to the "border rank" and other invariants. For small cases, we can sometimes
    find better decompositions.
    
    For 3x4x3, let me just use the schoolbook with rank 36 for now, and see if I can improve.
    
    Actually, wait. Let me reconsider the whole approach.
    
    The task is 3x4x7. Let me just build a good decomposition for that directly.
    """
    n, m, p = 3, 4, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_3x4x7_decomp():
    """
    Build a decomposition for 3x4x7 matrix multiplication.
    
    C = A * B where A is 3x4, B is 4x7, C is 3x7.
    C_{i,k} = sum_{j=0}^{3} A_{i,j} * B_{j,k}
    
    Schoolbook rank = 3*4*7 = 84.
    
    Can we do better? Let's try to use structure.
    
    Split the 7 columns of B into groups. B = [B1 | B2 | B3] where B1 is 4x3, B2 is 4x3, B3 is 4x1.
    C = [A*B1 | A*B2 | A*B3]
    
    A*B1: 3x4 * 4x3 -> 3x3, rank at most 36 (schoolbook) or less.
    A*B2: 3x4 * 4x3 -> 3x3, rank at most 36.
    A*B3: 3x4 * 4x1 -> 3x1, rank at most 12.
    
    Total: 36 + 36 + 12 = 84. Same as schoolbook.
    
    Not helpful. Let me try a different split.
    
    Split the 4 rows of B (and columns of A) into 2+2.
    A = [A1 | A2] where A1 is 3x2, A2 is 3x2.
    B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    C = A1*B1 + A2*B2
    
    A1*B1: 3x2 * 2x7 -> 3x7, rank at most 3*2*7 = 42.
    A2*B2: 3x2 * 2x7 -> 3x7, rank at most 42.
    Total: 84. Still same.
    
    It seems hard to beat schoolbook for this size without using known fast algorithms.
    
    Let me just return schoolbook for correctness.
    """
    n, m, p = 3, 4, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 3x4x7 matrix multiplication.
    Uses schoolbook decomposition with rank 84 for guaranteed correctness.
    """
    U, V, W = _build_3x4x7_decomp()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
