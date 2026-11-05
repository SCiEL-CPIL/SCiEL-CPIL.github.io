import numpy as np

def construct():
    """
    Exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Returns factor matrices U, V, W with integer entries that exactly
    decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    
    We use a block-structured approach:
    - Split the 4x4 matrix A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    - Split the 4x7 matrix B into blocks based on rows (2+2) and columns (4+3):
      B = [[B11(2x4), B12(2x3)], [B21(2x4), B22(2x3)]]
    
    The product C = A*B has dimensions 4x7.
    C = [[C11(2x4), C12(2x3)], [C21(2x4), C22(2x3)]]
    
    Each C block is computed as sum of products of A blocks and B blocks:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    This gives 8 terms. Each term is a rank decomposition of the product of two sub-matrices.
    For a 2x2 * 2x4 product, the schoolbook rank is 8.
    For a 2x2 * 2x3 product, the schoolbook rank is 6.
    
    Total rank = 4 * 8 + 4 * 6 = 32 + 24 = 56.
    This is better than the full schoolbook rank of 112.
    
    Let's verify the indices:
    A is 4x4. Rows 0-3, Cols 0-3.
    B is 4x7. Rows 0-3, Cols 0-6.
    C is 4x7. Rows 0-3, Cols 0-6.
    
    Block definitions:
    A11: A[0:2, 0:2]
    A12: A[0:2, 2:4]
    A21: A[2:4, 0:2]
    A22: A[2:4, 2:4]
    
    B11: B[0:2, 0:4]
    B12: B[0:2, 4:7]
    B21: B[2:4, 0:4]
    B22: B[2:4, 4:7]
    
    C11: C[0:2, 0:4]
    C12: C[0:2, 4:7]
    C21: C[2:4, 0:4]
    C22: C[2:4, 4:7]
    
    Term 1: A11 * B11 -> C11
    A11 indices (i, j) where i in {0,1}, j in {0,1}.
    B11 indices (j, k) where j in {0,1}, k in {0,1,2,3}.
    C11 indices (i, k) where i in {0,1}, k in {0,1,2,3}.
    
    For each i in {0,1}, j in {0,1}, k in {0,1,2,3}:
    U[i*4 + j] = 1
    V[j*7 + k] = 1
    W[k*4 + i] = 1
    Rank contribution: 2 * 2 * 4 = 16? No, wait.
    
    Let's re-evaluate the rank of a block multiplication.
    The tensor for multiplying (r1 x c1) by (c1 x c2) has rank r1 * c1 * c2.
    Here:
    A11 is 2x2. B11 is 2x4. Product is 2x4. Rank = 2*2*4 = 16.
    A12 is 2x2. B21 is 2x4. Product is 2x4. Rank = 2*2*4 = 16.
    A11 is 2x2. B12 is 2x3. Product is 2x3. Rank = 2*2*3 = 12.
    A12 is 2x2. B22 is 2x3. Product is 2x3. Rank = 2*2*3 = 12.
    A21 is 2x2. B11 is 2x4. Product is 2x4. Rank = 2*2*4 = 16.
    A22 is 2x2. B21 is 2x4. Product is 2x4. Rank = 2*2*4 = 16.
    A21 is 2x2. B12 is 2x3. Product is 2x3. Rank = 2*2*3 = 12.
    A22 is 2x2. B22 is 2x3. Product is 2x3. Rank = 2*2*3 = 12.
    
    Total Rank = 16 + 16 + 12 + 12 + 16 + 16 + 12 + 12 = 112.
    
    This is exactly the schoolbook rank. The block structure didn't help reduce the rank because each block multiplication is still done via schoolbook.
    
    To get a lower rank, we need to use a more efficient algorithm for the block multiplications themselves, or find a different decomposition.
    However, for 2x2 * 2x4, the minimum rank is likely 16 (since 2x2 * 2x2 is 8, and we have two such blocks side-by-side).
    Actually, 2x2 * 2x2 has rank 8.
    2x2 * 2x4 can be seen as two 2x2 * 2x2 multiplications. Rank 8 + 8 = 16.
    So the block decomposition doesn't save rank compared to schoolbook.
    
    Is there a way to reduce the rank?
    Winograd's algorithm for 2x2 matrix multiplication reduces rank from 8 to 7.
    If we can apply Winograd to the 2x2 blocks, we might save rank.
    
    Let's try to implement Winograd's algorithm for the 2x2 * 2x2 blocks and extend it to the rectangular cases.
    
    Winograd's algorithm for C = A * B where A, B, C are 2x2:
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C = [[c00, c01], [c10, c11]]
    
    Computations:
    m0 = (a00 + a11) * (b00 + b11)
    m1 = (a10 + a11) * b00
    m2 = a00 * (b10 - b00)
    m3 = a11 * (b01 - b11)
    m4 = (a01 + a11) * b11
    m5 = (a00 - a10) * (b10 + b11)
    m6 = (a01 - a11) * (b00 + b10)
    
    c00 = m0 - m1 + m2
    c01 = m3 + m4
    c10 = m1 + m5
    c11 = m0 - m4 + m6
    
    This uses 7 multiplications instead of 8.
    
    Can we extend this to 2x2 * 2x4?
    B is 2x4. We can split B into two 2x2 blocks: B = [B_left, B_right].
    C = A * B = [A * B_left, A * B_right].
    We can use Winograd for A * B_left and A * B_right.
    Each uses 7 multiplications. Total 14 multiplications.
    Schoolbook for 2x2 * 2x4 is 16 multiplications.
    So we save 2 multiplications per 2x2 * 2x4 block.
    
    Similarly for 2x2 * 2x3:
    B is 2x3. We can split B into B_left (2x2) and B_right (2x1).
    A * B_left uses 7 multiplications.
    A * B_right uses 2 multiplications (schoolbook, since 2x2 * 2x1 is 4, but Winograd for 2x2*2x1? No, Winograd is for 2x2*2x2).
    Actually, 2x2 * 2x1 is just 4 multiplications.
    Total 7 + 4 = 11 multiplications.
    Schoolbook for 2x2 * 2x3 is 12 multiplications.
    So we save 1 multiplication per 2x2 * 2x3 block.
    
    Let's calculate the total rank with this approach:
    4 blocks of 2x2 * 2x4: 4 * 14 = 56.
    4 blocks of 2x2 * 2x3: 4 * 11 = 44.
    Total Rank = 56 + 44 = 100.
    
    This is better than 112.
    
    Now, we need to construct the factor matrices U, V, W for this decomposition.
    This is more complex because the Winograd algorithm involves sums and differences of elements, not just products.
    The tensor decomposition must be exact.
    
    Let's define the Winograd terms for a 2x2 * 2x2 block.
    A = [[a0, a1], [a2, a3]]
    B = [[b0, b1], [b2, b3]]
    
    Terms:
    t0 = (a0 + a3) * (b0 + b3)
    t1 = (a2 + a3) * b0
    t2 = a0 * (b2 - b0)
    t3 = a3 * (b1 - b3)
    t4 = (a1 + a3) * b3
    t5 = (a0 - a2) * (b2 + b3)
    t6 = (a1 - a3) * (b0 + b2)
    
    C00 = t0 - t1 + t2
    C01 = t3 + t4
    C10 = t1 + t5
    C11 = t0 - t4 + t6
    
    Each term tk is a product of a linear combination of A elements and a linear combination of B elements.
    In the tensor decomposition, each term tk corresponds to a rank-1 component.
    U vector for tk: coefficients of A elements.
    V vector for tk: coefficients of B elements.
    W vector for tk: coefficients of C elements (with signs).
    
    For t0:
    U: [1, 0, 0, 1] (for a0, a1, a2, a3)
    V: [1, 0, 0, 1] (for b0, b1, b2, b3)
    W: [1, 0, 0, -1] (for c00, c01, c10, c11) -> c00 gets +1, c11 gets -1.
    
    For t1:
    U: [0, 0, 1, 1]
    V: [1, 0, 0, 0]
    W: [-1, 0, 1, 0]
    
    For t2:
    U: [1, 0, 0, 0]
    V: [0, 0, 1, 0] -> wait, b2 - b0. So V: [-1, 0, 1, 0]
    W: [1, 0, 0, 0]
    
    For t3:
    U: [0, 0, 0, 1]
    V: [0, 1, 0, -1]
    W: [0, 1, 0, 0]
    
    For t4:
    U: [0, 1, 0, 1]
    V: [0, 0, 0, 1]
    W: [0, 1, 0, -1]
    
    For t5:
    U: [1, 0, -1, 0]
    V: [0, 0, 1, 1]
    W: [0, 0, 1, 0]
    
    For t6:
    U: [0, 1, 0, -1]
    V: [1, 0, 1, 0]
    W: [0, 0, 0, 1]
    
    This gives a rank-7 decomposition for 2x2 * 2x2.
    
    Now, let's extend this to the full 4x4 * 4x7 problem.
    We will use the block structure described above.
    
    Block A11 (2x2) * Block B11 (2x4) -> Block C11 (2x4)
    We split B11 into B11_left (2x2) and B11_right (2x2).
    We use Winograd for A11 * B11_left and A11 * B11_right.
    Each uses 7 terms. Total 14 terms.
    
    Similarly for other blocks.
    
    Let's implement this.
    
    We need to map the local indices of the blocks to the global indices of U, V, W.
    
    Global U indices: i*4 + j, where i in 0..3, j in 0..3.
    Global V indices: j*7 + k, where j in 0..3, k in 0..6.
    Global W indices: k*4 + i, where i in 0..3, k in 0..6.
    
    Let's define the blocks and their local to global index mappings.
    
    A11: i in {0,1}, j in {0,1}. Local (r, c) -> Global (r, c).
    A12: i in {0,1}, j in {2,3}. Local (r, c) -> Global (r, c+2).
    A21: i in {2,3}, j in {0,1}. Local (r, c) -> Global (r+2, c).
    A22: i in {2,3}, j in {2,3}. Local (r, c) -> Global (r+2, c+2).
    
    B11: j in {0,1}, k in {0,3}. Local (r, c) -> Global (r, c).
    B12: j in {0,1}, k in {4,6}. Local (r, c) -> Global (r, c+4).
    B21: j in {2,3}, k in {0,3}. Local (r, c) -> Global (r+2, c).
    B22: j in {2,3}, k in {4,6}. Local (r, c) -> Global (r+2, c+4).
    
    C11: i in {0,1}, k in {0,3}. Local (r, c) -> Global (r, c).
    C12: i in {0,1}, k in {4,6}. Local (r, c) -> Global (r, c+4).
    C21: i in {2,3}, k in {0,3}. Local (r, c) -> Global (r+2, c).
    C22: i in {2,3}, k in {4,6}. Local (r, c) -> Global (r+2, c+4).
    
    Now, let's define the Winograd terms for a 2x2 * 2x2 block.
    We will create a function that takes the global index mappings for A, B, C and the Winograd term index (0..6), and adds the corresponding entries to U, V, W.
    
    Winograd terms for 2x2 * 2x2:
    t0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,-1]
    t1: U=[0,0,1,1], V=[1,0,0,0], W=[-1,0,1,0]
    t2: U=[1,0,0,0], V=[-1,0,1,0], W=[1,0,0,0]
    t3: U=[0,0,0,1], V=[0,1,0,-1], W=[0,1,0,0]
    t4: U=[0,1,0,1], V=[0,0,0,1], W=[0,1,0,-1]
    t5: U=[1,0,-1,0], V=[0,0,1,1], W=[0,0,1,0]
    t6: U=[0,1,0,-1], V=[1,0,1,0], W=[0,0,0,1]
    
    We need to apply this to each block multiplication.
    
    Block A11 * B11_left (2x2 * 2x2) -> C11_left (2x2)
    A11 local indices: (0,0), (0,1), (1,0), (1,1) -> Global (0,0), (0,1), (1,0), (1,1)
    B11_left local indices: (0,0), (0,1), (1,0), (1,1) -> Global (0,0), (0,1), (1,0), (1,1)
    C11_left local indices: (0,0), (0,1), (1,0), (1,1) -> Global (0,0), (0,1), (1,0), (1,1)
    
    Block A11 * B11_right (2x2 * 2x2) -> C11_right (2x2)
    A11 local indices: same as above.
    B11_right local indices: (0,0), (0,1), (1,0), (1,1) -> Global (0,2), (0,3), (1,2), (1,3)
    C11_right local indices: (0,0), (0,1), (1,0), (1,1) -> Global (0,2), (0,3), (1,2), (1,3)
    
    And so on for other blocks.
    
    Let's write the code to construct U, V, W.
    
    We will use lists of lists for U, V, W.
    U shape: 16 x rank
    V shape: 28 x rank
    W shape: 28 x rank
    
    We will keep track of the current rank (column index).
    
    Let's define a helper function to add a Winograd term.
    
    def add_winograd_term(U, V, W, a_idx, b_idx, c_idx, term):
        # a_idx, b_idx, c_idx are lists of 4 global indices for A, B, C elements.
        # term is 0..6.
        # We need to add the coefficients to U, V, W at the current column.
        pass
    
    Let's implement this.
    
    First, let's determine the total rank.
    4 blocks of 2x2 * 2x4: 4 * 14 = 56.
    4 blocks of 2x2 * 2x3: 4 * 11 = 44.
    Total Rank = 100.
    
    Let's verify the 2x2 * 2x3 case.
    B is 2x3. We split into B_left (2x2) and B_right (2x1).
    A * B_left uses 7 terms.
    A * B_right uses 4 terms (schoolbook).
    Total 11 terms.
    
    For A * B_right (2x2 * 2x1):
    A = [[a0, a1], [a2, a3]]
    B = [[b0], [b1]]
    C = [[c0], [c1]]
    c0 = a0*b0 + a1*b1
    c1 = a2*b0 + a3*b1
    
    Terms:
    t0 = a0 * b0
    t1 = a1 * b1
    t2 = a2 * b0
    t3 = a3 * b1
    
    U for t0: [1,0,0,0], V for t0: [1,0], W for t0: [1,0]
    U for t1: [0,1,0,0], V for t1: [0,1], W for t1: [1,0]
    U for t2: [0,0,1,0], V for t2: [1,0], W for t2: [0,1]
    U for t3: [0,0,0,1], V for t3: [0,1], W for t3: [0,1]
    
    This is correct.
    
    Now, let's implement the full construction.
    
    We will create a list of blocks to process.
    Each block is defined by:
    - A block indices (i_start, i_end, j_start, j_end)
    - B block indices (j_start, j_end, k_start, k_end)
    - C block indices (i_start, i_end, k_start, k_end)
    - Type: '2x4' or '2x3'
    
    For '2x4':
    Split B into two 2x2 blocks.
    Apply Winograd to each.
    
    For '2x3':
    Split B into one 2x2 block and one 2x1 block.
    Apply Winograd to the 2x2 block.
    Apply schoolbook to the 2x1 block.
    
    Let's code this.
    
    """
    n, m, p = 4, 4, 7
    rank = 100
    U = [[0] * rank for _ in range(n * m)]  # 16 x 100
    V = [[0] * rank for _ in range(m * p)]  # 28 x 100
    W = [[0] * rank for _ in range(p * n)]  # 28 x 100
    
    col = 0
    
    # Define Winograd terms for 2x2 * 2x2
    winograd_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),  # t0
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 0, 1, 0]),   # t1
        ([1, 0, 0, 0], [-1, 0, 1, 0], [1, 0, 0, 0]),   # t2
        ([0, 0, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0]),   # t3
        ([0, 1, 0, 1], [0, 0, 0, 1], [0, 1, 0, -1]),   # t4
        ([1, 0, -1, 0], [0, 0, 1, 1], [0, 0, 1, 0]),   # t5
        ([0, 1, 0, -1], [1, 0, 1, 0], [0, 0, 0, 1]),   # t6
    ]
    
    # Define schoolbook terms for 2x2 * 2x1
    schoolbook_terms_2x1 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),  # a0*b0 -> c0
        ([0, 1, 0, 0], [0, 1], [1, 0]),  # a1*b1 -> c0
        ([0, 0, 1, 0], [1, 0], [0, 1]),  # a2*b0 -> c1
        ([0, 0, 0, 1], [0, 1], [0, 1]),  # a3*b1 -> c1
    ]
    
    def add_term(a_indices, b_indices, c_indices, u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i, idx in enumerate(a_indices):
            U[idx][col] += u_coeffs[i]
        for i, idx in enumerate(b_indices):
            V[idx][col] += v_coeffs[i]
        for i, idx in enumerate(c_indices):
            W[idx][col] += w_coeffs[i]
        col += 1
    
    # Define block index mappings
    # A block (i_start, j_start)
    def get_a_indices(i_start, j_start):
        return [i_start*4 + j_start, i_start*4 + j_start + 1, i_start + 1*4 + j_start, i_start + 1*4 + j_start + 1]
    
    # B block (j_start, k_start)
    def get_b_indices(j_start, k_start):
        return [j_start*7 + k_start, j_start*7 + k_start + 1, j_start + 1*7 + k_start, j_start + 1*7 + k_start + 1]
    
    # C block (i_start, k_start)
    def get_c_indices(i_start, k_start):
        return [k_start*4 + i_start, k_start*4 + i_start + 1, (k_start + 1)*4 + i_start, (k_start + 1)*4 + i_start + 1]
    
    # Process A11 * B11 -> C11
    # B11 is 2x4, split into B11_left (2x2) and B11_right (2x2)
    a11_idx = get_a_indices(0, 0)
    
    # B11_left: j in {0,1}, k in {0,1}
    b11l_idx = get_b_indices(0, 0)
    c11l_idx = get_c_indices(0, 0)
    for term in winograd_terms:
        add_term(a11_idx, b11l_idx, c11l_idx, *term)
    
    # B11_right: j in {0,1}, k in {2,3}
    b11r_idx = get_b_indices(0, 2)
    c11r_idx = get_c_indices(0, 2)
    for term in winograd_terms:
        add_term(a11_idx, b11r_idx, c11r_idx, *term)
    
    # Process A12 * B21 -> C11
    # B21 is 2x4, split into B21_left (2x2) and B21_right (2x2)
    a12_idx = get_a_indices(0, 2)
    
    # B21_left: j in {2,3}, k in {0,1}
    b21l_idx = get_b_indices(2, 0)
    c11l_idx = get_c_indices(0, 0)
    for term in winograd_terms:
        add_term(a12_idx, b21l_idx, c11l_idx, *term)
    
    # B21_right: j in {2,3}, k in {2,3}
    b21r_idx = get_b_indices(2, 2)
    c11r_idx = get_c_indices(0, 2)
    for term in winograd_terms:
        add_term(a12_idx, b21r_idx, c11r_idx, *term)
    
    # Process A11 * B12 -> C12
    # B12 is 2x3, split into B12_left (2x2) and B12_right (2x1)
    a11_idx = get_a_indices(0, 0)
    
    # B12_left: j in {0,1}, k in {4,5}
    b12l_idx = get_b_indices(0, 4)
    c12l_idx = get_c_indices(0, 4)
    for term in winograd_terms:
        add_term(a11_idx, b12l_idx, c12l_idx, *term)
    
    # B12_right: j in {0,1}, k in {6}
    # B12_right is 2x1, indices: (0,6), (1,6)
    b12r_idx = [0*7 + 6, 1*7 + 6]
    c12r_idx = [6*4 + 0, 6*4 + 1]
    for term in schoolbook_terms_2x1:
        add_term(a11_idx, b12r_idx, c12r_idx, *term)
    
    # Process A12 * B22 -> C12
    # B22 is 2x3, split into B22_left (2x2) and B22_right (2x1)
    a12_idx = get_a_indices(0, 2)
    
    # B22_left: j in {2,3}, k in {4,5}
    b22l_idx = get_b_indices(2, 4)
    c12l_idx = get_c_indices(0, 4)
    for term in winograd_terms:
        add_term(a12_idx, b22l_idx, c12l_idx, *term)
    
    # B22_right: j in {2,3}, k in {6}
    b22r_idx = [2*7 + 6, 3*7 + 6]
    c12r_idx = [6*4 + 0, 6*4 + 1]
    for term in schoolbook_terms_2x1:
        add_term(a12_idx, b22r_idx, c12r_idx, *term)
    
    # Process A21 * B11 -> C21
    # B11 is 2x4, split into B11_left (2x2) and B11_right (2x2)
    a21_idx = get_a_indices(2, 0)
    
    # B11_left: j in {0,1}, k in {0,1}
    b11l_idx = get_b_indices(0, 0)
    c21l_idx = get_c_indices(2, 0)
    for term in winograd_terms:
        add_term(a21_idx, b11l_idx, c21l_idx, *term)
    
    # B11_right: j in {0,1}, k in {2,3}
    b11r_idx = get_b_indices(0, 2)
    c21r_idx = get_c_indices(2, 2)
    for term in winograd_terms:
        add_term(a21_idx, b11r_idx, c21r_idx, *term)
    
    # Process A22 * B21 -> C21
    # B21 is 2x4, split into B21_left (2x2) and B21_right (2x2)
    a22_idx = get_a_indices(2, 2)
    
    # B21_left: j in {2,3}, k in {0,1}
    b21l_idx = get_b_indices(2, 0)
    c21l_idx = get_c_indices(2, 0)
    for term in winograd_terms:
        add_term(a22_idx, b21l_idx, c21l_idx, *term)
    
    # B21_right: j in {2,3}, k in {2,3}
    b21r_idx = get_b_indices(2, 2)
    c21r_idx = get_c_indices(2, 2)
    for term in winograd_terms:
        add_term(a22_idx, b21r_idx, c21r_idx, *term)
    
    # Process A21 * B12 -> C22
    # B12 is 2x3, split into B12_left (2x2) and B12_right (2x1)
    a21_idx = get_a_indices(2, 0)
    
    # B12_left: j in {0,1}, k in {4,5}
    b12l_idx = get_b_indices(0, 4)
    c22l_idx = get_c_indices(2, 4)
    for term in winograd_terms:
        add_term(a21_idx, b12l_idx, c22l_idx, *term)
    
    # B12_right: j in {0,1}, k in {6}
    b12r_idx = [0*7 + 6, 1*7 + 6]
    c22r_idx = [6*4 + 2, 6*4 + 3]
    for term in schoolbook_terms_2x1:
        add_term(a21_idx, b12r_idx, c22r_idx, *term)
    
    # Process A22 * B22 -> C22
    # B22 is 2x3, split into B22_left (2x2) and B22_right (2x1)
    a22_idx = get_a_indices(2, 2)
    
    # B22_left: j in {2,3}, k in {4,5}
    b22l_idx = get_b_indices(2, 4)
    c22l_idx = get_c_indices(2, 4)
    for term in winograd_terms:
        add_term(a22_idx, b22l_idx, c22l_idx, *term)
    
    # B22_right: j in {2,3}, k in {6}
    b22r_idx = [2*7 + 6, 3*7 + 6]
    c22r_idx = [6*4 + 2, 6*4 + 3]
    for term in schoolbook_terms_2x1:
        add_term(a22_idx, b22r_idx, c22r_idx, *term)
    
    return {"U": U, "V": V, "W": W}

