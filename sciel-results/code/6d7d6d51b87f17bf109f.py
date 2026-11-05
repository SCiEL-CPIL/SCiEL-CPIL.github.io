import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns the Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    U: 4x7, V: 4x7, W: 4x7
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen formulas for C = AB:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # U: A00 (idx 0), A11 (idx 3)
    U[0][0] = 1
    U[3][0] = 1
    # V: B00 (idx 0), B11 (idx 3)
    V[0][0] = 1
    V[3][0] = 1
    # W: C00 (idx 0), C11 (idx 3)
    W[0][0] = 1
    W[3][0] = 1
    
    # M2: (A10+A11)B00
    # U: A10 (idx 2), A11 (idx 3)
    U[2][1] = 1
    U[3][1] = 1
    # V: B00 (idx 0)
    V[0][1] = 1
    # W: C10 (idx 2)
    W[2][1] = 1
    
    # M3: A00(B10-B11)
    # U: A00 (idx 0)
    U[0][2] = 1
    # V: B10 (idx 2), -B11 (idx 3)
    V[2][2] = 1
    V[3][2] = -1
    # W: C01 (idx 1)
    W[1][2] = 1
    
    # M4: A11(B01-B10)
    # U: A11 (idx 3)
    U[3][3] = 1
    # V: B01 (idx 1), -B10 (idx 2)
    V[1][3] = 1
    V[2][3] = -1
    # W: C00 (idx 0), C10 (idx 2)
    W[0][3] = 1
    W[2][3] = 1
    
    # M5: (A00+A01)B11
    # U: A00 (idx 0), A01 (idx 1)
    U[0][4] = 1
    U[1][4] = 1
    # V: B11 (idx 3)
    V[3][4] = 1
    # W: C00 (idx 0), C01 (idx 1)
    W[0][4] = 1
    W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10 (idx 2), -A00 (idx 0)
    U[2][5] = 1
    U[0][5] = -1
    # V: B00 (idx 0), B01 (idx 1)
    V[0][5] = 1
    V[1][5] = 1
    # W: C11 (idx 3)
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01 (idx 1), -A11 (idx 3)
    U[1][6] = 1
    U[3][6] = -1
    # V: B10 (idx 2), B11 (idx 3)
    V[2][6] = 1
    V[3][6] = 1
    # W: C00 (idx 0), C11 (idx 3)
    W[0][6] = 1
    W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_for(n, m, p):
    """Generic schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
    return _schoolbook_for(n, m, p)

def _verify_fast(decomp, n, m, p):
    """Fast verification using NumPy matrix operations."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[u, v, w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # and compare with the expected matmul tensor.
    
    # Expected tensor: T[u, v, w] = 1 if j_u == j_v and k_v == k_w and i_w == i_u
    # where u = i*m+j, v = j*p+k, w = k*n+i
    
    T = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
    
    # Build expected tensor
    expected = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                expected[u, v, w] = 1.0
    
    if not np.allclose(T, expected, atol=1e-9):
        return False
    return True

def _block_decomposition():
    """
    Construct a block decomposition for 5x5x5.
    Partition indices 0..4 into blocks:
    B0: {0, 1} (size 2)
    B1: {2, 3} (size 2)
    B2: {4}    (size 1)
    
    Total rank = sum over block pairs (a, k, b) of rank needed for that block interaction.
    Using Strassen for 2x2x2 blocks (rank 7) and schoolbook for others.
    
    Total rank calculation:
    a=0 (size 2):
      k=0 (size 2): b=0(7), b=1(7), b=2(4) -> 18
      k=1 (size 2): b=0(7), b=1(7), b=2(4) -> 18
      k=2 (size 1): b=0(4), b=1(4), b=2(2) -> 10
      Sum a=0: 46
    
    a=1 (size 2):
      k=0 (size 2): 18
      k=1 (size 2): 18
      k=2 (size 1): 10
      Sum a=1: 46
    
    a=2 (size 1):
      k=0 (size 2): b=0(4), b=1(4), b=2(2) -> 10
      k=1 (size 2): b=0(4), b=1(4), b=2(2) -> 10
      k=2 (size 1): b=0(2), b=1(2), b=2(1) -> 5
      Sum a=2: 25
    
    Total = 46 + 46 + 25 = 117.
    """
    n, m, p = DIMENSIONS
    # Block sizes
    sizes = [2, 2, 1]
    num_blocks = 3
    
    # Precompute offsets
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    
    # Total rank
    total_rank = 0
    # We need to store the decomposition for each block interaction
    # To assemble the final U, V, W matrices, we need to map local indices to global indices.
    
    # Let's build the matrices directly.
    # First, determine the total rank to allocate columns.
    # We'll use a list of (U_block, V_block, W_block, u_offset, v_offset, w_offset)
    
    block_decomps = []
    
    for a in range(num_blocks):
        for k in range(num_blocks):
            for b in range(num_blocks):
                sa = sizes[a]
                sk = sizes[k]
                sb = sizes[b]
                
                # Get decomposition for sa x sk * sk x sb -> sa x sb
                if sa == 2 and sk == 2 and sb == 2:
                    decomp = _strassen_2x2()
                else:
                    # Schoolbook
                    decomp = _schoolbook_for(sa, sk, sb)
                
                # Store with offsets
                # U block corresponds to A block (a, k). Rows are local i in [0, sa), local j in [0, sk).
                # Global U index: (offsets[a] + i) * m + (offsets[k] + j)
                # V block corresponds to B block (k, b). Rows are local j in [0, sk), local k in [0, sb).
                # Global V index: (offsets[k] + j) * p + (offsets[b] + k)
                # W block corresponds to C block (a, b). Rows are local k in [0, sb), local i in [0, sa).
                # Global W index: (offsets[b] + k) * n + (offsets[a] + i)
                
                block_decomps.append({
                    "U": decomp["U"],
                    "V": decomp["V"],
                    "W": decomp["W"],
                    "u_off_a": offsets[a],
                    "u_off_k": offsets[k],
                    "v_off_k": offsets[k],
                    "v_off_b": offsets[b],
                    "w_off_b": offsets[b],
                    "w_off_a": offsets[a],
                    "sa": sa,
                    "sk": sk,
                    "sb": sb
                })
                
                total_rank += len(decomp["U"][0]) if decomp["U"] else 0
    
    # Allocate final matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_col = 0
    for bd in block_decomps:
        sa, sk, sb = bd["sa"], bd["sk"], bd["sb"]
        U_block = bd["U"]
        V_block = bd["V"]
        W_block = bd["W"]
        rank_block = len(U_block[0])
        
        for local_i in range(sa):
            for local_j in range(sk):
                global_u_row = (bd["u_off_a"] + local_i) * m + (bd["u_off_k"] + local_j)
                for r in range(rank_block):
                    U[global_u_row][current_col + r] = U_block[local_i * sk + local_j][r]
        
        for local_j in range(sk):
            for local_k in range(sb):
                global_v_row = (bd["v_off_k"] + local_j) * p + (bd["v_off_b"] + local_k)
                for r in range(rank_block):
                    V[global_v_row][current_col + r] = V_block[local_j * sb + local_k][r]
        
        for local_k in range(sb):
            for local_i in range(sa):
                global_w_row = (bd["w_off_b"] + local_k) * n + (bd["w_off_a"] + local_i)
                for r in range(rank_block):
                    W[global_w_row][current_col + r] = W_block[local_k * sa + local_i][r]
        
        current_col += rank_block
    
    return {"U": U, "V": V, "W": W}

def _try_4x4_strassen_variant():
    """
    Try a decomposition based on splitting 5x5 into 4x4 + 1x1 blocks.
    The 4x4x4 block can potentially use a rank-49 decomposition (Strassen-like for 4x4).
    However, standard Strassen gives rank 7 for 2x2, and for 4x4 it would be 7^2 = 49 if we
    apply Strassen twice? No, that's not how it works directly.
    
    Actually, a 4x4x4 matmul can be done with rank 49 using the standard Strassen recursion
    (split 4x4 into four 2x2 blocks, use 7 multiplications of 2x2 blocks, each of rank 7,
    total 49). But wait, the rank of the tensor for 4x4x4 is at most 4*4*4=64 (schoolbook).
    Using Strassen on 4x4: split into 2x2 blocks. We have 4x4 = [2x2 2x2; 2x2 2x2].
    This requires 7 multiplications of 2x2 matrices. Each 2x2 multiplication has rank 7.
    So total rank is 7 * 7 = 49.
    
    So for the 4x4x4 part, rank is 49.
    The cross terms and 1x1 terms need to be handled.
    
    Partition: B0 = {0,1,2,3} (size 4), B1 = {4} (size 1).
    
    Blocks:
    a=0 (4), k=0 (4), b=0 (4): 4x4 * 4x4 -> 4x4. Rank 49.
    a=0 (4), k=0 (4), b=1 (1): 4x4 * 4x1 -> 4x1. Rank 16 (schoolbook 4*4*1).
    a=0 (4), k=1 (1), b=0 (4): 4x1 * 1x4 -> 4x4. Rank 16 (schoolbook 4*1*4).
    a=0 (4), k=1 (1), b=1 (1): 4x1 * 1x1 -> 4x1. Rank 4 (schoolbook 4*1*1).
    
    a=1 (1), k=0 (4), b=0 (4): 1x4 * 4x4 -> 1x4. Rank 16 (schoolbook 1*4*4).
    a=1 (1), k=0 (4), b=1 (1): 1x4 * 4x1 -> 1x1. Rank 4 (schoolbook 1*4*1).
    a=1 (1), k=1 (1), b=0 (4): 1x1 * 1x4 -> 1x4. Rank 4 (schoolbook 1*1*4).
    a=1 (1), k=1 (1), b=1 (1): 1x1 * 1x1 -> 1x1. Rank 1.
    
    Total rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110.
    
    This is better than 117! Let's implement this.
    We need a rank-49 decomposition for 4x4x4.
    We can construct it by applying Strassen to the 4x4 matrix, treating it as a 2x2 block matrix
    of 2x2 blocks.
    
    The 4x4 matrix A can be written as:
    A = [A00 A01; A10 A11] where each Aij is 2x2.
    Similarly for B and C.
    
    The Strassen formulas for the block matrix multiplication C = AB:
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M6 + M7
    
    Where M1..M7 are products of 2x2 blocks.
    Each Mj = Xj * Yj where Xj and Yj are linear combinations of the 2x2 blocks of A and B.
    Since each 2x2 multiplication has rank 7, and we have 7 such multiplications, the total rank
    is 7 * 7 = 49.
    
    We need to construct the U, V, W matrices for the 4x4x4 decomposition.
    The indices for 4x4x4:
    U: 16 x 49 (4*4 rows, 49 cols)
    V: 16 x 49
    W: 16 x 49
    
    The 16 rows of U correspond to A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ..., A[3,3].
    We can group them into 2x2 blocks:
    Block (0,0): A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0, 1, 4, 5
    Block (0,1): A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2, 3, 6, 7
    Block (1,0): A[2,0], A[2,1], A[3,0], A[3,1] -> indices 8, 9, 12, 13
    Block (1,1): A[2,2], A[2,3], A[3,2], A[3,3] -> indices 10, 11, 14, 15
    
    Similarly for V and W.
    
    For each of the 7 Strassen multiplications Mj = Xj * Yj, we have a rank-7 decomposition
    for the 2x2x2 block multiplication.
    So for each j, we have 7 rank-1 terms.
    The U vector for term (j, r) is the U vector for the 2x2 decomposition, placed in the
    appropriate block of the 4x4 U matrix.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS
    
    # We'll build the 4x4x4 decomposition first, then combine with the 1x1 parts.
    
    # First, let's define the 4x4x4 decomposition using Strassen on 2x2 blocks.
    
    # The 7 Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For each Mj, we need to express Xj and Yj in terms of the 2x2 blocks.
    # Xj is a linear combination of A00, A01, A10, A11 (each 2x2).
    # Yj is a linear combination of B00, B01, B10, B11 (each 2x2).
    
    # Let's define the coefficients for each Mj.
    # For Mj, Xj = sum_{i} c_ia * A_i, Yj = sum_{j} c_jb * B_j.
    # Then Mj = Xj * Yj = (sum_i c_ia A_i) * (sum_j c_jb B_j) = sum_{i,j} c_ia c_jb (A_i B_j).
    
    # But A_i B_j is a product of two 2x2 matrices, which has rank 7.
    # So Mj = sum_{i,j} c_ia c_jb (A_i B_j).
    # Each A_i B_j is a 2x2x2 matmul tensor, which we can decompose with rank 7.
    
    # So the total rank for Mj is 7 * 7 = 49? No, that's not right.
    # Actually, Mj = Xj * Yj where Xj and Yj are 2x2 matrices (linear combinations of blocks).
    # The product Xj * Yj is a 2x2 matrix, and its computation requires 7 scalar multiplications
    # (using Strassen for 2x2).
    # So the rank of the tensor for Mj is 7.
    # And since we have 7 such Mj's, the total rank is 7 * 7 = 49.
    
    # Wait, I need to be more careful.
    # The 4x4x4 matmul tensor is the sum of the tensors for each Mj.
    # Each Mj tensor has rank 7 (since it's a 2x2x2 matmul of the blocks Xj and Yj).
    # So the total rank is 7 * 7 = 49.
    
    # Let's construct the U, V, W matrices for the 4x4x4 decomposition.
    
    # For each of the 7 Strassen multiplications, we have a 2x2x2 decomposition with rank 7.
    # Let's denote the decomposition for Mj as (U_j, V_j, W_j) where each is 4x7.
    # The rows of U_j correspond to the 4 entries of Xj (which is a 2x2 matrix).
    # The rows of V_j correspond to the 4 entries of Yj.
    # The rows of W_j correspond to the 4 entries of the result C_block.
    
    # But Xj is a linear combination of the 2x2 blocks of A.
    # So the U matrix for the 4x4x4 decomposition will have 16 rows (for the 4x4 A matrix).
    # For a given Mj, the U vector for a particular Strassen term r will have non-zero entries
    # only in the rows corresponding to the blocks that appear in Xj.
    
    # Let's define the block structure:
    # A blocks: A00 (rows 0,1; cols 0,1), A01 (rows 0,1; cols 2,3), A10 (rows 2,3; cols 0,1), A11 (rows 2,3; cols 2,3)
    # In terms of the 16 rows of U (for 4x4 A):
    # A00: rows 0,1,4,5
    # A01: rows 2,3,6,7
    # A10: rows 8,9,12,13
    # A11: rows 10,11,14,15
    
    # Similarly for B blocks (V matrix, 16 rows):
    # B00: rows 0,1,4,5
    # B01: rows 2,3,6,7
    # B10: rows 8,9,12,13
    # B11: rows 10,11,14,15
    
    # And for C blocks (W matrix, 16 rows):
    # C00: rows 0,1,4,5
    # C01: rows 2,3,6,7
    # C10: rows 8,9,12,13
    # C11: rows 10,11,14,15
    
    # Now, for each Mj, we need to define Xj and Yj in terms of the blocks.
    # M1: X1 = A00 + A11, Y1 = B00 + B11
    # M2: X2 = A10 + A11, Y2 = B00
    # M3: X3 = A00, Y3 = B10 - B11
    # M4: X4 = A11, Y4 = B01 - B10
    # M5: X5 = A00 + A01, Y5 = B11
    # M6: X6 = A10 - A00, Y6 = B00 + B01
    # M7: X7 = A01 - A11, Y7 = B10 + B11
    
    # For each Mj, we have a 2x2x2 Strassen decomposition with rank 7.
    # Let's get the Strassen decomposition for 2x2x2.
    strassen = _strassen_2x2()
    U_s = strassen["U"]  # 4x7
    V_s = strassen["V"]  # 4x7
    W_s = strassen["W"]  # 4x7
    
    # Now, for each Mj, we construct the U, V, W vectors for the 4x4x4 decomposition.
    # The U matrix for 4x4x4 will have 16 rows and 49 columns (7 Mj's * 7 Strassen terms).
    
    U_4x4 = [[0]*49 for _ in range(16)]
    V_4x4 = [[0]*49 for _ in range(16)]
    W_4x4 = [[0]*49 for _ in range(16)]
    
    # Define the block row indices for A, B, C
    # A blocks: A00, A01, A10, A11
    A_block_rows = {
        'A00': [0, 1, 4, 5],
        'A01': [2, 3, 6, 7],
        'A10': [8, 9, 12, 13],
        'A11': [10, 11, 14, 15]
    }
    B_block_rows = {
        'B00': [0, 1, 4, 5],
        'B01': [2, 3, 6, 7],
        'B10': [8, 9, 12, 13],
        'B11': [10, 11, 14, 15]
    }
    C_block_rows = {
        'C00': [0, 1, 4, 5],
        'C01': [2, 3, 6, 7],
        'C10': [8, 9, 12, 13],
        'C11': [10, 11, 14, 15]
    }
    
    # For each Mj, define the coefficients for Xj and Yj.
    # Xj = sum c_ia * A_i, Yj = sum c_jb * B_j
    # The Strassen decomposition for 2x2x2 gives U_s, V_s, W_s.
    # U_s has 4 rows corresponding to the 4 entries of the 2x2 matrix Xj.
    # V_s has 4 rows corresponding to the 4 entries of the 2x2 matrix Yj.
    # W_s has 4 rows corresponding to the 4 entries of the 2x2 result matrix.
    
    # For a given Mj, the U vector for Strassen term r will have non-zero entries in the
    # rows corresponding to the blocks in Xj.
    # Specifically, if Xj = c_00 * A00 + c_01 * A01 + c_10 * A10 + c_11 * A11,
    # then the U vector for term r will have:
    # U_4x4[row][col] = c_ij * U_s[local_row][r]
    # where row is the global row index in A_block_rows[Aij], and local_row is the
    # corresponding local row index in the 2x2 block.
    
    # Let's define the Mj coefficients.
    Mj_coeffs = [
        # (X_coeffs, Y_coeffs) where X_coeffs = {block: coeff}, Y_coeffs = {block: coeff}
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}),  # M1
        ({'A10': 1, 'A11': 1}, {'B00': 1}),             # M2
        ({'A00': 1}, {'B10': 1, 'B11': -1}),            # M3
        ({'A11': 1}, {'B01': 1, 'B10': -1}),            # M4
        ({'A00': 1, 'A01': 1}, {'B11': 1}),             # M5
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}),  # M6
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}),  # M7
    ]
    
    # For each Mj, construct the U, V, W vectors.
    for j in range(7):
        X_coeffs, Y_coeffs = Mj_coeffs[j]
        
        for r in range(7):
            col = j * 7 + r
            
            # Construct U vector for this term
            for block_name, coeff in X_coeffs.items():
                block_rows = A_block_rows[block_name]
                for local_row, global_row in enumerate(block_rows):
                    U_4x4[global_row][col] = coeff * U_s[local_row][r]
            
            # Construct V vector for this term
            for block_name, coeff in Y_coeffs.items():
                block_rows = B_block_rows[block_name]
                for local_row, global_row in enumerate(block_rows):
                    V_4x4[global_row][col] = coeff * V_s[local_row][r]
            
            # Construct W vector for this term
            # The W vector depends on which output block C_block this Mj contributes to.
            # M1 contributes to C00 and C11.
            # M2 contributes to C10.
            # M3 contributes to C01.
            # M4 contributes to C00 and C10.
            # M5 contributes to C00 and C01.
            # M6 contributes to C11.
            # M7 contributes to C00 and C11.
            
            # Let's define the output blocks for each Mj.
            Mj_output_blocks = [
                ['C00', 'C11'],  # M1
                ['C10'],          # M2
                ['C01'],          # M3
                ['C00', 'C10'],  # M4
                ['C00', 'C01'],  # M5
                ['C11'],          # M6
                ['C00', 'C11'],  # M7
            ]
            
            # The W_s vector has 4 rows corresponding to the 4 entries of the 2x2 result.
            # For each output block, we need to place the W_s vector in the appropriate rows.
            # But wait, Mj is a single 2x2 matrix product, so it produces a single 2x2 result.
            # The W_s vector corresponds to this 2x2 result.
            # So for each output block, we place the entire W_s vector in the corresponding rows.
            
            for block_name in Mj_output_blocks[j]:
                block_rows = C_block_rows[block_name]
                for local_row, global_row in enumerate(block_rows):
                    W_4x4[global_row][col] = W_s[local_row][r]
    
    # Now we have the 4x4x4 decomposition with rank 49.
    # Let's verify it.
    decomp_4x4 = {"U": U_4x4, "V": V_4x4, "W": W_4x4}
    
    # We'll verify it later in the main construct function.
    return decomp_4x4

def _block_decomposition_3_2():
    """
    Construct a block decomposition for 5x5x5 using 3x3 + 2x2 partition.
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    
    Block sizes: [3, 2], offsets: [0, 3]
    
    Rank calculation:
    a=0 (size 3):
      k=0 (size 3): b=0 (27), b=1 (18) -> 45
      k=1 (size 2): b=0 (18), b=1 (7) -> 25
      Sum a=0: 70
    
    a=1 (size 2):
      k=0 (size 3): b=0 (18), b=1 (12) -> 30
      k=1 (size 2): b=0 (12), b=1 (7) -> 19
      Sum a=1: 49
    
    Total = 70 + 49 = 119... wait let me recalculate.
    
    Actually for sa x sk * sk x sb:
    - If all are 2: rank 7 (Strassen)
    - Otherwise: rank sa*sk*sb (schoolbook)
    
    a=0 (3):
      k=0 (3): b=0 (3): 3*3*3=27, b=1 (2): 3*3*2=18 -> 45
      k=1 (2): b=0 (3): 3*2*3=18, b=1 (2): 2*2*2=7 -> 25
      Sum: 70
    
    a=1 (2):
      k=0 (3): b=0 (3): 2*3*3=18, b=1 (2): 2*3*2=12 -> 30
      k=1 (2): b=0 (3): 2*2*3=12, b=1 (2): 2*2*2=7 -> 19
      Sum: 49
    
    Total = 119. That's worse than 110!
    
    Let me try 3+1+1:
    sizes = [3, 1, 1], offsets = [0, 3, 4]
    
    a=0 (3):
      k=0 (3): b=0 (27), b=1 (9), b=2 (9) -> 45
      k=1 (1): b=0 (9), b=1 (3), b=2 (3) -> 15
      k=2 (1): b=0 (9), b=1 (3), b=2 (3) -> 15
      Sum: 75
    
    a=1 (1):
      k=0 (3): b=0 (9), b=1 (3), b=2 (3) -> 15
      k=1 (1): b=0 (3), b=1 (1), b=2 (1) -> 5
      k=2 (1): b=0 (3), b=1 (1), b=2 (1) -> 5
      Sum: 25
    
    a=2 (1):
      k=0 (3): b=0 (9), b=1 (3), b=2 (3) -> 15
      k=1 (1): b=0 (3), b=1 (1), b=2 (1) -> 5
      k=2 (1): b=0 (3), b=1 (1), b=2 (1) -> 5
      Sum: 25
    
    Total = 75 + 25 + 25 = 125. Same as schoolbook!
    
    What about 2+2+1 (already done): 117
    What about 4+1: 110
    What about 3+2: 119
    
    So 4+1 with Strassen on 4x4x4 gives 110, which is the best so far.
    But wait - can we do better with a different approach?
    
    Let me try using Strassen more cleverly. What if we use a 3x3 block with a rank-19 decomposition?
    Actually, the minimum rank for 3x3x3 is 19 (known result).
    
    Let me try 3+2 with rank-19 for 3x3x3:
    a=0 (3):
      k=0 (3): b=0 (19), b=1 (18) -> 37
      k=1 (2): b=0 (18), b=1 (7) -> 25
      Sum: 62
    
    a=1 (2):
      k=0 (3): b=0 (18), b=1 (12) -> 30
      k=1 (2): b=0 (12), b=1 (7) -> 19
      Sum: 49
    
    Total = 62 + 49 = 111. Still worse than 110.
    
    Let me stick with the 4+1 approach but see if we can improve the 4x4x4 decomposition.
    Currently it's rank 49. Can we do better?
    
    Actually, let me just implement the 3+2 partition to see if my calculation is wrong.
    """
    n, m, p = DIMENSIONS
    
    # Block sizes
    sizes = [3, 2]
    num_blocks = 2
    
    # Precompute offsets
    offsets = [0, 3]
    
    # Total rank
    total_rank = 0
    block_decomps = []
    
    for a in range(num_blocks):
        for k in range(num_blocks):
            for b in range(num_blocks):
                sa = sizes[a]
                sk = sizes[k]
                sb = sizes[b]
                
                if sa == 2 and sk == 2 and sb == 2:
                    decomp = _strassen_2x2()
                else:
                    decomp = _schoolbook_for(sa, sk, sb)
                
                block_decomps.append({
                    "U": decomp["U"],
                    "V": decomp["V"],
                    "W": decomp["W"],
                    "u_off_a": offsets[a],
                    "u_off_k": offsets[k],
                    "v_off_k": offsets[k],
                    "v_off_b": offsets[b],
                    "w_off_b": offsets[b],
                    "w_off_a": offsets[a],
                    "sa": sa,
                    "sk": sk,
                    "sb": sb
                })
                
                total_rank += len(decomp["U"][0]) if decomp["U"] else 0
    
    # Allocate final matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_col = 0
    for bd in block_decomps:
        sa, sk, sb = bd["sa"], bd["sk"], bd["sb"]
        U_block = bd["U"]
        V_block = bd["V"]
        W_block = bd["W"]
        rank_block = len(U_block[0])
        
        for local_i in range(sa):
            for local_j in range(sk):
                global_u_row = (bd["u_off_a"] + local_i) * m + (bd["u_off_k"] + local_j)
                for r in range(rank_block):
                    U[global_u_row][current_col + r] = U_block[local_i * sk + local_j][r]
        
        for local_j in range(sk):
            for local_k in range(sb):
                global_v_row = (bd["v_off_k"] + local_j) * p + (bd["v_off_b"] + local_k)
                for r in range(rank_block):
                    V[global_v_row][current_col + r] = V_block[local_j * sb + local_k][r]
        
        for local_k in range(sb):
            for local_i in range(sa):
                global_w_row = (bd["w_off_b"] + local_k) * n + (bd["w_off_a"] + local_i)
                for r in range(rank_block):
                    W[global_w_row][current_col + r] = W_block[local_k * sa + local_i][r]
        
        current_col += rank_block
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try 4x4+1x1 block decomposition first (expected rank 110)
    best_decomp = None
    best_rank = float('inf')
    
    try:
        if time.time() < deadline:
            block_decomp_4_1 = _block_decomposition_4_1()
            if _verify_fast(block_decomp_4_1, n, m, p):
                rank = len(block_decomp_4_1["U"][0])
                if rank < best_rank:
                    best_decomp = block_decomp_4_1
                    best_rank = rank
    except Exception as e:
        pass
    
    # Try 2x2+2x1+1x1 block decomposition (expected rank 117)
    try:
        if time.time() < deadline:
            block_decomp = _block_decomposition()
            if _verify_fast(block_decomp, n, m, p):
                rank = len(block_decomp["U"][0])
                if rank < best_rank:
                    best_decomp = block_decomp
                    best_rank = rank
    except Exception as e:
        pass
    
    # Try 3x3+2x2 block decomposition
    try:
        if time.time() < deadline:
            block_decomp_3_2 = _block_decomposition_3_2()
            if _verify_fast(block_decomp_3_2, n, m, p):
                rank = len(block_decomp_3_2["U"][0])
                if rank < best_rank:
                    best_decomp = block_decomp_3_2
                    best_rank = rank
    except Exception as e:
        pass
    
    if best_decomp is not None:
        return best_decomp
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
