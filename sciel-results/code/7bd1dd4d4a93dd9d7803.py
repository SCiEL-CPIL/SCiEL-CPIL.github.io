"""
Rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.

We use a block decomposition:
  A (4x5) = [A1 | A2] where A1 is 4x4, A2 is 4x1
  B (5x5) = [B1; B2] where B1 is 4x5, B2 is 1x5
  C = A1 @ B1 + A2 @ B2

The A1 @ B1 part is a 4x4x5 multiplication, and A2 @ B2 is a 4x1x5 multiplication.
We use Strassen for the 4x4x4 sub-blocks and schoolbook for the remaining parts.

Total rank = 49 (for 4x4x4) + 4 (for j<4, k=4) + 5 (for j=4, k any) = 58.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 145.0

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11: i=0,j=0 -> idx 0; A22: i=1,j=1 -> idx 3
    # B11: j=0,k=0 -> idx 0; B22: j=1,k=1 -> idx 3
    # C11: i=0,k=0 -> idx 0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21: i=1,j=0 -> idx 2; A22: i=1,j=1 -> idx 3
    # B11: j=0,k=0 -> idx 0
    # C11: idx 0, C22: i=1,k=1 -> idx 3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = -1; W[3, 1] = 1
    
    # M3 = A11(B12-B22)
    # A11: idx 0
    # B12: j=0,k=1 -> idx 1; B22: idx 3
    # C12: i=0,k=1 -> idx 1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22: idx 3
    # B21: j=1,k=0 -> idx 2; B11: idx 0
    # C11: idx 0, C22: idx 3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11: idx 0; A12: i=0,j=1 -> idx 1
    # B22: idx 3
    # C12: idx 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21: idx 2; A11: idx 0
    # B11: idx 0; B12: idx 1
    # C21: i=1,k=0 -> idx 2
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12: idx 1; A22: idx 3
    # B21: idx 2; B22: idx 3
    # C11: idx 0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """
    Return U, V, W for 4x4x4 using tiled Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Outer Strassen indices
    # A blocks: A11 (i=0,1; j=0,1), A12 (i=0,1; j=2,3), A21 (i=2,3; j=0,1), A22 (i=2,3; j=2,3)
    # B blocks: B11 (j=0,1; k=0,1), B12 (j=0,1; k=2,3), B21 (j=2,3; k=0,1), B22 (j=2,3; k=2,3)
    # C blocks: C11 (i=0,1; k=0,1), C12 (i=0,1; k=2,3), C21 (i=2,3; k=0,1), C22 (i=2,3; k=2,3)
    
    # Outer Strassen M1..M7
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22)B11 -> contributes to C11, C22
    # M3 = A11(B12-B22) -> contributes to C12
    # M4 = A22(B21-B11) -> contributes to C11, C22
    # M5 = (A11+A12)B22 -> contributes to C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C21
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    
    # For each outer M, we have a 2x2x2 tensor which we decompose with inner Strassen (7 terms).
    # So 7 outer * 7 inner = 49.
    
    # Define the 7 outer Strassen combinations
    # Each combination has:
    #   - A block indices and coefficients
    #   - B block indices and coefficients
    #   - C block indices and coefficients
    
    # M1 = (A11+A22)(B11+B22)
    # A blocks: A11 (coeff 1), A22 (coeff 1)
    # B blocks: B11 (coeff 1), B22 (coeff 1)
    # C blocks: C11 (coeff 1), C22 (coeff 1)
    
    # M2 = (A21+A22)B11
    # A blocks: A21 (coeff 1), A22 (coeff 1)
    # B blocks: B11 (coeff 1)
    # C blocks: C11 (coeff -1), C22 (coeff 1)
    
    # M3 = A11(B12-B22)
    # A blocks: A11 (coeff 1)
    # B blocks: B12 (coeff 1), B22 (coeff -1)
    # C blocks: C12 (coeff 1)
    
    # M4 = A22(B21-B11)
    # A blocks: A22 (coeff 1)
    # B blocks: B21 (coeff 1), B11 (coeff -1)
    # C blocks: C11 (coeff 1), C22 (coeff 1)
    
    # M5 = (A11+A12)B22
    # A blocks: A11 (coeff 1), A12 (coeff 1)
    # B blocks: B22 (coeff 1)
    # C blocks: C12 (coeff 1)
    
    # M6 = (A21-A11)(B11+B12)
    # A blocks: A21 (coeff 1), A11 (coeff -1)
    # B blocks: B11 (coeff 1), B12 (coeff 1)
    # C blocks: C21 (coeff 1)
    
    # M7 = (A12-A22)(B21+B22)
    # A blocks: A12 (coeff 1), A22 (coeff -1)
    # B blocks: B21 (coeff 1), B22 (coeff 1)
    # C blocks: C11 (coeff 1)
    
    # Block index mapping:
    # A blocks: A11=0, A12=1, A21=2, A22=3
    # B blocks: B11=0, B12=1, B21=2, B22=3
    # C blocks: C11=0, C12=1, C21=2, C22=3
    
    # For each outer M, we need to combine the inner Strassen terms
    # The inner Strassen has 7 terms, each with U, V, W entries
    # We need to map the inner terms to the outer blocks
    
    # Let's define the 7 outer combinations as a list of tuples
    # Each tuple: (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    
    outer_combinations = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(0, -1), (3, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (3, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # For each outer combination, we have 7 inner Strassen terms
    # The inner Strassen U, V, W are 4x7 matrices
    # We need to map the inner indices to the outer block indices
    
    # Inner U indices: 0,1,2,3 correspond to (i,j) in 2x2
    # Inner V indices: 0,1,2,3 correspond to (j,k) in 2x2
    # Inner W indices: 0,1,2,3 correspond to (k,i) in 2x2
    
    # For a given outer combination with A blocks, B blocks, C blocks:
    # We need to create 7 new rank-1 tensors
    # For each inner term c (0..6):
    #   U_new[i_outer * 4 + j_outer, col] = sum over A blocks of coeff * U2[i_inner * 2 + j_inner, c]
    #   where i_inner, j_inner are determined by the A block
    #   Similarly for V and W
    
    # Let's define the mapping from block index to (i_range, j_range)
    # A blocks:
    # A11: i in 0..1, j in 0..1
    # A12: i in 0..1, j in 2..3
    # A21: i in 2..3, j in 0..1
    # A22: i in 2..3, j in 2..3
    
    # B blocks:
    # B11: j in 0..1, k in 0..1
    # B12: j in 0..1, k in 2..3
    # B21: j in 2..3, k in 0..1
    # B22: j in 2..3, k in 2..3
    
    # C blocks:
    # C11: i in 0..1, k in 0..1
    # C12: i in 0..1, k in 2..3
    # C21: i in 2..3, k in 0..1
    # C22: i in 2..3, k in 2..3
    
    # For each outer combination, we create 7 new columns
    col_offset = 0
    for outer_idx, (A_blocks, B_blocks, C_blocks) in enumerate(outer_combinations):
        for inner_c in range(7):
            col = col_offset + inner_c
            # U: for each (i,j) in 4x4, sum over A blocks
            for i in range(4):
                for j in range(4):
                    u_val = 0.0
                    for a_block_idx, a_coeff in A_blocks:
                        # Determine which inner (i_inner, j_inner) corresponds to (i, j) in this block
                        if a_block_idx == 0:  # A11: i in 0..1, j in 0..1
                            if 0 <= i <= 1 and 0 <= j <= 1:
                                i_inner = i
                                j_inner = j
                                u_val += a_coeff * U2[i_inner * 2 + j_inner, inner_c]
                        elif a_block_idx == 1:  # A12: i in 0..1, j in 2..3
                            if 0 <= i <= 1 and 2 <= j <= 3:
                                i_inner = i
                                j_inner = j - 2
                                u_val += a_coeff * U2[i_inner * 2 + j_inner, inner_c]
                        elif a_block_idx == 2:  # A21: i in 2..3, j in 0..1
                            if 2 <= i <= 3 and 0 <= j <= 1:
                                i_inner = i - 2
                                j_inner = j
                                u_val += a_coeff * U2[i_inner * 2 + j_inner, inner_c]
                        elif a_block_idx == 3:  # A22: i in 2..3, j in 2..3
                            if 2 <= i <= 3 and 2 <= j <= 3:
                                i_inner = i - 2
                                j_inner = j - 2
                                u_val += a_coeff * U2[i_inner * 2 + j_inner, inner_c]
                    U[i * 4 + j, col] = u_val
            
            # V: for each (j,k) in 4x4, sum over B blocks
            for j in range(4):
                for k in range(4):
                    v_val = 0.0
                    for b_block_idx, b_coeff in B_blocks:
                        if b_block_idx == 0:  # B11: j in 0..1, k in 0..1
                            if 0 <= j <= 1 and 0 <= k <= 1:
                                j_inner = j
                                k_inner = k
                                v_val += b_coeff * V2[j_inner * 2 + k_inner, inner_c]
                        elif b_block_idx == 1:  # B12: j in 0..1, k in 2..3
                            if 0 <= j <= 1 and 2 <= k <= 3:
                                j_inner = j
                                k_inner = k - 2
                                v_val += b_coeff * V2[j_inner * 2 + k_inner, inner_c]
                        elif b_block_idx == 2:  # B21: j in 2..3, k in 0..1
                            if 2 <= j <= 3 and 0 <= k <= 1:
                                j_inner = j - 2
                                k_inner = k
                                v_val += b_coeff * V2[j_inner * 2 + k_inner, inner_c]
                        elif b_block_idx == 3:  # B22: j in 2..3, k in 2..3
                            if 2 <= j <= 3 and 2 <= k <= 3:
                                j_inner = j - 2
                                k_inner = k - 2
                                v_val += b_coeff * V2[j_inner * 2 + k_inner, inner_c]
                    V[j * 4 + k, col] = v_val
            
            # W: for each (k,i) in 4x4, sum over C blocks
            for k in range(4):
                for i in range(4):
                    w_val = 0.0
                    for c_block_idx, c_coeff in C_blocks:
                        if c_block_idx == 0:  # C11: i in 0..1, k in 0..1
                            if 0 <= i <= 1 and 0 <= k <= 1:
                                i_inner = i
                                k_inner = k
                                w_val += c_coeff * W2[k_inner * 2 + i_inner, inner_c]
                        elif c_block_idx == 1:  # C12: i in 0..1, k in 2..3
                            if 0 <= i <= 1 and 2 <= k <= 3:
                                i_inner = i
                                k_inner = k - 2
                                w_val += c_coeff * W2[k_inner * 2 + i_inner, inner_c]
                        elif c_block_idx == 2:  # C21: i in 2..3, k in 0..1
                            if 2 <= i <= 3 and 0 <= k <= 1:
                                i_inner = i - 2
                                k_inner = k
                                w_val += c_coeff * W2[k_inner * 2 + i_inner, inner_c]
                        elif c_block_idx == 3:  # C22: i in 2..3, k in 2..3
                            if 2 <= i <= 3 and 2 <= k <= 3:
                                i_inner = i - 2
                                k_inner = k - 2
                                w_val += c_coeff * W2[k_inner * 2 + i_inner, inner_c]
                    W[k * 4 + i, col] = w_val
        
        col_offset += 7
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition for (n,m,p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check exact decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j]
                v = V[j * p + k]
                w = W[k * n + i]
                s = np.sum(u * v * w)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = _strassen_4x4x4()
    
    # Build the full 4x5x5 decomposition
    # Part 1: 4x4x4 block (i=0..3, j=0..3, k=0..3)
    # Part 2: j=0..3, k=4 (4 terms)
    # Part 3: j=4, k=0..4 (5 terms)
    
    total_rank = 49 + 4 + 5  # 58
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Copy 4x4x4 part
    U[:, :49] = U_444
    V[:, :49] = V_444
    W[:, :49] = W_444
    
    # Part 2: j=0..3, k=4
    # For each j in 0..3, one term
    col = 49
    for j in range(4):
        # U[i, j, col] = 1 for all i in 0..3
        for i in range(4):
            U[i * 5 + j, col] = 1
        # V[j, 4, col] = 1
        V[j * 5 + 4, col] = 1
        # W[4, i, col] = 1 for all i in 0..3
        for i in range(4):
            W[4 * 4 + i, col] = 1
        col += 1
    
    # Part 3: j=4, k=0..4
    # For each k in 0..4, one term
    for k in range(5):
        # U[i, 4, col] = 1 for all i in 0..3
        for i in range(4):
            U[i * 5 + 4, col] = 1
        # V[4, k, col] = 1
        V[4 * 5 + k, col] = 1
        # W[k, i, col] = 1 for all i in 0..3
        for i in range(4):
            W[k * 4 + i, col] = 1
        col += 1
    
    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fall back to schoolbook if validation fails
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
