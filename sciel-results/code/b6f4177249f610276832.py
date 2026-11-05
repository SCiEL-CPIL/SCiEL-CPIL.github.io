import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank 189 decomposition: Strassen (7) on 2x2 blocks of 3x3, Schoolbook (27) on 3x3 blocks.
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    # Block size is 3. Number of blocks is 2.
    block_size = 3
    num_blocks = n // block_size # 2
    
    # Strassen terms for 2x2 block multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_t is a product of linear combinations of A-blocks and B-blocks.
    # We represent each M_t as a list of (coeff_A, A_block_idx, coeff_B, B_block_idx) tuples?
    # Actually, M_t = (sum c_a A_a) * (sum d_b B_b) = sum_{a,b} c_a d_b A_a B_b.
    # But we can keep the linearity in U and V.
    # For each local index (i, j, k) in 0..2:
    # We add one rank-1 term.
    # U has 1 at (A_block_row, i, A_block_col, j) for each A_block in the sum for M_t with coeff 1.
    # V has 1 at (B_block_row, j, B_block_col, k) for each B_block in the sum for M_t with coeff 1.
    # W has coeff at (C_block_row, k, C_block_col, i) for each C_block receiving M_t.
    
    # Let's define the structure of the 7 terms.
    # A_blocks: (row, col) in 0..1
    # B_blocks: (row, col) in 0..1
    
    # Term 1: (A00 + A11) * (B00 + B11)
    # A parts: [(0,0, 1), (1,1, 1)]
    # B parts: [(0,0, 1), (1,1, 1)]
    # C parts: C00(+1), C11(+1) -- wait, C00 = M1 + ..., C11 = M1 + ...
    # So M1 contributes +1 to C00 and +1 to C11.
    
    # Term 2: (A10 + A11) * B00
    # A parts: [(1,0, 1), (1,1, 1)]
    # B parts: [(0,0, 1)]
    # C parts: C10(+1) -- C10 = M2 + ...
    
    # Term 3: A00 * (B01 - B11)
    # A parts: [(0,0, 1)]
    # B parts: [(0,1, 1), (1,1, -1)]
    # C parts: C01(+1) -- C01 = M3 + ...
    
    # Term 4: A11 * (B10 - B00)
    # A parts: [(1,1, 1)]
    # B parts: [(1,0, 1), (0,0, -1)]
    # C parts: C00(+1), C10(+1) -- C00 = ... + M4 ..., C10 = ... + M4 ...
    
    # Term 5: (A00 + A01) * B11
    # A parts: [(0,0, 1), (0,1, 1)]
    # B parts: [(1,1, 1)]
    # C parts: C00(-1), C01(+1) -- C00 = ... - M5 ..., C01 = ... + M5 ...
    
    # Term 6: (A10 - A00) * (B00 + B01)
    # A parts: [(1,0, 1), (0,0, -1)]
    # B parts: [(0,0, 1), (0,1, 1)]
    # C parts: C11(+1) -- C11 = ... + M6 ...
    
    # Term 7: (A01 - A11) * (B10 + B11)
    # A parts: [(0,1, 1), (1,1, -1)]
    # B parts: [(1,0, 1), (1,1, 1)]
    # C parts: C00(+1) -- C00 = ... + M7 ...
    
    terms_def = [
        {
            "A": [(0,0, 1), (1,1, 1)],
            "B": [(0,0, 1), (1,1, 1)],
            "C": [(0,0, 1), (1,1, 1)]
        },
        {
            "A": [(1,0, 1), (1,1, 1)],
            "B": [(0,0, 1)],
            "C": [(1,0, 1)]
        },
        {
            "A": [(0,0, 1)],
            "B": [(0,1, 1), (1,1, -1)],
            "C": [(0,1, 1)]
        },
        {
            "A": [(1,1, 1)],
            "B": [(1,0, 1), (0,0, -1)],
            "C": [(0,0, 1), (1,0, 1)]
        },
        {
            "A": [(0,0, 1), (0,1, 1)],
            "B": [(1,1, 1)],
            "C": [(0,0, -1), (0,1, 1)]
        },
        {
            "A": [(1,0, 1), (0,0, -1)],
            "B": [(0,0, 1), (0,1, 1)],
            "C": [(1,1, 1)]
        },
        {
            "A": [(0,1, 1), (1,1, -1)],
            "B": [(1,0, 1), (1,1, 1)],
            "C": [(0,0, 1)]
        }
    ]
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Preallocate if needed, or just append
    # Total rank = 7 * 3 * 3 * 3 = 189
    
    for term in terms_def:
        A_parts = term["A"]
        B_parts = term["B"]
        C_parts = term["C"]
        
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    # Create U vector
                    U = [0] * (n * m)
                    for a_row, a_col, coeff in A_parts:
                        # Global index for A
                        # A is n x m.
                        # Block (a_row, a_col) corresponds to rows a_row*3 .. a_row*3+2, cols a_col*3 .. a_col*3+2
                        # Local index (i_l, j_l)
                        global_i = a_row * block_size + i_l
                        global_j = a_col * block_size + j_l
                        idx = global_i * m + global_j
                        U[idx] += coeff
                    
                    # Create V vector
                    V = [0] * (m * p)
                    for b_row, b_col, coeff in B_parts:
                        # B is m x p.
                        # Block (b_row, b_col)
                        global_j = b_row * block_size + j_l
                        global_k = b_col * block_size + k_l
                        idx = global_j * p + global_k
                        V[idx] += coeff
                    
                    # Create W vector
                    W = [0] * (p * n)
                    for c_row, c_col, coeff in C_parts:
                        # C is p x n? No, C is n x p?
                        # Task says: W[k*n+i, r].
                        # Standard matrix mult: C = A B. A is n x m, B is m x p. C is n x p.
                        # Index in W: k*n + i. Here k is row of C? i is col of C?
                        # Let's check the seed code.
                        # Seed: W[k*n+i, c].
                        # Loop: for i in range(n): for j in range(m): for k in range(p):
                        # U[i*m+j] = 1. (Row i, Col j of A)
                        # V[j*p+k] = 1. (Row j, Col k of B)
                        # W[k*n+i] = 1. (Row k, Col i of C? No, C is n x p. Row i, Col k?)
                        # If C is n x p, index is i*p + k.
                        # But seed uses k*n + i.
                        # This implies the output matrix is transposed or indexed differently.
                        # "The output factor follows the official trace convention W[k*n+i,r]".
                        # This usually means the tensor is $T_{i,j,k} = A_{ij} B_{jk} C_{ki}$?
                        # Or $C_{ik} = \sum_j A_{ij} B_{jk}$.
                        # If W index is $k*n + i$, and $k \in [0,p), i \in [0,n)$, then the matrix is $p \times n$.
                        # But $C$ is $n \times p$.
                        # Let's assume the evaluator expects the matrix to be flattened in row-major order of size $p \times n$?
                        # Or maybe $i$ is row, $k$ is col, but flattened as $k*n + i$ (col-major)?
                        # Let's stick to the seed's indexing: $idx = k * n + i$.
                        
                        global_k = c_row * block_size + k_l
                        global_i = c_col * block_size + i_l
                        idx = global_k * n + global_i
                        W[idx] += coeff
                    
                    U_list.append(U)
                    V_list.append(V)
                    W_list.append(W)
                    rank += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
