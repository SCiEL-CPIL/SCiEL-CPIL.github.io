"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy: Hybrid Strassen/Schoolbook.
    - Split the 3x7 product into a 2x7 and a 1x7 part.
    - For the 2x2 sub-block (first 2 cols of B, first 2 rows of A), use Strassen (rank 7 instead of 8).
    - For the remaining parts, use schoolbook.
    
    Dimensions: A is 2x3, B is 3x7.
    Let A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Let B = [B1; B2] where B1 is 2x7, B2 is 1x7.
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x7):
      B1 = [B1a | B1b] where B1a is 2x3, B1b is 2x4.
      A1*B1a (2x2 * 2x3): Use Strassen for the 2x2 part, schoolbook for the 1x3.
        Actually, let's be more systematic.
      
    Better approach:
    A is 2x3. Split A into A_left (2x2) and A_right (2x1).
    B is 3x7. Split B into B_top (2x7) and B_bottom (1x7).
    
    C = A_left @ B_top + A_right @ B_bottom
    
    For A_left @ B_top (2x2 * 2x7):
      We can use Strassen for each pair of columns? No, Strassen works for square blocks.
      Instead, view B_top as 7 separate 2x1 vectors.
      For each column k of B_top, compute A_left @ B_top[:,k] (2x2 * 2x1).
      This is like multiplying a 2x2 by a 2x1. Rank is 4 (schoolbook) or can we do better?
      Actually, 2x2 * 2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1]
                                  C[1] = A[1,0]*B[0] + A[1,1]*B[1]
      This is 4 products. Can we do it in 3? 
      Let's see: This is equivalent to computing 2 independent linear combinations.
      The tensor is of size 4x2x2. The rank of this tensor... 
      Actually, this is just 2 separate dot products of length 2, so rank 4 is likely optimal.
      
    Alternative: Use Strassen for the 2x2 * 2x2 block, and handle the rest.
    Let B_top = [B1 | B2] where B1 is 2x3, B2 is 2x4.
    A_left @ B1 (2x2 * 2x3):
      B1 = [b1 | b2 | b3] (three 2x1 columns)
      A_left @ b1: 4 products (schoolbook)
      A_left @ b2: 4 products
      A_left @ b3: 4 products
      Total: 12 products.
      
    A_left @ B2 (2x2 * 2x4):
      B2 = [b4 | b5 | b6 | b7]
      4 * 4 = 16 products.
      
    A_right @ B_bottom (2x1 * 1x7):
      2 * 7 = 14 products.
      
    Total schoolbook: 12 + 16 + 14 = 42.
    
    Now, can we reduce?
    For the 2x2 * 2x3 part, we have 3 independent 2x2*2x1 multiplications.
    Each 2x2*2x1 has rank 4. Total 12.
    
    What if we group two columns? 2x2 * 2x2 block.
    A_left @ B1[:,0:2] (2x2 * 2x2): Use Strassen (rank 7).
    A_left @ B1[:,2] (2x2 * 2x1): Schoolbook (rank 4).
    Total for 2x2*2x3: 7 + 4 = 11.
    
    Similarly, A_left @ B2 (2x2 * 2x4):
    Split B2 into two 2x2 blocks.
    A_left @ B2[:,0:2] (2x2 * 2x2): Strassen (rank 7).
    A_left @ B2[:,2:4] (2x2 * 2x2): Strassen (rank 7).
    Total for 2x2*2x4: 7 + 7 = 14.
    
    A_right @ B_bottom (2x1 * 1x7): Schoolbook (rank 14).
    
    Total: 11 + 14 + 14 = 39.
    
    This gives rank 39, which is better than 42.
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (21, rank)
    # W has shape (p*n, rank) = (14, rank)
    
    # Let's collect the decomposition terms.
    # Each term is (u_vec, v_vec, w_vec) where u_vec has length 6, v_vec has length 21, w_vec has length 14.
    
    terms = []
    
    def add_term(u, v, w):
        """Add a term to the decomposition."""
        # u, v, w are dicts mapping index to value
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    # Helper to create index for U: row i, col j -> i*m + j
    def u_idx(i, j):
        return i * m + j
    
    # Helper to create index for V: row j, col k -> j*p + k
    def v_idx(j, k):
        return j * p + k
    
    # Helper to create index for W: row k, col i -> k*n + i
    def w_idx(k, i):
        return k * n + i
    
    # Part 1: A_left @ B1[:,0:2] (2x2 * 2x2) using Strassen
    # A_left = [[a00, a01], [a10, a11]]
    # B1_block = [[b00, b01], [b10, b11]] where b00=B[0,0], b01=B[0,1], b10=B[1,0], b11=B[1,1]
    # Strassen formulas:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we have a product of a linear combination of A entries 
    # and a linear combination of B entries, contributing to specific C entries.
    
    # Strassen term 1: (a00 + a11)(b00 + b11)
    # U part: a00 -> u_idx(0,0)=0, a11 -> u_idx(1,1)=4
    # V part: b00 -> v_idx(0,0)=0, b11 -> v_idx(1,1)=8
    # W part: contributes to C00 (k=0,i=0) -> w_idx(0,0)=0, C01 (k=1,i=0) -> w_idx(1,0)=1,
    #          C10 (k=0,i=1) -> w_idx(0,1)=2, C11 (k=1,i=1) -> w_idx(1,1)=3
    # Coefficients: C00: +1, C01: 0, C10: 0, C11: +1 (from M1 in C00 and C11)
    # Wait, let me re-derive.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (a00 + a11)(b00 + b11)
    # Contribution to C00: +1, to C11: +1
    add_term(
        {u_idx(0,0): 1, u_idx(1,1): 1},
        {v_idx(0,0): 1, v_idx(1,1): 1},
        {w_idx(0,0): 1, w_idx(1,1): 1}
    )
    
    # M2 = (a10 + a11)b00
    # Contribution to C10: +1, to C11: -1
    add_term(
        {u_idx(1,0): 1, u_idx(1,1): 1},
        {v_idx(0,0): 1},
        {w_idx(0,1): 1, w_idx(1,1): -1}
    )
    
    # M3 = a00(b01 - b11)
    # Contribution to C01: +1, to C11: +1
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,1): 1, v_idx(1,1): -1},
        {w_idx(1,0): 1, w_idx(1,1): 1}
    )
    
    # M4 = a11(b10 - b00)
    # Contribution to C00: +1, to C10: +1
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,0): 1, v_idx(0,0): -1},
        {w_idx(0,0): 1, w_idx(0,1): 1}
    )
    
    # M5 = (a00 + a01)b11
    # Contribution to C00: -1, to C01: +1
    add_term(
        {u_idx(0,0): 1, u_idx(0,1): 1},
        {v_idx(1,1): 1},
        {w_idx(0,0): -1, w_idx(1,0): 1}
    )
    
    # M6 = (a10 - a00)(b00 + b01)
    # Contribution to C11: +1
    add_term(
        {u_idx(1,0): 1, u_idx(0,0): -1},
        {v_idx(0,0): 1, v_idx(0,1): 1},
        {w_idx(1,1): 1}
    )
    
    # M7 = (a01 - a11)(b10 + b11)
    # Contribution to C00: +1
    add_term(
        {u_idx(0,1): 1, u_idx(1,1): -1},
        {v_idx(1,0): 1, v_idx(1,1): 1},
        {w_idx(0,0): 1}
    )
    
    # Part 2: A_left @ B1[:,2] (2x2 * 2x1) using schoolbook
    # C0 = a00*b02 + a01*b12
    # C1 = a10*b02 + a11*b12
    # where b02=B[0,2], b12=B[1,2]
    
    # Term: a00*b02 -> C0
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,2): 1},
        {w_idx(2,0): 1}
    )
    
    # Term: a01*b12 -> C0
    add_term(
        {u_idx(0,1): 1},
        {v_idx(1,2): 1},
        {w_idx(2,0): 1}
    )
    
    # Term: a10*b02 -> C1
    add_term(
        {u_idx(1,0): 1},
        {v_idx(0,2): 1},
        {w_idx(2,1): 1}
    )
    
    # Term: a11*b12 -> C1
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,2): 1},
        {w_idx(2,1): 1}
    )
    
    # Part 3: A_left @ B2[:,0:2] (2x2 * 2x2) using Strassen
    # B2[:,0:2] = B[0:2, 3:5] = [[B[0,3], B[0,4]], [B[1,3], B[1,4]]]
    # Same Strassen structure as Part 1, but with different column indices.
    
    # M1 = (a00 + a11)(b03 + b14)
    add_term(
        {u_idx(0,0): 1, u_idx(1,1): 1},
        {v_idx(0,3): 1, v_idx(1,4): 1},
        {w_idx(3,0): 1, w_idx(4,1): 1}
    )
    
    # M2 = (a10 + a11)b03
    add_term(
        {u_idx(1,0): 1, u_idx(1,1): 1},
        {v_idx(0,3): 1},
        {w_idx(3,1): 1, w_idx(4,1): -1}
    )
    
    # M3 = a00(b04 - b14)
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,4): 1, v_idx(1,4): -1},
        {w_idx(4,0): 1, w_idx(4,1): 1}
    )
    
    # M4 = a11(b13 - b03)
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,3): 1, v_idx(0,3): -1},
        {w_idx(3,0): 1, w_idx(3,1): 1}
    )
    
    # M5 = (a00 + a01)b14
    add_term(
        {u_idx(0,0): 1, u_idx(0,1): 1},
        {v_idx(1,4): 1},
        {w_idx(3,0): -1, w_idx(4,0): 1}
    )
    
    # M6 = (a10 - a00)(b03 + b04)
    add_term(
        {u_idx(1,0): 1, u_idx(0,0): -1},
        {v_idx(0,3): 1, v_idx(0,4): 1},
        {w_idx(4,1): 1}
    )
    
    # M7 = (a01 - a11)(b13 + b14)
    add_term(
        {u_idx(0,1): 1, u_idx(1,1): -1},
        {v_idx(1,3): 1, v_idx(1,4): 1},
        {w_idx(3,0): 1}
    )
    
    # Part 4: A_left @ B2[:,2:4] (2x2 * 2x2) using Strassen
    # B2[:,2:4] = B[0:2, 5:7] = [[B[0,5], B[0,6]], [B[1,5], B[1,6]]]
    
    # M1 = (a00 + a11)(b05 + b16)
    add_term(
        {u_idx(0,0): 1, u_idx(1,1): 1},
        {v_idx(0,5): 1, v_idx(1,6): 1},
        {w_idx(5,0): 1, w_idx(6,1): 1}
    )
    
    # M2 = (a10 + a11)b05
    add_term(
        {u_idx(1,0): 1, u_idx(1,1): 1},
        {v_idx(0,5): 1},
        {w_idx(5,1): 1, w_idx(6,1): -1}
    )
    
    # M3 = a00(b06 - b16)
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,6): 1, v_idx(1,6): -1},
        {w_idx(6,0): 1, w_idx(6,1): 1}
    )
    
    # M4 = a11(b15 - b05)
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,5): 1, v_idx(0,5): -1},
        {w_idx(5,0): 1, w_idx(5,1): 1}
    )
    
    # M5 = (a00 + a01)b16
    add_term(
        {u_idx(0,0): 1, u_idx(0,1): 1},
        {v_idx(1,6): 1},
        {w_idx(5,0): -1, w_idx(6,0): 1}
    )
    
    # M6 = (a10 - a00)(b05 + b06)
    add_term(
        {u_idx(1,0): 1, u_idx(0,0): -1},
        {v_idx(0,5): 1, v_idx(0,6): 1},
        {w_idx(6,1): 1}
    )
    
    # M7 = (a01 - a11)(b15 + b16)
    add_term(
        {u_idx(0,1): 1, u_idx(1,1): -1},
        {v_idx(1,5): 1, v_idx(1,6): 1},
        {w_idx(5,0): 1}
    )
    
    # Part 5: A_right @ B_bottom (2x1 * 1x7) using schoolbook
    # A_right = [a02, a12] (column vector)
    # B_bottom = [b20, b21, ..., b26] (row vector)
    # C0 = a02 * b20 + a02 * b21 + ... + a02 * b26 (but each term is separate in tensor)
    # Actually, C0[i] = a_i2 * b_2k for each k
    # So for each k in 0..6:
    #   Term: a02 * b2k -> C[0,k]
    #   Term: a12 * b2k -> C[1,k]
    
    for k in range(p):
        # a02 * b2k -> C[0,k]
        add_term(
            {u_idx(0,2): 1},
            {v_idx(2,k): 1},
            {w_idx(k,0): 1}
        )
        
        # a12 * b2k -> C[1,k]
        add_term(
            {u_idx(1,2): 1},
            {v_idx(2,k): 1},
            {w_idx(k,1): 1}
        )
    
    # Now assemble U, V, W
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u_list[i]
        for i in range(m * p):
            V[i][c] = v_list[i]
        for i in range(p * n):
            W[i][c] = w_list[i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
