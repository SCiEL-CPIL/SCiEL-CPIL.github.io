import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition for 2x4x5
    # Strategy: Split m=4 into two m=2 blocks.
    # Each 2x2x5 block is split into 2x2x2 (Strassen) + 2x2x2 (Strassen) + 2x2x1 (Schoolbook)?
    # No, 2x2x5 = 2x2x4 + 2x2x1.
    # 2x2x4 = 2x2x2 (Strassen) + 2x2x2 (Schoolbook). Rank 7+4=11.
    # 2x2x1 = Schoolbook. Rank 4.
    # Total per 2x2x5 block: 15.
    # Total rank: 30.
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Block 1: A1 (rows 0,1 of A) * B1 (rows 0,1 of B)
    # A1 is 2x2. B1 is 2x5.
    # B1 cols 0,1 -> 2x2x2 Strassen
    # B1 cols 2,3 -> 2x2x2 Schoolbook
    # B1 col 4 -> 2x2x1 Schoolbook
    
    # Strassen for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    def strassen_222(a_idx, b_idx, c_idx, offset):
        # a_idx: indices for A (i, j)
        # b_idx: indices for B (j, k)
        # c_idx: indices for C (i, k)
        # offset: column offset
        # A indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # B indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # C indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        
        # m1
        add_term(a_idx[0], b_idx[0], c_idx[0], 1, 1, 1) # a*e
        add_term(a_idx[0], b_idx[1], c_idx[0], 1, 1, 1) # a*f
        add_term(a_idx[1], b_idx[0], c_idx[0], 1, 1, 1) # d*e
        add_term(a_idx[1], b_idx[1], c_idx[0], 1, 1, 1) # d*f
        # Wait, Strassen combines inputs.
        # The tensor decomposition requires linear combinations in U, V, W.
        # We cannot just add terms. We must define the U, V, W vectors for each rank-1 term.
        
        # This approach is flawed. I need to generate the U, V, W matrices directly.
        pass

    # Correct approach:
    # Define the 30 terms explicitly.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
