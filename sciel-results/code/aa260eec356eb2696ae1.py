"""
Matrix multiplication 2x3 by 3x7 tensor decomposition.

Strategy: Hybrid Strassen/Schoolbook decomposition.
- Split the inner dimension (3) into 2+1.
- Use Strassen (rank 7) for each 2x2x2 block in the 2x2 x 2x7 part.
- Use schoolbook for the 2x2 x 2x1 part (rank 4).
- Use schoolbook for the 2x1 x 1x7 part (rank 14).
Total: 7*3 + 4 + 14 = 39.

This is a structural improvement over schoolbook (rank 42).
"""
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # Shapes:
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (14, rank)
    
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u_dict.items():
            u_list[idx] = val
        for idx, val in v_dict.items():
            v_list[idx] = val
        for idx, val in w_dict.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def u_idx(i, j):
        """Index into U for A[i,j] where i in [0,n), j in [0,m)."""
        return i * m + j
    
    def v_idx(j, k):
        """Index into V for B[j,k] where j in [0,m), k in [0,p)."""
        return j * p + k
    
    def w_idx(k, i):
        """Index into W for C[i,k] where i in [0,n), k in [0,p). Using trace convention W[k*n+i]."""
        return k * n + i
    
    # Part 1: A_left (2x2) @ B_top (2x7)
    # A_left = [[a00, a01], [a10, a11]] = A[0:2, 0:2]
    # B_top = B[0:2, 0:7]
    # Split B_top into three 2x2 blocks and one 2x1 block:
    #   B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:7]
    
    # For each 2x2 block, use Strassen (7 terms).
    # For the final 2x1 block, use schoolbook (4 terms).
    
    def strassen_block(col_start, col_end):
        """
        Add Strassen terms for A_left @ B_top[:, col_start:col_end]
        where col_end - col_start == 2.
        The result contributes to C[:, col_start:col_end].
        """
        c0, c1 = col_start, col_start + 1
        
        # M1 = (a00 + a11)(b0c0 + b1c1)
        # A part: a00, a11
        # B part: B[0,c0], B[1,c1]
        # C part: C[0,c0] (+1), C[1,c1] (+1)
        add_term(
            {u_idx(0,0): 1, u_idx(1,1): 1},
            {v_idx(0,c0): 1, v_idx(1,c1): 1},
            {w_idx(c0,0): 1, w_idx(c1,1): 1}
        )
        
        # M2 = (a10 + a11)b0c0
        # A part: a10, a11
        # B part: B[0,c0]
        # C part: C[1,c0] (+1), C[1,c1] (-1)
        add_term(
            {u_idx(1,0): 1, u_idx(1,1): 1},
            {v_idx(0,c0): 1},
            {w_idx(c0,1): 1, w_idx(c1,1): -1}
        )
        
        # M3 = a00(b0c1 - b1c1)
        # A part: a00
        # B part: B[0,c1], B[1,c1]
        # C part: C[0,c1] (+1), C[1,c1] (+1)
        add_term(
            {u_idx(0,0): 1},
            {v_idx(0,c1): 1, v_idx(1,c1): -1},
            {w_idx(c1,0): 1, w_idx(c1,1): 1}
        )
        
        # M4 = a11(b1c0 - b0c0)
        # A part: a11
        # B part: B[1,c0], B[0,c0]
        # C part: C[0,c0] (+1), C[1,c0] (+1)
        add_term(
            {u_idx(1,1): 1},
            {v_idx(1,c0): 1, v_idx(0,c0): -1},
            {w_idx(c0,0): 1, w_idx(c0,1): 1}
        )
        
        # M5 = (a00 + a01)b1c1
        # A part: a00, a01
        # B part: B[1,c1]
        # C part: C[0,c0] (-1), C[0,c1] (+1)
        add_term(
            {u_idx(0,0): 1, u_idx(0,1): 1},
            {v_idx(1,c1): 1},
            {w_idx(c0,0): -1, w_idx(c1,0): 1}
        )
        
        # M6 = (a10 - a00)(b0c0 + b0c1)
        # A part: a10, a00
        # B part: B[0,c0], B[0,c1]
        # C part: C[1,c1] (+1)
        add_term(
            {u_idx(1,0): 1, u_idx(0,0): -1},
            {v_idx(0,c0): 1, v_idx(0,c1): 1},
            {w_idx(c1,1): 1}
        )
        
        # M7 = (a01 - a11)(b1c0 + b1c1)
        # A part: a01, a11
        # B part: B[1,c0], B[1,c1]
        # C part: C[0,c0] (+1)
        add_term(
            {u_idx(0,1): 1, u_idx(1,1): -1},
            {v_idx(1,c0): 1, v_idx(1,c1): 1},
            {w_idx(c0,0): 1}
        )
    
    # Strassen blocks for columns 0-1, 2-3, 4-5
    for col_start in [0, 2, 4]:
        strassen_block(col_start, col_start + 2)
    
    # Schoolbook for the final 2x1 block: columns 6
    # A_left @ B_top[:, 6]
    # C[0,6] = a00*B[0,6] + a01*B[1,6]
    # C[1,6] = a10*B[0,6] + a11*B[1,6]
    
    c0 = 6
    # Term: a00 * B[0,c0] -> C[0,c0]
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,c0): 1},
        {w_idx(c0,0): 1}
    )
    
    # Term: a01 * B[1,c0] -> C[0,c0]
    add_term(
        {u_idx(0,1): 1},
        {v_idx(1,c0): 1},
        {w_idx(c0,0): 1}
    )
    
    # Term: a10 * B[0,c0] -> C[1,c0]
    add_term(
        {u_idx(1,0): 1},
        {v_idx(0,c0): 1},
        {w_idx(c0,1): 1}
    )
    
    # Term: a11 * B[1,c0] -> C[1,c0]
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,c0): 1},
        {w_idx(c0,1): 1}
    )
    
    # Part 2: A_right (2x1) @ B_bottom (1x7)
    # A_right = [a02, a12]^T = A[0:2, 2]
    # B_bottom = B[2, 0:7]
    # C[i,k] += a_i2 * B[2,k]
    
    for k in range(p):
        # Term: a02 * B[2,k] -> C[0,k]
        add_term(
            {u_idx(0,2): 1},
            {v_idx(2,k): 1},
            {w_idx(k,0): 1}
        )
        
        # Term: a12 * B[2,k] -> C[1,k]
        add_term(
            {u_idx(1,2): 1},
            {v_idx(2,k): 1},
            {w_idx(k,1): 1}
        )
    
    # Assemble U, V, W
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        U[:, c] = u_list
        V[:, c] = v_list
        W[:, c] = w_list
    
    return {"U": U, "V": V, "W": W}
