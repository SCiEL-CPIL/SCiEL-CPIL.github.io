"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Uses block-Strassen: split middle dim 4 into two 2s, and for each 2x2 x 2x5
    sub-problem, split the 5 columns as 2+2+1:
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x1: schoolbook (rank 4)
    Total per sub-problem: 18
    Total rank: 2 * 18 = 36
    """
    import time
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        """
        Add Strassen terms for multiplying two 2x2 matrices.
        a_offset: (i0, j0) - top-left corner of the 2x2 block of A
        b_offset: (j0, k0) - top-left corner of the 2x2 block of B  
        c_offset: (i0, k0) - top-left corner of the 2x2 block of C
        """
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            # m1 = (x11+x22)*(y11+y22) -> z11 += 1, z22 += 1
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2 = (x21+x22)*y11 -> z21 += 1, z22 -= 1
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3 = x11*(y12-y22) -> z12 += 1, z22 += 1
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4 = x22*(y21-y11) -> z11 += 1, z21 += 1
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5 = (x11+x12)*y22 -> z11 -= 1, z12 += 1
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6 = (x21-x11)*(y11+y12) -> z22 += 1
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            # m7 = (x12-x22)*(y21+y22) -> z11 += 1
            ({(i0, j0+1): 1, (i0+1, j0+1): -1}, 
             {(j0+1, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        """
        Add schoolbook terms for multiplying 2x2 * 2x1.
        """
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    # Build the full decomposition
    # A is 2x4: i in {0,1}, j in {0,1,2,3}
    # B is 4x5: j in {0,1,2,3}, k in {0,1,2,3,4}
    # C is 2x5: i in {0,1}, k in {0,1,2,3,4}
    
    # Split middle dim: j in {0,1} (top) and j in {2,3} (bottom)
    # For each A-block (2x2), multiply with B-block (2x5)
    # Split B's 5 columns as: cols 0,1 (2x2) + cols 2,3 (2x2) + col 4 (2x1)
    
    # Top A-block: A[0:2, 0:2], multiply with B[0:2, :]
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))    # A_top * B[0:2, 0:2]
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))    # A_top * B[0:2, 2:4]
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))  # A_top * B[0:2, 4:5]
    
    # Bottom A-block: A[0:2, 2:4], multiply with B[2:4, :]
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))    # A_bot * B[2:4, 0:2]
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))    # A_bot * B[2:4, 2:4]
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))  # A_bot * B[2:4, 4:5]
    
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
