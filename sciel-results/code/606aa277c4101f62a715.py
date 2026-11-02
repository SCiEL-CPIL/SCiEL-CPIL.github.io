import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 51
    
    # We will build the matrices U, V, W incrementally.
    # U: (n*m, rank) = (6, 51)
    # V: (m*p, rank) = (27, 51)
    # W: (p*n, rank) = (18, 51)
    
    # We'll use a list of tuples (u_vec, v_vec, w_vec) for each term
    # u_vec is length n*m
    # v_vec is length m*p
    # w_vec is length p*n
    
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # Define indices
    # A: i in 0..1, j in 0..2. idx = i*m + j = i*3 + j
    # B: j in 0..2, k in 0..8. idx = j*p + k = j*9 + k
    # C: i in 0..1, k in 0..8. idx = k*n + i = k*2 + i (Trace convention)
    
    # Block 0: k in 0..2
    # Block 1: k in 3..5
    # Block 2: k in 6..8
    
    for block_idx in range(3):
        k_offset = block_idx * 3
        
        # Part 1: 2x3 by 3x2 (Left part of 3x3 block)
        # B columns k_offset, k_offset+1
        # A full 2x3
        # Result C columns k_offset, k_offset+1
        
        # Sub-part 1a: 2x2 by 2x2 (Strassen)
        # A_left: cols 0,1 of A.
        # B_top: rows 0,1 of B_block.
        
        # Strassen terms for 2x2 by 2x2
        # Inputs:
        # a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
        # b11=B[0,k_off], b12=B[0,k_off+1]
        # b21=B[1,k_off], b22=B[1,k_off+1]
        
        # M1 = (a11+a22)(b11+b22)
        u1 = [0]*6; v1 = [0]*27; w1 = [0]*18
        u1[0] = 1; u1[3] = 1 # a11, a22 (idx 0, 3)
        v1[0*9+k_offset] = 1; v1[1*9+k_offset+1] = 1 # b11, b22
        # Output C[0,k_off] and C[0,k_off+1] and C[1,k_off] and C[1,k_off+1]
        # C11 = M1 + ...
        # We need to add to W.
        # W[k*n+i].
        # C[0,k_off] -> W[2*k_off + 0]
        # C[0,k_off+1] -> W[2*(k_off+1) + 0]
        # C[1,k_off] -> W[2*k_off + 1]
        # C[1,k_off+1] -> W[2*(k_off+1) + 1]
        
        # This manual construction is error-prone.
        # Better to define the sub-decompositions as functions returning lists of (u,v,w) relative to local indices, then map them.
        
    pass
