DIMENSIONS = [2, 2, 14]

def construct():
    """
    Constructs a rank-49 exact tensor decomposition for 2x2 by 2x14 matrix multiplication.
    
    The multiplication is viewed as a tensor network where the 2x14 matrix B is decomposed 
    into 7 blocks of 2x2 matrices. The 2x2 matrix A is shared across all blocks.
    Since the blocks of B and the corresponding blocks of the output C are independent 
    (disjoint supports in the tensor factors V and W), the total tensor is the direct sum 
    of 7 instances of the 2x2x2 matrix multiplication tensor.
    
    The minimal rank of the 2x2x2 tensor is 7 (Strassen's algorithm).
    Therefore, the minimal rank for the 2x2x2x14 tensor is 7 * 7 = 49.
    
    This implementation uses the block-wise (MPS-like) structure to generate the factors.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Initialize factor matrices
    # U: (n*m, rank) -> (4, 49)
    # V: (m*p, rank) -> (28, 49)
    # W: (p*n, rank) -> (28, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for the 2x2 tensor
    # Indices for U (A): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for V (B_block): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
    # Indices for W (C_block): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
    
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of 4 floats
    # v_coeffs: list of 4 floats
    # w_coeffs: list of 4 floats
    
    strassen_terms = [
        # M1: (A00+A11) * (B00+B11) -> C00+C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11) * B00 -> C10+C11? No, standard Strassen:
        # M2 = (A10+A11) * B00. 
        # C10 += M2, C11 -= M2? 
        # Let's stick to the specific forms used in the current/inspiration code which are verified.
        # From Inspiration 1:
        # s=0: M1 = (A00+A11)(B00+B11) -> C00+M1, C11+M1
        # s=1: M2 = (A10+A11)B00 -> C10+M2, C11-M2
        # s=2: M3 = A00(B01-B11) -> C01+M3, C11+M3
        # s=3: M4 = A11(B10-B00) -> C00+M4, C10+M4
        # s=4: M5 = (A00+A01)B11 -> C00-M5, C01+M5
        # s=5: M6 = (A10-A00)B01 -> C11+M6
        # s=6: M7 = (A01-A11)B10 -> C00+M7
    ]
    
    # Re-defining terms explicitly to match the verified logic
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})        # M7
    ]
    
    for b in range(num_blocks):
        # Global indices for V (B) and W (C) for this block
        # V: B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
        # V indices: 0*14 + 2b, 0*14 + 2b+1, 1*14 + 2b, 1*14 + 2b+1
        v_idx = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # W: C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1]
        # W indices: k*n + i. 
        # C[0, 2b] -> k=2b, i=0 -> 2b*2 + 0 = 4b
        # C[0, 2b+1] -> k=2b+1, i=0 -> (2b+1)*2 + 0 = 4b+2
        # C[1, 2b] -> k=2b, i=1 -> 2b*2 + 1 = 4b+1
        # C[1, 2b+1] -> k=2b+1, i=1 -> (2b+1)*2 + 1 = 4b+3
        w_idx = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for s in range(7):
            col = b * 7 + s
            u_dict, v_dict, w_dict = terms[s]
            
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_idx[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_idx[l_idx]][col] += val
                
    return {"U": U, "V": V, "W": W}
