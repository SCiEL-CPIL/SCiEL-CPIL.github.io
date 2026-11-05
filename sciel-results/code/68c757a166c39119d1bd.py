def _strassen_4x4():
    """Generate 28-term decomposition for 4x4x4 matrix multiplication."""
    # Local indices for 4x4 matrix: index = row*4 + col
    # Block (br, bc) covers rows br*2..br*2+1, cols bc*2..bc*2+1
    # Block entry (br,bc,dr,dc) has index (br*2+dr)*4 + (bc*2+dc)
    
    def blk(br, bc, dr, dc):
        return (br*2+dr)*4 + (bc*2+dc)
    
    # A blocks
    A00 = [(blk(0,0,dr,dc)) for dr in range(2) for dc in range(2)]  # [0,1,4,5]
    A01 = [(blk(0,1,dr,dc)) for dr in range(2) for dc in range(2)]  # [2,3,6,7]
    A10 = [(blk(1,0,dr,dc)) for dr in range(2) for dc in range(2)]  # [8,9,12,13]
    A11 = [(blk(1,1,dr,dc)) for dr in range(2) for dc in range(2)]  # [10,11,14,15]
    
    # B blocks (same layout)
    B00 = A00[:]
    B01 = A01[:]
    B10 = A10[:]
    B11 = A11[:]
    
    # C blocks
    C00 = A00[:]
    C01 = A01[:]
    C10 = A10[:]
    C11 = A11[:]
    
    terms = []
    
    # Strassen block terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each M, it's a 2x2 matrix product X*Y
    # X[dr,dc] is a linear combination of A entries
    # Y[dr,dc] is a linear combination of B entries
    # M[dr,dc] = sum_s X[dr,s]*Y[s,dc]
    # M contributes to C with specific signs
    
    # Define Strassen terms as (X_blocks, Y_blocks, C_contrib)
    # X_blocks: list of (block_id, coeff) for the 2x2 matrix X
    # Y_blocks: list of (block_id, coeff) for the 2x2 matrix Y
    # C_contrib: dict mapping C block (br,bc) to coefficient
    
    strassen_defs = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1,(0,1):-1,(1,0):1,(1,1):1}),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], {(1,0):1,(1,1):-1}),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], {(0,1):1,(1,1):1}),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], {(0,0):1,(1,0):1}),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], {(0,0):-1,(0,1):1}),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], {(1,1):1}),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], {(0,0):1}),
    ]
    
    block_to_indices = {
        (0,0): A00, (0,1): A01, (1,0): A10, (1,1): A11
    }
    
    for x_blocks, y_blocks, c_contrib in strassen_defs:
        # Build X as 2x2 matrix of linear combinations
        # X[dr,dc] = sum of A entries from blocks with appropriate coeffs
        # For a block (br,bc) with coeff c, it contributes c*A[br*2+dr, bc*2+dc] to X[dr,dc]
        
        # X[dr,dc] is a dict: {A_local_idx: coeff}
        X = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc, coeff) in x_blocks:
            for dr in range(2):
                for dc in range(2):
                    idx = blk(br, bc, dr, dc)
                    X[dr][dc][idx] = X[dr][dc].get(idx, 0) + coeff
        
        # Y[dr,dc] is a dict: {B_local_idx: coeff}
        Y = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc, coeff) in y_blocks:
            for dr in range(2):
                for dc in range(2):
                    idx = blk(br, bc, dr, dc)
                    Y[dr][dc][idx] = Y[dr][dc].get(idx, 0) + coeff
        
        # M[dr,dc] = sum_s X[dr,s] * Y[s,dc]
        # Each product X[dr,s]*Y[s,dc] is a bilinear form
        # For the tensor decomposition, each such product is one rank-1 term
        
        for dr in range(2):
            for dc in range(2):
                # M[dr,dc] = sum_s X[dr,s] * Y[s,dc]
                # This is a sum of bilinear forms, but in the tensor decomposition,
                # each product X[dr,s]*Y[s,dc] is a separate term
                
                for s in range(2):
                    # u = X[dr,s] (dict of A indices)
                    # v = Y[s,dc] (dict of B indices)
                    # w = C[br,dc] for each C block in c_contrib
                    
                    u_dict = dict(X[dr][s])
                    v_dict = dict(Y[s][dc])
                    
                    # W: for each C block (cbr,cbc) with coeff cc
                    w_dict = {}
                    for (cbr, cbc), cc in c_contrib.items():
                        for cdr in range(2):
                            for cdc in range(2):
                                if cdr == dr and cdc == dc:
                                    idx = blk(cbr, cbc, cdr, cdc)
                                    w_dict[idx] = w_dict.get(idx, 0) + cc
                    
                    if u_dict and v_dict and w_dict:
                        terms.append((u_dict, v_dict, w_dict))
    
    return terms
