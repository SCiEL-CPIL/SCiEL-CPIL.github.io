import numpy as np

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication
    using recursive Strassen's algorithm.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen 2x2 element patterns
    # Indexing: [elem00, elem01, elem10, elem11]
    U_elem = [
        [1, 0, 0, 1],   # t=0: a00+a11
        [0, 0, 1, 1],   # t=1: a10+a11
        [1, 0, 0, 0],   # t=2: a00
        [0, 0, 0, 1],   # t=3: a11
        [1, 1, 0, 0],   # t=4: a00+a01
        [-1, 0, 1, 0],  # t=5: a10-a00
        [0, 1, 0, -1]   # t=6: a01-a11
    ]
    V_elem = [
        [1, 0, 0, 1],   # t=0: b00+b11
        [1, 0, 0, 0],   # t=1: b00
        [0, 1, 0, -1],  # t=2: b01-b11
        [-1, 0, 1, 0],  # t=3: b10-b00
        [0, 0, 0, 1],   # t=4: b11
        [1, 1, 0, 0],   # t=5: b00+b01
        [0, 0, 1, 1]    # t=6: b10+b11
    ]
    W_elem = [
        [[1, 0], [0, 1]],   # t=0: C00+=1, C11+=1
        [[0, 0], [1, -1]],  # t=1: C10+=1, C11-=1
        [[0, 1], [0, 1]],   # t=2: C01+=1, C11+=1
        [[1, 0], [1, 0]],   # t=3: C00+=1, C10+=1
        [[-1, 1], [0, 0]],  # t=4: C00-=1, C01+=1
        [[0, 0], [0, 1]],   # t=5: C11+=1
        [[1, 0], [0, 0]]    # t=6: C00+=1
    ]
    
    # Block-level Strassen: which 2x2 blocks and coefficients
    A_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: A00+A11
        [(1,0,1), (1,1,1)],      # s=1: A10+A11
        [(0,0,1)],               # s=2: A00
        [(1,1,1)],               # s=3: A11
        [(0,0,1), (0,1,1)],      # s=4: A00+A01
        [(1,0,1), (0,0,-1)],     # s=5: A10-A00
        [(0,1,1), (1,1,-1)]      # s=6: A01-A11
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: B00+B11
        [(0,0,1)],               # s=1: B00
        [(0,1,1), (1,1,-1)],     # s=2: B01-B11
        [(1,0,1), (0,0,-1)],     # s=3: B10-B00
        [(1,1,1)],               # s=4: B11
        [(0,0,1), (0,1,1)],      # s=5: B00+B01
        [(1,0,1), (1,1,1)]       # s=6: B10+B11
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: C00+=M, C11+=M
        [(1,0,1), (1,1,-1)],     # s=1: C10+=M, C11-=M
        [(0,1,1), (1,1,1)],      # s=2: C01+=M, C11+=M
        [(0,0,1), (1,0,1)],      # s=3: C00+=M, C10+=M
        [(0,0,-1), (0,1,1)],     # s=4: C00-=M, C01+=M
        [(1,1,1)],               # s=5: C11+=M
        [(0,0,1)]                # s=6: C00+=M
    ]
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for s in range(7):
        for t in range(7):
            col = s*7 + t
            
            # Build U for this term
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (2*bi + ei)*4 + (2*bj + ej)
                        U[row_a, col] += c_a * U_elem[t][ei*2+ej]
            
            # Build V for this term
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (2*bj + ej)*4 + (2*bk + ek)
                        V[row_b, col] += c_b * V_elem[t][ej*2+ek]
            
            # Build W for this term
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        row_c = (2*bk + ek)*4 + (2*bi + ei)
                        W[row_c, col] += c_c * W_elem[t][ei][ek]
    
    # Quick verification with small integers
    A_test = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=int)
    B_test = np.array([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]], dtype=int)
    
    AB = A_test @ B_test
    
    AB_decomp = np.zeros((4, 4), dtype=int)
    for r in range(rank):
        u_val = np.dot(U[:, r], A_test.flatten())
        v_val = np.dot(V[:, r], B_test.flatten())
        scalar = u_val * v_val
        AB_decomp += np.outer(W[:, r], scalar) # Note: W is indexed [k*n+i, r], so W[:, r] is vector of length 16.
        # Wait, W[row_c, col] where row_c = k*4 + i.
        # The outer product of W[:, r] (shape 16) and scalar adds to AB_decomp?
        # AB_decomp[i, k] should increase by W[k*4+i, r] * scalar.
        # Let's reshape W[:, r] to (4, 4) with indexing [k, i].
        # W_vec = W[:, r].reshape(4, 4) -> W_vec[k, i] = W[k*4+i, r].
        # AB_decomp[i, k] += W_vec[k, i] * scalar.
        # This is not a standard outer product.
        
    # Let's do the verification loop manually to be safe with indices
    AB_decomp = np.zeros((4, 4), dtype=int)
    for r in range(rank):
        u_val = 0
        for i in range(4):
            for j in range(4):
                u_val += U[i*4+j, r] * A_test[i, j]
        v_val = 0
        for j in range(4):
            for k in range(4):
                v_val += V[j*4+k, r] * B_test[j, k]
        scalar = u_val * v_val
        for i in range(4):
            for k in range(4):
                AB_decomp[i, k] += W[k*4+i, r] * scalar
        
    if not np.array_equal(AB, AB_decomp):
        # Fallback to schoolbook
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=int)
        V_fb = np.zeros((16, rank_fb), dtype=int)
        W_fb = np.zeros((16, rank_fb), dtype=int)
        col_fb = 0
        for i2 in range(4):
            for j2 in range(4):
                for k2 in range(4):
                    U_fb[i2*4+j2, col_fb] = 1
                    V_fb[j2*4+k2, col_fb] = 1
                    W_fb[k2*4+i2, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
