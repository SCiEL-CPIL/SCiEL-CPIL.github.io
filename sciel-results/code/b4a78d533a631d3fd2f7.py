import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # We need U1, V1, W1 for 4x4x4
    U1, V1, W1 = get_strassen_4x4x4()
    
    # Part 2: 4x4x1 Schoolbook (Rank 16)
    # A is 4x4, B is 4x1. C is 4x1.
    # U2[i*4+j, c] = 1 if i,j match
    # V2[j*1+k, c] = 1 if j,k match (k=0)
    # W2[k*4+i, c] = 1 if k,i match (k=0)
    U2 = np.zeros((4*4, 16))
    V2 = np.zeros((4*1, 16))
    W2 = np.zeros((1*4, 16))
    c = 0
    for i in range(4):
        for j in range(4):
            U2[i*4+j, c] = 1
            V2[j*1+0, c] = 1
            W2[0*4+i, c] = 1
            c += 1
            
    # Part 3: 4x1x5 Schoolbook (Rank 20)
    # A is 4x1, B is 1x5. C is 4x5.
    # U3[i*1+j, c] = 1 if i,j match (j=0)
    # V3[j*5+k, c] = 1 if j,k match (j=0)
    # W3[k*4+i, c] = 1 if k,i match
    U3 = np.zeros((4*1, 20))
    V3 = np.zeros((1*5, 20))
    W3 = np.zeros((5*4, 20))
    c = 0
    for i in range(4):
        for k in range(5):
            U3[i*1+0, c] = 1
            V3[0*5+k, c] = 1
            W3[k*4+i, c] = 1
            c += 1
            
    # Combine
    # Total U: (20, 49+16+20)
    # Total V: (25, 49+16+20)
    # Total W: (20, 49+16+20)
    
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    return {"U": U, "V": V, "W": W}

def get_strassen_4x4x4():
    # Get U, V, W for 2x2x2
    U222, V222, W222 = get_strassen_2x2x2()
    
    # U1 is 16x49
    # V1 is 16x49
    # W1 is 16x49
    
    U1 = np.zeros((16, 49))
    V1 = np.zeros((16, 49))
    W1 = np.zeros((16, 49))
    
    # Strassen recursion
    # The 7 multiplications M1..M7
    # Each Mk is a 2x2x2 multiplication
    # We need to map the inputs/outputs of M1..M7 to the 4x4x4 inputs/outputs
    
    # Input A is 4x4. Blocks A11, A12, A21, A22 are 2x2.
    # Input B is 4x4. Blocks B11, B12, B21, B22 are 2x2.
    # Output C is 4x4. Blocks C11, C12, C21, C22 are 2x2.
    
    # The U, V, W for 2x2x2 operate on 4 inputs, 4 inputs, 4 outputs.
    # For 4x4x4, we have 16 inputs for A, 16 for B, 16 for C.
    
    # We construct U1, V1, W1 by placing the U222, V222, W222 in the appropriate blocks.
    
    # M1: (A11+A22)(B11+B22)
    # A parts: A11 (indices 0-3), A22 (indices 8-11)
    # B parts: B11 (indices 0-3), B22 (indices 8-11)
    # C parts: C11 (indices 0-3), C22 (indices 12-15)
    
    # We need to sum the contributions.
    # This is getting complex to hardcode.
    # Let's use a helper to build the block structure.
    
    # For each of the 7 Strassen steps:
    # Define the linear combinations of A blocks, B blocks.
    # The result is a 2x2 matrix.
    # This 2x2 matrix is then used in the 2x2x2 multiplication.
    
    # Actually, it's easier to just generate the U, V, W for the 7 steps and sum them up.
    
    # Strassen steps definitions:
    # (a, b) are block indices (0:11, 1:12, 2:21, 3:22)
    # A blocks: 0:0-3, 1:4-7, 2:8-11, 3:12-15
    # B blocks: 0:0-3, 1:4-7, 2:8-11, 3:12-15
    
    steps = [
        # (A_coeff, B_coeff)
        # M1
        ([{0: 1, 3: 1}], [{0: 1, 3: 1}]),
        # M2
        ([{2: 1, 3: 1}], [{0: 1}]),
        # M3
        ([{0: 1}], [{1: 1, 3: -1}]),
        # M4
        ([{3: 1}], [{2: 1, 0: -1}]),
        # M5
        ([{0: 1, 1: 1}], [{3: 1}]),
        # M6
        ([{2: 1, 0: -1}], [{0: 1, 1: 1}]),
        # M7
        ([{1: 1, 3: -1}], [{2: 1, 3: 1}])
    ]
    
    # Output C blocks for each step:
    # M1 -> C11 (+1), C22 (+1)
    # M2 -> C21 (+1), C22 (-1)
    # M3 -> C12 (+1), C22 (+1)
    # M4 -> C11 (+1), C21 (+1)
    # M5 -> C11 (-1), C12 (+1)
    # M6 -> C22 (+1)
    # M7 -> C11 (+1)
    
    c_contrib = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: -1}, # M2
        {1: 1, 3: 1}, # M3
        {0: 1, 2: 1}, # M4
        {0: -1, 1: 1}, # M5
        {3: 1}, # M6
        {0: 1} # M7
    ]
    
    for i in range(7):
        a_coeff = steps[i][0][0]
        b_coeff = steps[i][1][0]
        c_coeff = c_contrib[i]
        
        # For this step, we add 7 rank-1 tensors (from 2x2x2)
        for k in range(7):
            col = i * 7 + k
            
            # Update U
            for block_idx, val in a_coeff.items():
                # Block indices: 0:0-3, 1:4-7, 2:8-11, 3:12-15
                start = block_idx * 4
                # U222 is 4x7. The k-th column of U222 gives the weights for the 4 elements of the 2x2 block.
                U1[start:start+4, col] += val * U222[:, k]
                
            # Update V
            for block_idx, val in b_coeff.items():
                start = block_idx * 4
                V1[start:start+4, col] += val * V222[:, k]
                
            # Update W
            for block_idx, val in c_coeff.items():
                start = block_idx * 4
                W1[start:start+4, col] += val * W222[:, k]
                
    return U1, V1, W1

def get_strassen_2x2x2():
    # U, V, W for 2x2x2
    # Inputs: A (4), B (4)
    # Outputs: C (4)
    # Rank 7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1] # C11, C22
    
    # M2 = (A21+A22)B11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, -1] # C21, C22
    
    # M3 = A11(B12-B22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, 1] # C12, C22
    
    # M4 = A22(B21-B11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    W[:, 3] = [1, 0, 1, 0] # C11, C21
    
    # M5 = (A11+A12)B22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 1, 0, 0] # C11, C12
    
    # M6 = (A21-A11)(B11+B12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1] # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0] # C11
    
    return U, V, W
