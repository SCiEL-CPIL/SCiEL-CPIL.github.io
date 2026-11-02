import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=3 into 2+1
    # Split p=9 into 8+1
    # A is 2x3: A_L (2x2), A_R (2x1)
    # B is 3x9: B_TL (2x8), B_TR (2x1), B_BL (1x8), B_BR (1x1)
    
    # Ranks:
    # Part 1: A_L * B_TL (2x2 * 2x8) -> 4 * M(2,2,2) = 4 * 7 = 28
    # Part 2: A_L * B_TR (2x2 * 2x1) -> M(2,2,1) = 4
    # Part 3: A_R * B_BL (2x1 * 1x8) -> M(2,1,8) = 16
    # Part 4: A_R * B_BR (2x1 * 1x1) -> M(2,1,1) = 2
    # Total Rank = 50
    
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a trivial term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1

    # Part 1: A_L (2x2) * B_TL (2x8)
    # B_TL columns 0..7 correspond to B[0:2, 0:8]
    # A_L rows 0..1 correspond to A[0:2, 0:2]
    # Output C[0:2, 0:8]
    
    for block_idx in range(4):
        # B_TL block is 2x2, cols 2*block_idx to 2*block_idx+1
        # In B global index: j in {0,1}, k in {2*block_idx, 2*block_idx+1}
        # V index: j*p + k
        # A_L is 2x2, i in {0,1}, j in {0,1}
        # U index: i*m + j
        # W index: k*n + i
        
        # Strassen 7 terms for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # A11: U[0,0]=1, A22: U[1,1]=1
        # B11: V[0, 2*block_idx]=1, B22: V[1, 2*block_idx+1]=1
        
        # We need to add linear combinations.
        # Instead of explicit Strassen variables, we add the 7 basis tensors.
        
        # Term 1: (A11+A22)(B11+B22)
        # U: [1, 1, 0, 0]
        # V: [1, 1, 0, 0] (for B11, B22)
        # W: [1, 1, 0, 0] (for C11, C22)
        
        # Let's just use the standard Strassen decomposition explicitly.
        
        # 1. (A11+A22)(B11+B22)
        U[0, col] = 1; U[1, col] = 1
        V[0, col] = 1; V[1, col] = 1 # B11, B22 relative to block
        # Wait, V index is j*p + k.
        # j=0, k=2*block_idx -> V[0*3 + 2*block_idx, col]
        # j=1, k=2*block_idx+1 -> V[1*3 + 2*block_idx+1, col]
        
        V[0*3 + 2*block_idx, col] = 1
        V[1*3 + (2*block_idx+1), col] = 1
        
        # W: C11, C22
        # C11: k=2*block_idx, i=0 -> W[2*block_idx*2 + 0, col]
        # C22: k=2*block_idx+1, i=1 -> W[(2*block_idx+1)*2 + 1, col]
        W[(2*block_idx)*2 + 0, col] = 1
        W[(2*block_idx+1)*2 + 1, col] = 1
        col += 1
        
        # 2. (A21+A22)B11
        U[1, col] = 1 # A21 (i=1, j=0)
        U[1, col] = 1 # A22 (i=1, j=1) -> U[1, col] += 1
        # U[1, col] = 1, U[1, col] = 1 -> U[1, col] = 2? No, U[1,0]=1, U[1,1]=1.
        U[1*3 + 0, col] = 1 # A21
        U[1*3 + 1, col] = 1 # A22
        
        V[0*3 + 2*block_idx, col] = 1 # B11
        
        # W: C21
        # C21: k=2*block_idx, i=1
        W[(2*block_idx)*2 + 1, col] = 1
        col += 1
        
        # 3. A11(B12-B22)
        U[0*3 + 0, col] = 1 # A11
        
        V[0*3 + (2*block_idx+1), col] = 1 # B12
        V[1*3 + (2*block_idx+1), col] = -1 # -B22
        
        # W: C12
        # C12: k=2*block_idx+1, i=0
        W[(2*block_idx+1)*2 + 0, col] = 1
        col += 1
        
        # 4. A22(B21-B11)
        U[1*3 + 1, col] = 1 # A22
        
        V[1*3 + 2*block_idx, col] = 1 # B21
        V[0*3 + 2*block_idx, col] = -1 # -B11
        
        # W: C21
        # C21: k=2*block_idx, i=1
        W[(2*block_idx)*2 + 1, col] = 1
        col += 1
        
        # 5. (A11+A12)B22
        U[0*3 + 0, col] = 1 # A11
        U[0*3 + 1, col] = 1 # A12
        
        V[1*3 + (2*block_idx+1), col] = 1 # B22
        
        # W: C12
        # C12: k=2*block_idx+1, i=0
        W[(2*block_idx+1)*2 + 0, col] = 1
        col += 1
        
        # 6. (A21-A11)(B11+B12)
        U[1*3 + 0, col] = 1 # A21
        U[0*3 + 0, col] = -1 # -A11
        
        V[0*3 + 2*block_idx, col] = 1 # B11
        V[0*3 + (2*block_idx+1), col] = 1 # B12
        
        # W: C22
        # C22: k=2*block_idx+1, i=1
        W[(2*block_idx+1)*2 + 1, col] = 1
        col += 1
        
        # 7. (A12-A22)(B21+B22)
        U[0*3 + 1, col] = 1 # A12
        U[1*3 + 1, col] = -1 # -A22
        
        V[1*3 + 2*block_idx, col] = 1 # B21
        V[1*3 + (2*block_idx+1), col] = 1 # B22
        
        # W: C22
        # C22: k=2*block_idx+1, i=1
        W[(2*block_idx+1)*2 + 1, col] = 1
        col += 1

    # Part 2: A_L (2x2) * B_TR (2x1)
    # B_TR is B[0:2, 8]
    # A_L is A[0:2, 0:2]
    # Output C[0:2, 8]
    
    # Trivial 4 terms
    # C11 = A11 B11 + A12 B21
    # C21 = A21 B11 + A22 B21
    
    # Term 1: A11 B11
    U[0*3 + 0, col] = 1
    V[0*3 + 8, col] = 1
    W[8*2 + 0, col] = 1
    col += 1
    
    # Term 2: A12 B21
    U[0*3 + 1, col] = 1
    V[1*3 + 8, col] = 1
    W[8*2 + 0, col] = 1
    col += 1
    
    # Term 3: A21 B11
    U[1*3 + 0, col] = 1
    V[0*3 + 8, col] = 1
    W[8*2 + 1, col] = 1
    col += 1
    
    # Term 4: A22 B21
    U[1*3 + 1, col] = 1
    V[1*3 + 8, col] = 1
    W[8*2 + 1, col] = 1
    col += 1

    # Part 3: A_R (2x1) * B_BL (1x8)
    # A_R is A[0:2, 2]
    # B_BL is B[2:3, 0:8]
    # Output C[0:2, 0:8]
    
    # Trivial 16 terms
    for i in range(2):
        for k in range(8):
            U[i*3 + 2, col] = 1
            V[2*3 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1

    # Part 4: A_R (2x1) * B_BR (1x1)
    # A_R is A[0:2, 2]
    # B_BR is B[2:3, 8]
    # Output C[0:2, 8]
    
    # Trivial 2 terms
    for i in range(2):
        U[i*3 + 2, col] = 1
        V[2*3 + 8, col] = 1
        W[8*2 + i, col] = 1
        col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
