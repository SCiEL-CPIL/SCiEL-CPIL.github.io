import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U2, V2, W2 are 4x7 matrices
    # Indices for U2, V2, W2 are 0..3 (flattened 2x2)
    # Term 0: (A00+A11)(B00+B11)
    # Term 1: (A10+A11)B00
    # Term 2: A00(B10-B11)
    # Term 3: A11(B01-B00)
    # Term 4: (A00+A01)B11
    # Term 5: (A10-A00)(B00+B01)
    # Term 6: (A01-A11)(B10+B11)
    
    U2 = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 0, -1, -1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, 1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0], # B01
        [0, 0, 1, 0, 0, 0, 1], # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Outer Strassen for 4x4 (blocks of 2x2)
    # Uout, Vout, Wout are 4x7 matrices
    # Indices 0..3 correspond to blocks 00, 01, 10, 11
    
    Uout = [
        [1, 0, 1, 0, 1, 0, 0], # Block 00
        [0, 0, 0, 0, 1, 0, -1], # Block 01
        [0, 1, 0, 0, 0, 1, 0], # Block 10
        [1, 1, 0, 1, 0, 0, 1]  # Block 11
    ]
    
    Vout = [
        [1, 1, 0, -1, 0, 1, 0], # Block 00
        [0, 0, 0, 1, 0, 1, 0], # Block 01
        [0, 0, 1, 0, 0, 0, 1], # Block 10
        [1, 0, -1, 0, 1, 0, 1]  # Block 11
    ]
    
    Wout = [
        [1, 0, 0, 1, -1, 0, 1], # Block 00
        [0, 0, 1, 0, 1, 0, 0], # Block 01
        [0, 1, 0, 1, 0, 0, 0], # Block 10
        [1, -1, 1, 0, 0, 1, 0]  # Block 11
    ]
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get block index and local indices
    def get_block_indices(i, j):
        # i, j are 0..3
        # Block index b is 0..3
        # Local indices u, v are 0..1
        b = (i // 2) * 2 + (j // 2)
        u = i % 2
        v = j % 2
        return b, u, v

    # Construct first half (k=0..3)
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            # U part
            for i in range(4):
                for j in range(4):
                    b, u, v = get_block_indices(i, j)
                    # Coeff in L_A[s] at [u,v] is Uout[b, s]
                    # Coeff in inner term t for entry [u,v] is U2[u*2+v, t]
                    coeff = Uout[b][s] * U2[u*2+v][t]
                    U[i*m+j][c] = coeff
            
            # V part (for k=0..3)
            for j in range(4):
                for k in range(4):
                    b, u, v = get_block_indices(j, k)
                    # Coeff in L_B[s] at [u,v] is Vout[b, s]
                    # Coeff in inner term t for entry [u,v] is V2[u*2+v, t]
                    coeff = Vout[b][s] * V2[u*2+v][t]
                    V[j*p+k][c] = coeff
            
            # W part (for k=0..3)
            for k in range(4):
                for i in range(4):
                    b, u, v = get_block_indices(i, k)
                    # Coeff in C block b at [u,v] is Wout[b, s]
                    # Coeff in inner term t for entry [u,v] is W2[u*2+v, t]
                    # Note: W index is k*n + i
                    coeff = Wout[b][s] * W2[u*2+v][t]
                    W[k*n+i][c] = coeff

    # Construct second half (k=4..7)
    # Same U, different V and W indices
    for s in range(7):
        for t in range(7):
            c = 49 + s * 7 + t
            # U part is same
            for i in range(4):
                for j in range(4):
                    b, u, v = get_block_indices(i, j)
                    coeff = Uout[b][s] * U2[u*2+v][t]
                    U[i*m+j][c] = coeff
            
            # V part (for k=4..7)
            for j in range(4):
                for k in range(4, 8):
                    # Map k=4..7 to local k'=0..3
                    k_local = k - 4
                    b, u, v = get_block_indices(j, k_local)
                    coeff = Vout[b][s] * V2[u*2+v][t]
                    V[j*p+k][c] = coeff
            
            # W part (for k=4..7)
            for k in range(4, 8):
                k_local = k - 4
                for i in range(4):
                    b, u, v = get_block_indices(i, k_local)
                    coeff = Wout[b][s] * W2[u*2+v][t]
                    W[k*n+i][c] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
