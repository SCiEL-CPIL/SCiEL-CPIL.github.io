import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The matrix multiplication $C = AB$ with $A \in \mathbb{R}^{2 \times 2}$ and $B \in \mathbb{R}^{2 \times 10}$
    can be decomposed into 5 independent $2 \times 2$ matrix multiplications by pairing the columns of B.
    Let $B$ be partitioned into 5 blocks $B_0, \dots, B_4$ of size $2 \times 2$.
    Then $C$ is partitioned into 5 blocks $C_0, \dots, C_4$ of size $2 \times 2$.
    Each block product $C_b = A B_b$ can be computed using Strassen's algorithm with rank 7.
    Since the A factor is shared, but the tensor rank of the sum of such blocks (with shared A)
    for $2 \times 2$ matrices does not allow further reduction below the sum of the individual ranks
    (as $R(M_{2,2,2})=7$ is tight), the total rank is $5 \times 7 = 35$.
    
    Returns U, V, W matrices of shapes (4, 35), (20, 35), (20, 35) respectively.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Coefficients for A (u vectors)
    # Indices: 0:a, 1:b, 2:c, 3:d
    strassen_U = np.array([
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ])
    
    # Coefficients for B (v vectors)
    # Indices: 0:e, 1:f, 2:g, 3:h
    strassen_V = np.array([
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ])
    
    # Coefficients for C (w vectors)
    # C = [[c00, c01], [c10, c11]]
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    
    # Indices for C in the block: 0:c00, 1:c01, 2:c10, 3:c11
    strassen_W = np.array([
        [1, 0, 0, 0],  # P1 contributes to c00
        [0, 0, 0, 0],  # P2 contributes to c10 (idx 2) and c11 (idx 3) -> wait, let's use explicit matrix
    ])
    
    # Let's construct W matrix explicitly for the 7 products
    # w[r][c_idx]
    w_block = np.zeros((7, 4))
    # P1
    w_block[0, 0] = 1  # c00
    w_block[0, 3] = 1  # c11
    # P2
    w_block[1, 2] = 1  # c10
    w_block[1, 3] = -1 # c11
    # P3
    w_block[2, 1] = 1  # c01
    w_block[2, 3] = 1  # c11
    # P4
    w_block[3, 0] = 1  # c00
    w_block[3, 2] = 1  # c10
    # P5
    w_block[4, 0] = -1 # c00
    w_block[4, 1] = 1  # c01
    # P6
    w_block[5, 3] = 1  # c11
    # P7
    w_block[6, 0] = 1  # c00
    
    col = 0
    for b in range(5):
        for r in range(7):
            # U part (same for all blocks)
            U[:, col] = strassen_U[r]
            
            # V part
            # B indices for block b:
            # e = B[0, 2b] -> idx 2b
            # f = B[0, 2b+1] -> idx 2b+1
            # g = B[1, 2b] -> idx 10 + 2b
            # h = B[1, 2b+1] -> idx 10 + 2b + 1
            
            V[2*b, col] = strassen_V[r, 0]
            V[2*b+1, col] = strassen_V[r, 1]
            V[10+2*b, col] = strassen_V[r, 2]
            V[10+2*b+1, col] = strassen_V[r, 3]
            
            # W part
            # C indices for block b:
            # c00 -> C[0, 2b] -> idx (2b)*2 + 0 = 4b
            # c01 -> C[0, 2b+1] -> idx (2b+1)*2 + 0 = 4b+2
            # c10 -> C[1, 2b] -> idx (2b)*2 + 1 = 4b+1
            # c11 -> C[1, 2b+1] -> idx (2b+1)*2 + 1 = 4b+3
            
            W[4*b, col] = w_block[r, 0]
            W[4*b+2, col] = w_block[r, 1]
            W[4*b+1, col] = w_block[r, 2]
            W[4*b+3, col] = w_block[r, 3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
