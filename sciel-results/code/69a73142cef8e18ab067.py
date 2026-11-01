"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    
    # Rank 53: 7 Strassen blocks for 14 columns + 4 naive terms for last column.
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # u: [A00, A01, A10, A11]
    # v: [B00, B01, B10, B11] (indices j*2+k)
    # w: [C00, C10, C01, C11] (indices k*n+i)
    # W index mapping:
    # idx 0: k=0, i=0 (C00)
    # idx 1: k=0, i=1 (C10)
    # idx 2: k=1, i=0 (C01)
    # idx 3: k=1, i=1 (C11)
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    # Process 7 pairs of columns
    for q in range(7):
        for u_vec, v_vec, w_vec in terms:
            # Fill U
            for idx in range(4):
                U[idx][col] = u_vec[idx]
            
            # Fill V
            V[0 * 15 + (2 * q)][col] = v_vec[0]
            V[0 * 15 + (2 * q + 1)][col] = v_vec[1]
            V[1 * 15 + (2 * q)][col] = v_vec[2]
            V[1 * 15 + (2 * q + 1)][col] = v_vec[3]
            
            # Fill W
            W[2 * (2 * q) + 0][col] = w_vec[0]
            W[2 * (2 * q) + 1][col] = w_vec[1]
            W[2 * (2 * q + 1) + 0][col] = w_vec[2]
            W[2 * (2 * q + 1) + 1][col] = w_vec[3]
            
            col += 1
            
    # Process last column (k=14) with 4 naive terms
    U[0][col] = 1
    V[0 * 15 + 14][col] = 1
    W[2 * 14 + 0][col] = 1
    col += 1
    
    U[1][col] = 1
    V[1 * 15 + 14][col] = 1
    W[2 * 14 + 0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[0 * 15 + 14][col] = 1
    W[2 * 14 + 1][col] = 1
    col += 1
    
    U[3][col] = 1
    V[1 * 15 + 14][col] = 1
    W[2 * 14 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
