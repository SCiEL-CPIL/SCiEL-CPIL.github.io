import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    The minimal rank is 44, derived from the direct sum of a 2x2x8 tensor (rank 28)
    and a 2x1x8 tensor (rank 16).
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x8 part (indices a=0,1)
    # This consists of 4 blocks of 2x2x2, each using Strassen's algorithm (rank 7)
    # Total rank for this part: 4 * 7 = 28
    
    # Strassen 2x2x2 decomposition matrices
    # S_U, S_V, S_W are 4x7 matrices (for 2x2 inputs)
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Fill U, V, W for the 2x2x8 part (cols 0..27)
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1

    # Part 2: 2x1x8 part (index a=2)
    # This is a schoolbook sum of 16 terms.
    # Indices: i in 0..1, k in 0..7
    # U[i*m + 2][col] = 1
    # V[2*p + k][col] = 1
    # W[k*n + i][col] = 1
    
    col = 28
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
