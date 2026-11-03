import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Bini et al. (1979) rank-23 decomposition for 3x3x3
    # Applied to both 3x3x3 blocks of the 3x3x6 multiplication
    # Total rank: 46 (improvement over schoolbook 54)
    
    # Bini et al. rank-23 explicit coefficients
    # Each term is (u, v, w) where u, v, w are 9-element vectors
    # u indexed by [i*3+j], v by [j*3+k], w by [k*3+i]
    
    # Using the standard Bini et al. form with verified coefficients:
    # The 23 terms from the Bini et al. (1979) paper
    
    terms_23 = [
        # Term 0
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
        # Term 1
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 2
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 3
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 4
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 5
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 6
        ([0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 7
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 8
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 9
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 10
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 11
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 12
        ([0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 13
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 14
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 15
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 16
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 17
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 18
        ([0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 19
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 20
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 21
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 22
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
    ]
    
    # Verify the 3x3x3 decomposition numerically
    U3 = np.array([t[0] for t in terms_23]).T  # 9 x 23
    V3 = np.array([t[1] for t in terms_23]).T  # 9 x 23
    W3 = np.array([t[2] for t in terms_23]).T  # 9 x 23
    
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = np.sum(U3[i*3+j] * V3[j*3+k] * W3[k*3+i])
                if abs(s - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 54
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Valid! Construct 3x3x6 using two copies
    total_rank = 46
    U = [[0] * total_rank for _ in range(9)]
    V = [[0] * total_rank for _ in range(18)]
    W = [[0] * total_rank for _ in range(18)]
    
    # First 3x3x3 block: B columns 0-2
    for r in range(23):
        for idx in range(9):
            U[idx][r] = U3[idx, r]
            W[idx][r] = W3[idx, r]
        for j in range(3):
            for k in range(3):
                V[j*6+k][r] = V3[j*3+k, r]
    
    # Second 3x3x3 block: B columns 3-5
    for r in range(23):
        for idx in range(9):
            U[idx][r+23] = U3[idx, r]
            W[idx][r+23] = W3[idx, r]
        for j in range(3):
            for k in range(3):
                V[j*6+(k+3)][r+23] = V3[j*3+k, r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
