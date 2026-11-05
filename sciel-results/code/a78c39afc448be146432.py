DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2+2+1
    # Split p=5 into 5 (kept whole for simplicity in this draft, but actually p is the output dim)
    # We are decomposing the tensor.
    # Term 1: i in 0..1, j in 0..1, k in 0..4
    # Term 2: i in 0..1, j in 2..3, k in 0..4
    # Term 3: i in 0..1, j in 4, k in 0..4
    
    # Rank 1: Schoolbook for 2x2x5 (j=0,1)
    # Rank 2: Schoolbook for 2x2x5 (j=2,3)
    # Rank 3: Schoolbook for 2x1x5 (j=4)
    
    # Total rank = (2*2*5) + (2*2*5) + (2*1*5) = 20 + 20 + 10 = 50?
    # Wait. 2x1x5 schoolbook is 10.
    # 20+20+10 = 50. No gain.
    
    # To gain, we MUST use Strassen for the 2x2 parts.
    # 2x2x5 with Strassen?
    # As discussed, 2x2xp rank is not simply 7.
    
    # I will return the schoolbook solution to ensure validity.
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
# EVOLVE-BLOCK-END
