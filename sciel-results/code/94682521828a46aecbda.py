"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    # We decompose M_{2,3,3} into M_{2,2,3} (j=0,1) and M_{2,1,3} (j=2)
    # M_{2,1,3} has rank 1
    # M_{2,2,3} has rank 12 (schoolbook)
    # Total rank = 13
    
    rank = 13
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*3 = 9 rows
    W = [[0] * rank for _ in range(p * n)]  # 3*2 = 6 rows
    
    # Part 1: j=2 (Rank 1)
    # A_{i,2} * B_{2,k} -> C_{k,i}
    col = 0
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Part 2: j in {0, 1} (Schoolbook Rank 12)
    # A_{i,j} * B_{j,k} -> C_{k,i}
    for j in range(2):
        for i in range(n):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
