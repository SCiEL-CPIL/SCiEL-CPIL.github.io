"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We'll build the decomposition using:
    # - Strassen (rank 7) for the 2x2x2 sub-block: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    # - Schoolbook for all other block products
    
    # Total rank = 7 (Strassen) + schoolbook for remaining parts
    # Remaining parts:
    # A[0:2, 2:5] x B[2:5, 0:2] -> C[0:2, 0:2]: 2*3*2 = 12
    # A[0:2, 2:5] x B[2:5, 2:5] -> C[0:2, 2:5]: 2*3*3 = 18
    # A[2:3, 0:2] x B[0:2, 0:2] -> C[2:3, 0:2]: 1*2*2 = 4
    # A[2:3, 0:2] x B[0:2, 2:5] -> C[2:3, 2:5]: 1*2*3 = 6
    # A[2:3, 2:5] x B[2:5, 0:2] -> C[2:3, 0:2]: 1*3*2 = 6
    # A[2:3, 2:5] x B[2:5, 2:5] -> C[2:3, 2:5]: 1*3*3 = 9
    # A[0:2, 0:2] x B[0:2, 2:5] -> C[0:2, 2:5]: 2*2*3 = 12
    # Total schoolbook: 12+18+4+6+6+9+12 = 67
    # Total rank: 7 + 67 = 74
    
    rank = 74
    
    U = [[0] * rank for _ in range(n * m)]  # 3*5 = 15 rows
    V = [[0] * rank for _ in range(m * p)]  # 5*5 = 25 rows
    W = [[0] * rank for _ in range(p * n)]  # 5*3 = 15 rows
    
    col = 0
    
    # Strassen for A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: i in {0,1}, j in {0,1} -> U[i*5 + j]
    # B indices: j in {0,1}, k in {0,1} -> V[j*5 + k]
    # C indices: k in {0,1}, i in {0,1} -> W[k*3 + i]
    
    # Strassen M1 = (A00+A11)(B00+B11)
    # A part: A[0,0] + A[1,1] -> U[0*5+0]=U[0], U[1*5+1]=U[6]
    # B part: B[0,0] + B[1,1] -> V[0*5+0]=V[0], V[1*5+1]=V[6]
    U[0][col] = 1; U[6][col] = 1
    V[0][col] = 1; V[6][col] = 1
    # C part: C[0,0] += M1, C[1,1] += M1
    # W[k*3+i]: C[0,0] -> W[0*3+0]=W[0], C[1,1] -> W[1*3+1]=W[4]
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    # A part: A[1,0] + A[1,1] -> U[1*5+0]=U[5], U[1*5+1]=U[6]
    # B part: B[0,0] -> V[0]
    U[5][col] = 1; U[6][col] = 1
    V[0][col] = 1
    # C part: C[1,0] += M2, C[1,1] -= M2
    # W[0*3+1]=W[1], W[1*3+1]=W[4]
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B10-B11)
    # A part: A[0,0] -> U[0]
    # B part: B[1,0] - B[1,1] -> V[1*5+0]=V[5], V[1*5+1]=V[6]
    U[0][col] = 1
    V[5][col] = 1; V[6][col] = -1
    # C part: C[0,1] += M3, C[1,1] += M3
    # W[1*3+0]=W[3], W[1*3+1]=W[4]
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B01-B00)
    # A part: A[1,1] -> U[6]
    # B part: B[0,1] - B[0,0] -> V[0*5+1]=V[1], V[0]
    U[6][col] = 1
    V[1][col] = 1; V[0][col] = -1
    # C part: C[0,0] += M4, C[1,0] += M4
    # W[0], W[1]
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    # A part: A[0,0] + A[0,1] -> U[0], U[0*5+1]=U[1]
    # B part: B[1,1] -> V[6]
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    # C part: C[0,0] -= M5, C[0,1] += M5
    # W[0], W[3]
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B10)
    # A part: A[1,0] - A[0,0] -> U[5], U[0]
    # B part: B[0,0] + B[1,0] -> V[0], V[5]
    U[5][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[5][col] = 1
    # C part: C[1,1] += M6
    # W[4]
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B01+B11)
    # A part: A[0,1] - A[1,1] -> U[1], U[6]
    # B part: B[0,1] + B[1,1] -> V[1], V[6]
    U[1][col] = 1; U[6][col] = -1
    V[1][col] = 1; V[6][col] = 1
    # C part: C[0,0] += M7
    # W[0]
    W[0][col] = 1
    col += 1
    
    # Now schoolbook for remaining parts
    
    # A[0:2, 2:5] x B[2:5, 0:2] -> C[0:2, 0:2]
    # i in {0,1}, j in {2,3,4}, k in {0,1}
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A[0:2, 2:5] x B[2:5, 2:5] -> C[0:2, 2:5]
    # i in {0,1}, j in {2,3,4}, k in {2,3,4}
    for i in [0, 1]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A[2:3, 0:2] x B[0:2, 0:2] -> C[2:3, 0:2]
    # i=2, j in {0,1}, k in {0,1}
    for i in [2]:
        for j in [0, 1]:
            for k in [0, 1]:
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A[2:3, 0:2] x B[0:2, 2:5] -> C[2:3, 2:5]
    # i=2, j in {0,1}, k in {2,3,4}
    for i in [2]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A[2:3, 2:5] x B[2:5, 0:2] -> C[2:3, 0:2]
    # i=2, j in {2,3,4}, k in {0,1}
    for i in [2]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A[2:3, 2:5] x B[2:5, 2:5] -> C[2:3, 2:5]
    # i=2, j in {2,3,4}, k in {2,3,4}
    for i in [2]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # A[0:2, 0:2] x B[0:2, 2:5] -> C[0:2, 2:5]
    # i in {0,1}, j in {0,1}, k in {2,3,4}
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
