import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2 factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: U2[i*2+j, t], V2[j*2+k, t], W2[k*2+i, t]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # A11: (0,0), A22: (1,1)
    # B11: (0,0), B22: (1,1)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    # C11 += M1, C22 += M1
    W2[0][0] = 1; W2[3][0] = 1 # W2[k*2+i]. C11 is k=0,i=0 -> 0. C22 is k=1,i=1 -> 3.
    
    # M2 = (A21+A22)B11
    # A21: (1,0), A22: (1,1)
    # B11: (0,0)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    # C21 += M2, C22 -= M2
    W2[2][1] = 1 # C21 is k=0,i=1 -> 2
    W2[3][1] = -1 # C22 is k=1,i=1 -> 3
    
    # M3 = A11(B21-B22)
    # A11: (0,0)
    # B21: (1,0), B22: (1,1)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    # C12 += M3, C21 += M3
    W2[1][2] = 1 # C12 is k=1,i=0 -> 1
    W2[2][2] = 1 # C21 is k=0,i=1 -> 2
    
    # M4 = A22(B12-B11)
    # A22: (1,1)
    # B12: (0,1), B11: (0,0)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    # C11 += M4, C21 += M4
    W2[0][3] = 1 # C11
    W2[2][3] = 1 # C21
    
    # M5 = (A11+A12)B22
    # A11: (0,0), A12: (0,1)
    # B22: (1,1)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    # C11 -= M5, C12 += M5
    W2[0][4] = -1 # C11
    W2[1][4] = 1 # C12
    
    # M6 = (A21-A11)(B11+B12)
    # A21: (1,0), A11: (0,0)
    # B11: (0,0), B12: (0,1)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    # C22 += M6
    W2[3][5] = 1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    # A12: (0,1), A22: (1,1)
    # B21: (1,0), B22: (1,1)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    # C11 += M7
    W2[0][6] = 1 # C11

    # Outer Strassen factors (same structure)
    # Uo, Vo, Wo are 4x7
    Uo = [[0]*7 for _ in range(4)]
    Vo = [[0]*7 for _ in range(4)]
    Wo = [[0]*7 for _ in range(4)]
    
    # Copy U2 to Uo, V2 to Vo, W2 to Wo
    for i in range(4):
        for t in range(7):
            Uo[i][t] = U2[i][t]
            Vo[i][t] = V2[i][t]
            Wo[i][t] = W2[i][t]

    # Construct 4x4 factors
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Map global indices to local block indices
    # A: 4x4. Block p (0..3). Local (r,c).
    # p = (r//2)*2 + (c//2)
    # Local U2 index: r*2 + c
    
    for i_outer in range(7):
        for k_inner in range(7):
            col = i_outer * 7 + k_inner
            for r in range(4):
                for c in range(4):
                    p = (r // 2) * 2 + (c // 2)
                    local_idx = (r % 2) * 2 + (c % 2)
                    U[r * 4 + c][col] = Uo[p][i_outer] * U2[local_idx][k_inner]
                    
            for j in range(4):
                for k in range(4):
                    q = (j // 2) * 2 + (k // 2)
                    local_idx = (j % 2) * 2 + (k % 2)
                    V[j * 4 + k][col] = Vo[q][i_outer] * V2[local_idx][k_inner]
                    
            for k in range(4):
                for i in range(4):
                    m = (i // 2) * 2 + (k // 2)
                    local_idx = (k % 2) * 2 + (i % 2) # W2[k*2+i] -> local k is col, i is row?
                    # W2 index is k*2+i. In local 2x2, k is col, i is row.
                    # So local_idx = (k%2)*2 + (i%2)
                    W[k * 4 + i][col] = Wo[m][i_outer] * W2[local_idx][k_inner]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
