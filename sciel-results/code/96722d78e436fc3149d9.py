"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 38 decomposition
    # Split m=4 into two m=2 blocks.
    # Each block is 2x2x5.
    # 2x2x5 = 2x2x2 (Strassen, rank 7) + 2x2x3 (Rank 12, schoolbook for 1x2x3 rows)
    # Total rank per block: 19.
    # Total rank: 38.
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 0: j in {0, 1}
    # A indices: i in {0,1}, j in {0,1} -> A[i, j]
    # B indices: j in {0,1}, k in {0..4} -> B[j, k]
    # C indices: i in {0,1}, k in {0..4} -> C[i, k]
    
    # Strassen for 2x2x2 part (k=0,1)
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(y-w)
    # M4 = d(z-x)
    # M5 = (a+b)w
    # M6 = (c+a)(x-z)
    # M7 = (b+d)(y+z)
    
    # U: A parts
    # V: B parts
    # W: C parts
    
    # M1
    U[0*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*5+0][col] = 1; V[0*5+1][col] = 1; V[1*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+0][col] = -1; W[0*2+1][col] = 1; W[1*2+1][col] = 1
    col += 1
    
    # M2
    U[1*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*5+0][col] = 1
    W[1*2+0][col] = 1
    col += 1
    
    # M3
    U[0*2+0][col] = 1
    V[0*5+1][col] = 1; V[1*5+1][col] = -1
    W[0*2+1][col] = 1
    col += 1
    
    # M4
    U[1*2+1][col] = 1
    V[1*5+0][col] = 1; V[0*5+0][col] = -1
    W[1*2+0][col] = 1; W[0*2+0][col] = 1
    col += 1
    
    # M5
    U[0*2+0][col] = 1; U[0*2+1][col] = 1
    V[1*5+1][col] = 1
    W[0*2+0][col] = -1; W[1*2+1][col] = 1
    col += 1
    
    # M6
    U[1*2+0][col] = 1; U[0*2+0][col] = 1
    V[0*5+0][col] = 1; V[1*5+0][col] = -1
    W[1*2+1][col] = 1
    col += 1
    
    # M7
    U[0*2+1][col] = 1; U[1*2+1][col] = 1
    V[0*5+1][col] = 1; V[1*5+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = -1
    col += 1
    
    # Schoolbook for 2x2x3 part (k=2,3,4)
    # 6 mults per row? No, 2 mults per element?
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # We can just do schoolbook for these 6 elements.
    # 6 elements * 2 mults = 12 mults.
    
    for i in range(2):
        for k in range(2, 5):
            # Term 1: A[i,0] * B[0,k]
            U[i*2+0][col] = 1
            V[0*5+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            # Term 2: A[i,1] * B[1,k]
            U[i*2+1][col] = 1
            V[1*5+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    # Block 1: j in {2, 3}
    # A indices: i in {0,1}, j in {2,3} -> A[i, j]
    # B indices: j in {2,3}, k in {0..4} -> B[j, k]
    # C indices: i in {0,1}, k in {0..4} -> C[i, k]
    
    # Strassen for 2x2x2 part (k=0,1)
    # Note: B indices are offset by 2 in j.
    
    # M1
    U[0*2+2][col] = 1; U[1*2+3][col] = 1
    V[2*5+0][col] = 1; V[2*5+1][col] = 1; V[3*5+0][col] = 1; V[3*5+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+0][col] = -1; W[0*2+1][col] = 1; W[1*2+1][col] = 1
    col += 1
    
    # M2
    U[1*2+2][col] = 1; U[1*2+3][col] = 1
    V[2*5+0][col] = 1
    W[1*2+0][col] = 1
    col += 1
    
    # M3
    U[0*2+2][col] = 1
    V[2*5+1][col] = 1; V[3*5+1][col] = -1
    W[0*2+1][col] = 1
    col += 1
    
    # M4
    U[1*2+3][col] = 1
    V[3*5+0][col] = 1; V[2*5+0][col] = -1
    W[1*2+0][col] = 1; W[0*2+0][col] = 1
    col += 1
    
    # M5
    U[0*2+2][col] = 1; U[0*2+3][col] = 1
    V[3*5+1][col] = 1
    W[0*2+0][col] = -1; W[1*2+1][col] = 1
    col += 1
    
    # M6
    U[1*2+2][col] = 1; U[0*2+2][col] = 1
    V[2*5+0][col] = 1; V[3*5+0][col] = -1
    W[1*2+1][col] = 1
    col += 1
    
    # M7
    U[0*2+3][col] = 1; U[1*2+3][col] = 1
    V[2*5+1][col] = 1; V[3*5+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = -1
    col += 1
    
    # Schoolbook for 2x2x3 part (k=2,3,4)
    for i in range(2):
        for k in range(2, 5):
            # Term 1: A[i,2] * B[2,k]
            U[i*2+2][col] = 1
            V[2*5+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            # Term 2: A[i,3] * B[3,k]
            U[i*2+3][col] = 1
            V[3*5+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
