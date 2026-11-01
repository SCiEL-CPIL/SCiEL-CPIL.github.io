"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition: Strassen for first 2 columns, Schoolbook for 3rd.
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (n*m=4):
    # 0: (0,0) a, 1: (0,1) b, 2: (1,0) c, 3: (1,1) d
    # Indices for B (m*p=6):
    # 0: (0,0) e, 1: (0,1) f, 2: (0,2) x
    # 3: (1,0) g, 4: (1,1) h, 5: (1,2) y
    # Indices for C (p*n=6):
    # k*n+i
    # 0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1), 4: (0,2), 5: (1,2)
    # Wait, W index is k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5

    # Strassen for B[:, 0:2]
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)(g+h)
    U[2][1] = 1; U[3][1] = 1
    V[3][1] = 1; V[4][1] = 1
    W[3][1] = -1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d(g-f)
    U[3][3] = 1
    V[3][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1; W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1

    # Schoolbook for B[:, 2]
    # C02 = a*x + b*y
    # C12 = c*x + d*y
    
    # M8 = a*x
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # M9 = b*y
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # M10 = c*x
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # M11 = d*y
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
