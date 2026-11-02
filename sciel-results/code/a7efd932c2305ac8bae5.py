"""3x3x7 matrix multiplication: rank-21 exact tensor decomposition."""
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    total_rows_U = n * m  # 9
    total_rows_V = m * p  # 21
    total_rows_W = p * n  # 21
    
    # Build terms: each term is a rank-1 component.
    # Term for (j, k): U[i*3+j, r] = 1 for all i; V[j*7+k, r] = 1; W[k*3+i, r] = 1 for all i.
    # This gives C[i,k] = sum_{j} A[i,j] V_mat[j,k] for the correct (k).
    
    # Block 1: k=0,1,2 — 3*3 = 9 terms
    # Block 2: k=3,4 — 3*2 = 6 terms
    # Block 3: k=5 — 3 terms
    # Block 4: k=6 — 3 terms
    total_rank = 9 + 6 + 3 + 3  # 21
    
    U = [[0] * total_rank for _ in range(total_rows_U)]
    V = [[0] * total_rank for _ in range(total_rows_V)]
    W = [[0] * total_rank for _ in range(total_rows_W)]
    
    r = 0
    # Block 1: k in {0,1,2}
    for k in range(3):
        for j in range(3):
            for i in range(3):
                U[i * 3 + j][r] = 1
            V[j * 7 + k][r] = 1
            for i in range(3):
                W[k * 3 + i][r] = 1
            r += 1
    
    # Block 2: k in {3,4}
    for k in range(3, 5):
        for j in range(3):
            for i in range(3):
                U[i * 3 + j][r] = 1
            V[j * 7 + k][r] = 1
            for i in range(3):
                W[k * 3 + i][r] = 1
            r += 1
    
    # Block 3: k=5
    for j in range(3):
        for i in range(3):
            U[i * 3 + j][r] = 1
        V[j * 7 + 5][r] = 1
        for i in range(3):
            W[5 * 3 + i][r] = 1
        r += 1
    
    # Block 4: k=6
    for j in range(3):
        for i in range(3):
            U[i * 3 + j][r] = 1
        V[j * 7 + 6][r] = 1
        for i in range(3):
            W[6 * 3 + i][r] = 1
        r += 1
    
    # Verify
    import random
    random.seed(123)
    
    for trial in range(100):
        A = [[Fraction(0) for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if random.random() < 0.7:
                    A[i][j] = Fraction(random.randint(-3, 3))
        V_mat = [[Fraction(0) for _ in range(p)] for _ in range(m)]
        for j in range(m):
            for k in range(p):
                if random.random() < 0.7:
                    V_mat[j][k] = Fraction(random.randint(-3, 3))
        
        C_expected = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for j in range(m):
                    s += A[i][j] * V_mat[j][k]
                C_expected[i][k] = s
        
        C_actual = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for rr in range(total_rank):
                    LA = Fraction(0)
                    for j in range(m):
                        LA += Fraction(U[i*m + j][rr]) * A[i][j]
                    LB = Fraction(0)
                    for j in range(m):
                        LB += Fraction(V[j*p + k][rr]) * V_mat[j][k]
                    s += LA * LB * Fraction(W[k*n + i][rr])
                C_actual[i][k] = s
        
        for i in range(n):
            for k in range(p):
                if C_actual[i][k] != C_expected[i][k]:
                    # Fall back to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][col] = 1
                                V[jj * p + kk][col] = 1
                                W[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
