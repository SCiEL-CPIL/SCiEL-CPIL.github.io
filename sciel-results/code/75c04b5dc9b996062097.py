"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=4
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 14)
    V = [[0] * rank for _ in range(m * p)]  # (8, 14)
    W = [[0] * rank for _ in range(p * n)]  # (8, 14)
    
    # Left half (k=0,1) - Strassen
    # M1_L = (A00+A11)(B00+B11) -> C00+=, C11+=
    c = 0
    U[0][c] = 1; U[3][c] = 1
    V[0][c] = 1; V[5][c] = 1
    W[0][c] = 1; W[3][c] = 1
    
    # M2_L = (A10+A11)B00 -> C10+=, C11-=
    c = 1
    U[2][c] = 1; U[3][c] = 1
    V[0][c] = 1
    W[1][c] = 1; W[3][c] = -1
    
    # M3_L = A00(B01-B11) -> C01+=, C11+=
    c = 2
    U[0][c] = 1
    V[1][c] = 1; V[5][c] = -1
    W[2][c] = 1; W[3][c] = 1
    
    # M4_L = A11(B10-B00) -> C00+=, C10+=
    c = 3
    U[3][c] = 1
    V[4][c] = 1; V[0][c] = -1
    W[0][c] = 1; W[1][c] = 1
    
    # M5_L = (A00+A01)B11 -> C00-=, C01+=
    c = 4
    U[0][c] = 1; U[1][c] = 1
    V[5][c] = 1
    W[0][c] = -1; W[2][c] = 1
    
    # M6_L = (A10-A00)(B00+B01) -> C11+=
    c = 5
    U[2][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[1][c] = 1
    W[3][c] = 1
    
    # M7_L = (A01-A11)(B10+B11) -> C00+=
    c = 6
    U[1][c] = 1; U[3][c] = -1
    V[4][c] = 1; V[5][c] = 1
    W[0][c] = 1
    
    # Right half (k=2,3) - Strassen
    # M1_R = (A00+A11)(B02+B13) -> C02+=, C13+=
    c = 7
    U[0][c] = 1; U[3][c] = 1
    V[2][c] = 1; V[7][c] = 1
    W[4][c] = 1; W[7][c] = 1
    
    # M2_R = (A10+A11)B02 -> C12+=, C13-=
    c = 8
    U[2][c] = 1; U[3][c] = 1
    V[2][c] = 1
    W[5][c] = 1; W[7][c] = -1
    
    # M3_R = A00(B03-B13) -> C03+=, C13+=
    c = 9
    U[0][c] = 1
    V[3][c] = 1; V[7][c] = -1
    W[6][c] = 1; W[7][c] = 1
    
    # M4_R = A11(B12-B02) -> C02+=, C12+=
    c = 10
    U[3][c] = 1
    V[6][c] = 1; V[2][c] = -1
    W[4][c] = 1; W[5][c] = 1
    
    # M5_R = (A00+A01)B13 -> C02-=, C03+=
    c = 11
    U[0][c] = 1; U[1][c] = 1
    V[7][c] = 1
    W[4][c] = -1; W[6][c] = 1
    
    # M6_R = (A10-A00)(B02+B03) -> C13+=
    c = 12
    U[2][c] = 1; U[0][c] = -1
    V[2][c] = 1; V[3][c] = 1
    W[7][c] = 1
    
    # M7_R = (A01-A11)(B12+B13) -> C02+=
    c = 13
    U[1][c] = 1; U[3][c] = -1
    V[6][c] = 1; V[7][c] = 1
    W[4][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
