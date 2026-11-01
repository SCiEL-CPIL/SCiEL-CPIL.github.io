"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    # Indices:
    # U: 0=a00, 1=a01, 2=a10, 3=a11
    # V: 0=b00, 1=b01, 2=b02, 3=b10, 4=b11, 5=b12
    # W: 0=c00, 1=c10, 2=c01, 3=c11, 4=c02, 5=c12
    
    rank = 11
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]
    
    # Strassen for cols 0, 1
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[2][0] = 1 # c00, c01? No, W index is k*n+i.
    # c00 (k=0,i=0) -> 0. c10 (k=0,i=1) -> 1. c01 (k=1,i=0) -> 2. c11 (k=1,i=1) -> 3.
    # Strassen outputs:
    # c00 = M1 + M4 - M5 + M7
    # c10 = M2 + M4
    # c01 = M3 + M5
    # c11 = M1 - M2 + M3 - M6
    
    # M1
    W[0][0] = 1 # c00
    W[2][0] = 0 # c01? No, M1 contributes to c00 and c11.
    # Let's re-verify Strassen output mapping.
    # c00 (top-left) = M1 + M4 - M5 + M7
    # c10 (bottom-left) = M2 + M4
    # c01 (top-right) = M3 + M5
    # c11 (bottom-right) = M1 - M2 + M3 - M6
    
    # So:
    # M1: +1 to c00, +1 to c11
    W[0][0] = 1; W[3][0] = 1
    # M2: +1 to c10, -1 to c11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    # M3: -1 to c00, +1 to c01, -1 to c11? No.
    # M3 = a00(b01-b11).
    # c01 = M3 + M5. c11 = M1 - M2 + M3 - M6.
    # So M3: +1 to c01, +1 to c11.
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    # M4 = a11(b10-b00).
    # c00 = M1 + M4 - M5 + M7. c10 = M2 + M4.
    # So M4: +1 to c00, +1 to c10.
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    # M5 = (a00+a01)b11.
    # c00 = ... - M5. c01 = M3 + M5.
    # So M5: -1 to c00, +1 to c01.
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    # M6 = (a10-a00)(b00+b01).
    # c11 = ... - M6.
    # So M6: -1 to c11.
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = -1
    # M7 = (a01-a11)(b10+b11).
    # c00 = ... + M7.
    # So M7: +1 to c00.
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Schoolbook for col 2
    # M8 = a00 b02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1 # c02
    # M9 = a01 b12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1 # c02
    # M10 = a10 b02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1 # c12
    # M11 = a11 b12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1 # c12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
