"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # Rank is 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for h in range(4):
        # Strassen part for 2x2x2 block (j=0,1 and k=2h, 2h+1)
        # U rows: 0 (0,0), 1 (0,1), 3 (1,0), 4 (1,1)
        # V rows: 2h (0,2h), 2h+1 (0,2h+1), 8+2h (1,2h), 8+2h+1 (1,2h+1)
        # W rows: 4h (2h,0), 4h+1 (2h+1,0), 4h+2 (2h,1), 4h+3 (2h+1,1)
        
        # Term 1: M1 = (a00+a11)(b00+b11)
        U[0][col] += 1
        U[4][col] += 1
        V[2*h][col] += 1
        V[8+2*h+1][col] += 1
        W[4*h][col] += 1
        W[4*h+3][col] += 1
        col += 1
        
        # Term 2: M2 = (a10+a11)b00
        U[3][col] += 1
        U[4][col] += 1
        V[2*h][col] += 1
        W[4*h+2][col] += 1
        W[4*h+3][col] -= 1
        col += 1
        
        # Term 3: M3 = a00(b10-b11)
        U[0][col] += 1
        V[8+2*h][col] += 1
        V[8+2*h+1][col] -= 1
        W[4*h+1][col] += 1
        W[4*h+3][col] += 1
        col += 1
        
        # Term 4: M4 = a11(b01-b00)
        U[4][col] += 1
        V[2*h+1][col] += 1
        V[2*h][col] -= 1
        W[4*h][col] += 1
        W[4*h+2][col] += 1
        col += 1
        
        # Term 5: M5 = (a00+a01)b11
        U[0][col] += 1
        U[1][col] += 1
        V[8+2*h+1][col] += 1
        W[4*h][col] -= 1
        W[4*h+1][col] += 1
        col += 1
        
        # Term 6: M6 = (a10-a00)(b00+b01)
        U[3][col] += 1
        U[0][col] -= 1
        V[2*h][col] += 1
        V[2*h+1][col] += 1
        W[4*h+3][col] += 1
        col += 1
        
        # Term 7: M7 = (a01-a11)(b10+b11)
        U[1][col] += 1
        U[4][col] -= 1
        V[8+2*h][col] += 1
        V[8+2*h+1][col] += 1
        W[4*h][col] += 1
        col += 1
        
        # Outer product part for j=2, k=2h, 2h+1
        # U rows: 2 (0,2), 5 (1,2)
        # V rows: 16+2h (2,2h), 16+2h+1 (2,2h+1)
        # W rows: 4h (2h,0), 4h+1 (2h+1,0), 4h+2 (2h,1), 4h+3 (2h+1,1)
        
        # Term 8: N1 = a02*b20
        U[2][col] += 1
        V[16+2*h][col] += 1
        W[4*h][col] += 1
        col += 1
        
        # Term 9: N2 = a02*b21
        U[2][col] += 1
        V[16+2*h+1][col] += 1
        W[4*h+1][col] += 1
        col += 1
        
        # Term 10: N3 = a12*b20
        U[5][col] += 1
        V[16+2*h][col] += 1
        W[4*h+2][col] += 1
        col += 1
        
        # Term 11: N4 = a12*b21
        U[5][col] += 1
        V[16+2*h+1][col] += 1
        W[4*h+3][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
