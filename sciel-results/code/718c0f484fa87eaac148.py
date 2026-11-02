"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen's algorithm on 4 blocks of 2x2
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U rows: 0:(1,0), 1:(1,0), 2:(0,1), 3:(0,1) -> Actually U[i*m+j]
    # U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
    # V rows: j*p+k
    # W rows: k*n+i
    
    # Strassen products for one block:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    for b in range(4):
        base_col = b * 7
        # Block b uses B columns 2b, 2b+1
        # V indices: 0*8+2b, 0*8+2b+1, 1*8+2b, 1*8+2b+1
        v00 = 2*b
        v01 = 2*b+1
        v10 = 8+2*b
        v11 = 8+2*b+1
        
        # W indices: 2b*2+0, 2b*2+1, (2b+1)*2+0, (2b+1)*2+1
        w00 = 4*b
        w10 = 4*b+1
        w01 = 4*b+2
        w11 = 4*b+3
        
        # P1
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[v00][base_col] = 1
        V[v11][base_col] = 1
        W[w00][base_col] = 1
        W[w11][base_col] = 1
        
        # P2
        U[2][base_col+1] = 1
        U[3][base_col+1] = 1
        V[v00][base_col+1] = 1
        W[w10][base_col+1] = 1
        W[w11][base_col+1] = -1
        
        # P3
        U[0][base_col+2] = 1
        V[v01][base_col+2] = 1
        V[v11][base_col+2] = -1
        W[w01][base_col+2] = 1
        W[w11][base_col+2] = 1
        
        # P4
        U[3][base_col+3] = 1
        V[v10][base_col+3] = 1
        V[v00][base_col+3] = -1
        W[w00][base_col+3] = 1
        W[w10][base_col+3] = 1
        
        # P5
        U[0][base_col+4] = 1
        U[1][base_col+4] = 1
        V[v11][base_col+4] = 1
        W[w00][base_col+4] = -1
        W[w01][base_col+4] = 1
        
        # P6
        U[2][base_col+5] = 1
        U[0][base_col+5] = -1
        V[v00][base_col+5] = 1
        V[v01][base_col+5] = 1
        W[w11][base_col+5] = 1
        
        # P7
        U[1][base_col+6] = 1
        U[3][base_col+6] = -1
        V[v10][base_col+6] = 1
        V[v11][base_col+6] = 1
        W[w00][base_col+6] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
