"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using Strassen's algorithm for each 2x2 block of the 2x16 matrix
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(p // 2):
        base = 7 * b
        # M1 = (a00 + a11)(b00 + b11)
        U[0][base] += 1
        U[3][base] += 1
        V[2*b][base] += 1
        V[16+2*b+1][base] += 1
        W[4*b][base] += 1
        W[4*b+3][base] += 1
        
        # M2 = (a10 + a11)b00
        U[2][base+1] += 1
        U[3][base+1] += 1
        V[2*b][base+1] += 1
        W[4*b+1][base+1] += 1
        W[4*b+3][base+1] -= 1
        
        # M3 = a00(b01 - b11)
        U[0][base+2] += 1
        V[2*b+1][base+2] += 1
        V[16+2*b+1][base+2] -= 1
        W[4*b+2][base+2] += 1
        W[4*b+3][base+2] += 1
        
        # M4 = a11(b10 - b00)
        U[3][base+3] += 1
        V[16+2*b][base+3] += 1
        V[2*b][base+3] -= 1
        W[4*b][base+3] += 1
        W[4*b+1][base+3] += 1
        
        # M5 = (a00 + a01)b11
        U[0][base+4] += 1
        U[1][base+4] += 1
        V[16+2*b+1][base+4] += 1
        W[4*b][base+4] -= 1
        W[4*b+2][base+4] += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[2][base+5] += 1
        U[0][base+5] -= 1
        V[2*b][base+5] += 1
        V[2*b+1][base+5] += 1
        W[4*b+3][base+5] += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[1][base+6] += 1
        U[3][base+6] -= 1
        V[16+2*b][base+6] += 1
        V[16+2*b+1][base+6] += 1
        W[4*b][base+6] += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
