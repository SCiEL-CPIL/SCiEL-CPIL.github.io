"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(4):
        base = b * 7
        # M1 = (A00+A11)*(B[0,2b]+B[1,2b+1])
        U[0][base+0] += 1
        U[3][base+0] += 1
        V[2*b][base+0] += 1
        V[9+2*b][base+0] += 1
        W[4*b][base+0] += 1
        W[4*b+3][base+0] += 1
        
        # M2 = (A10+A11)*B[0,2b]
        U[2][base+1] += 1
        U[3][base+1] += 1
        V[2*b][base+1] += 1
        W[4*b+1][base+1] += 1
        
        # M3 = A00*(B[0,2b+1]-B[1,2b+1])
        U[0][base+2] += 1
        V[2*b+1][base+2] += 1
        V[9+2*b][base+2] -= 1
        W[4*b+2][base+2] += 1
        
        # M4 = A11*(B[1,2b]-B[0,2b])
        U[3][base+3] += 1
        V[8+2*b][base+3] += 1
        V[2*b][base+3] -= 1
        W[4*b][base+3] += 1
        W[4*b+1][base+3] += 1
        
        # M5 = (A00+A01)*B[1,2b+1]
        U[0][base+4] += 1
        U[1][base+4] += 1
        V[9+2*b][base+4] += 1
        W[4*b][base+4] -= 1
        W[4*b+2][base+4] += 1
        
        # M6 = (A10-A00)*(B[0,2b]+B[1,2b])
        U[2][base+5] += 1
        U[0][base+5] -= 1
        V[2*b][base+5] += 1
        V[8+2*b][base+5] += 1
        W[4*b+3][base+5] += 1
        
        # M7 = (A01-A11)*(B[0,2b+1]+B[1,2b+1])
        U[1][base+6] += 1
        U[3][base+6] -= 1
        V[2*b+1][base+6] += 1
        V[9+2*b][base+6] += 1
        W[4*b][base+6] += 1
        W[4*b+3][base+6] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
