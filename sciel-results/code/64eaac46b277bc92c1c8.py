"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55
    
    col = 0
    for g in range(5):
        c0 = 2 * g
        c1 = 2 * g + 1
        
        # p1 = (A[0,0]+A[1,1])*(B[0,c0]+B[1,c1])
        U[0][col] = 1
        U[4][col] = 1
        V[0*10+c0][col] = 1
        V[1*10+c1][col] = 1
        W[c0*2+0][col] = 1   # C[0,c0] += p1
        W[c1*2+0][col] = 0   # C[0,c1]
        W[c0*2+1][col] = 0   # C[1,c0]
        W[c1*2+1][col] = 1   # C[1,c1] += p1
        col += 1
        
        # p2 = (A[1,0]+A[1,1])*B[0,c0]
        U[3][col] = 1   # A[1,0]
        U[4][col] = 1   # A[1,1]
        V[0*10+c0][col] = 1
        W[c0*2+1][col] = 1   # C[1,c0] += p2
        W[c1*2+1][col] = 1   # C[1,c1] += -p2? No, C[1,c1] -= p2
        # Wait, C[1,c1] = p1 - p2 + p3 + p6, so p2 contributes -1 to C[1,c1]
        W[c1*2+1][col] = -1
        col += 1
        
        # p3 = A[0,0]*(B[1,c0]-B[1,c1])
        U[0][col] = 1
        V[1*10+c0][col] = 1
        V[1*10+c1][col] = -1
        W[c0*2+0][col] = 0   # C[0,c0]
        W[c1*2+0][col] = 1   # C[0,c1] += p3
        W[c0*2+1][col] = 0
        W[c1*2+1][col] = 1   # C[1,c1] += p3
        col += 1
        
        # p4 = A[1,1]*(B[0,c1]-B[0,c0])
        U[4][col] = 1
        V[0*10+c1][col] = 1
        V[0*10+c0][col] = -1
        W[c0*2+0][col] = 1   # C[0,c0] += p4
        W[c1*2+0][col] = 0
        W[c0*2+1][col] = 1   # C[1,c0] += p4
        W[c1*2+1][col] = 0
        col += 1
        
        # p5 = (A[0,0]+A[0,1])*B[1,c1]
        U[0][col] = 1
        U[1][col] = 1
        V[1*10+c1][col] = 1
        W[c0*2+0][col] = -1  # C[0,c0] -= p5
        W[c1*2+0][col] = 1   # C[0,c1] += p5
        W[c0*2+1][col] = 0
        W[c1*2+1][col] = 0
        col += 1
        
        # p6 = (A[1,0]-A[0,0])*(B[0,c0]+B[0,c1])
        U[3][col] = 1
        U[0][col] = -1
        V[0*10+c0][col] = 1
        V[0*10+c1][col] = 1
        W[c0*2+0][col] = 0
        W[c1*2+0][col] = 0
        W[c0*2+1][col] = 0
        W[c1*2+1][col] = 1   # C[1,c1] += p6
        col += 1
        
        # p7 = (A[0,1]-A[1,1])*(B[1,c0]+B[1,c1])
        U[1][col] = 1
        U[4][col] = -1
        V[1*10+c0][col] = 1
        V[1*10+c1][col] = 1
        W[c0*2+0][col] = 1   # C[0,c0] += p7
        W[c1*2+0][col] = 0
        W[c0*2+1][col] = 0
        W[c1*2+1][col] = 0
        col += 1
        
        # j=2 terms
        U[2][col] = 1
        V[2*10+c0][col] = 1
        W[c0*2+0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[2*10+c1][col] = 1
        W[c1*2+0][col] = 1
        col += 1
        
        U[5][col] = 1
        V[2*10+c0][col] = 1
        W[c0*2+1][col] = 1
        col += 1
        
        U[5][col] = 1
        V[2*10+c1][col] = 1
        W[c1*2+1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
