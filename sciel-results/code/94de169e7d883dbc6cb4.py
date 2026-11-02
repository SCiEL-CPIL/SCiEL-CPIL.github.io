"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 8 blocks of 2 columns each
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 32 x rank
    W = [[0] * rank for _ in range(p * n)]  # 32 x rank
    
    for b in range(num_blocks):
        c0 = 2 * b      # first column in block
        c1 = 2 * b + 1  # second column in block
        base = 7 * b     # starting rank index for this block
        
        # P1 = (A11 - A00)(B[1,c0] + B[1,c1])
        # U: A11=1, A00=-1
        U[1*m+1][base+0] += 1
        U[0*m+0][base+0] += -1
        # V: B[1,c0]=1, B[1,c1]=1
        V[1*p+c0][base+0] += 1
        V[1*p+c1][base+0] += 1
        # W: C[0,c1]+=1, C[1,c1]+=1
        W[c1*n+0][base+0] += 1
        W[c1*n+1][base+0] += 1
        
        # P2 = (A00 + A10)B[1,c1]
        U[0*m+0][base+1] += 1
        U[1*m+0][base+1] += 1
        V[1*p+c1][base+1] += 1
        # W: C[0,c0]-=1, C[0,c1]+=1
        W[c0*n+0][base+1] += -1
        W[c1*n+0][base+1] += 1
        
        # P3 = (A01 + A11)(B[0,c0] - B[1,c0])
        U[0*m+1][base+2] += 1
        U[1*m+1][base+2] += 1
        V[0*p+c0][base+2] += 1
        V[1*p+c0][base+2] += -1
        # W: C[1,c0]+=1, C[1,c1]-=1
        W[c0*n+1][base+2] += 1
        W[c1*n+1][base+2] += -1
        
        # P4 = (A11 - A01)(B[0,c1] + B[1,c1])
        U[1*m+1][base+3] += 1
        U[0*m+1][base+3] += -1
        V[0*p+c1][base+3] += 1
        V[1*p+c1][base+3] += 1
        # W: C[0,c0]+=1, C[1,c0]+=1
        W[c0*n+0][base+3] += 1
        W[c0*n+1][base+3] += 1
        
        # P5 = (A00 + A11)(B[0,c0] + B[1,c1])
        U[0*m+0][base+4] += 1
        U[1*m+1][base+4] += 1
        V[0*p+c0][base+4] += 1
        V[1*p+c1][base+4] += 1
        # W: C[0,c0]+=1, C[1,c1]+=1
        W[c0*n+0][base+4] += 1
        W[c1*n+1][base+4] += 1
        
        # P6 = (A01 - A00)(B[1,c0] + B[1,c1])
        U[0*m+1][base+5] += 1
        U[0*m+0][base+5] += -1
        V[1*p+c0][base+5] += 1
        V[1*p+c1][base+5] += 1
        # W: C[1,c1]-=1
        W[c1*n+1][base+5] += -1
        
        # P7 = (A10 - A11)(B[0,c0] + B[0,c1])
        U[1*m+0][base+6] += 1
        U[1*m+1][base+6] += -1
        V[0*p+c0][base+6] += 1
        V[0*p+c1][base+6] += 1
        # W: C[0,c0]-=1
        W[c0*n+0][base+6] += -1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
