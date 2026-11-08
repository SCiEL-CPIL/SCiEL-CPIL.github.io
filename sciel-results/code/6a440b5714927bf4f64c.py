import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Decomposition strategy:
    # M_{2,3,9} = 3 * M_{2,3,3}
    # M_{2,3,3} = M_{2,2,2} + M_{2,2,1} + M_{2,1,3}
    # R(M_{2,2,2}) = 7 (Strassen)
    # R(M_{2,2,1}) = 2
    # R(M_{2,1,3}) = 6
    # Total per block = 15
    # Total rank = 45
    
    rank = 45
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # 3 blocks of M_{2,3,3}
    for block in range(3):
        k_off = block * 3
        
        # 1. M_{2,2,2} part: j in {0,1}, k in {k_off, k_off+1}
        # Strassen terms
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        
        # Local indices: i in {0,1}, j_local in {0,1}, k_local in {0,1}
        # Global j = j_local, k = k_off + k_local
        
        # m1
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k_off][col] = 1; V[1*p + k_off + 1][col] = 1
        W[k_off*2 + 0][col] = 1; W[(k_off+1)*2 + 1][col] = 1
        col += 1
        
        # m2
        U[1][col] = 1; U[3][col] = 1
        V[0*p + k_off][col] = 1
        W[k_off*2 + 1][col] = 1
        col += 1
        
        # m3
        U[0][col] = 1
        V[1*p + k_off + 1][col] = 1; V[1*p + k_off + 1][col] = -1 # Wait, b01 - b11
        # V index is j*p + k.
        # b01 is j=0, k=k_off+1. V[0*p + k_off+1]
        # b11 is j=1, k=k_off+1. V[1*p + k_off+1]
        V[0*p + k_off + 1][col] = 1
        V[1*p + k_off + 1][col] = -1
        W[k_off*2 + 0][col] = 1; W[(k_off+1)*2 + 1][col] = 1
        col += 1
        
        # m4
        U[3][col] = 1
        V[1*p + k_off][col] = 1; V[0*p + k_off][col] = -1
        W[k_off*2 + 0][col] = 1; W[(k_off+1)*2 + 1][col] = 1
        col += 1
        
        # m5
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k_off + 1][col] = 1
        W[k_off*2 + 0][col] = 1; W[(k_off+1)*2 + 1][col] = -1
        col += 1
        
        # m6
        U[1][col] = 1; U[0][col] = -1
        V[0*p + k_off][col] = 1; V[1*p + k_off][col] = 1
        W[k_off*2 + 1][col] = 1
        col += 1
        
        # m7
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k_off + 1][col] = 1; V[0*p + k_off + 1][col] = 1
        W[k_off*2 + 0][col] = 1; W[(k_off+1)*2 + 1][col] = -1
        col += 1
        
        # 2. M_{2,2,1} part: j in {0,1}, k = k_off+2
        # Rank 2
        # c0k = a00*b0k + a01*b1k
        # c1k = a10*b0k + a11*b1k
        # This is C[:, k] = A[:, :] * B[:, k]
        # Term 1: U[0]=1, U[1]=1? No.
        # We need sum_c U_{ij} V_{jk} W_{ki} = delta.
        # For M_{2,2,1}, k is fixed.
        # T_{ij, jk, ki} = 1 if i=i', j=j', k=k'.
        # We found rank 2.
        # c=0: U[0]=1, U[1]=1? 
        # Let's use the explicit solution for M_{2,2,1}.
        # M_{(i,j), i'} = delta_{ii'}.
        # c=0: U[0,0]=1, U[0,1]=1, U[1,0]=0, U[1,1]=0. V[0,k]=1, V[1,k]=1. W[k,0]=1, W[k,1]=0.
        # c=1: U[0,0]=0, U[0,1]=0, U[1,0]=1, U[1,1]=1. V[0,k]=1, V[1,k]=1. W[k,0]=0, W[k,1]=1.
        
        k_fix = k_off + 2
        # c=0
        U[0][col] = 1; U[1][col] = 1
        V[0*p + k_fix][col] = 1; V[1*p + k_fix][col] = 1
        W[k_fix*2 + 0][col] = 1
        col += 1
        
        # c=1
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k_fix][col] = 1; V[1*p + k_fix][col] = 1
        W[k_fix*2 + 1][col] = 1
        col += 1
        
        # 3. M_{2,1,3} part: j=2, k in {k_off, k_off+1, k_off+2}
        # Rank 6
        # T_{i,2, i',k'} = delta_{ii'} delta_{kk'}
        # c=0..5
        # c=0: i=0, k=k_off
        # c=1: i=0, k=k_off+1
        # c=2: i=0, k=k_off+2
        # c=3: i=1, k=k_off
        # c=4: i=1, k=k_off+1
        # c=5: i=1, k=k_off+2
        
        for i in range(2):
            for k_local in range(3):
                k = k_off + k_local
                U[i][col] = 1
                V[2*p + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
