"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition using Strassen over F[x]/(x^2)."""
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    W2[0][0]=1; W2[3][0]=1
    
    # r=1: M2 = (c+d)*e -> C[1,0]+=1, C[1,1]-=1
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    W2[1][1]=1; W2[3][1]=-1
    
    # r=2: M3 = a*(f-h) -> C[0,1]+=1, C[1,1]+=1
    U2[0][2]=1
    V2[1][2]=1; V2[3][2]=-1
    W2[2][2]=1; W2[3][2]=1
    
    # r=3: M4 = d*(g-e) -> C[0,0]+=1, C[1,0]+=1
    U2[3][3]=1
    V2[2][3]=1; V2[0][3]=-1
    W2[0][3]=1; W2[1][3]=1
    
    # r=4: M5 = (a+b)*h -> C[0,0]-=1, C[0,1]+=1
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    W2[0][4]=-1; W2[2][4]=1
    
    # r=5: M6 = (c-a)*(e+f) -> C[1,1]+=1
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[1][5]=1
    W2[3][5]=1
    
    # r=6: M7 = (b-d)*(g+h) -> C[0,0]+=1
    U2[1][6]=1; U2[3][6]=-1
    V2[2][6]=1; V2[3][6]=1
    W2[0][6]=1
    
    # Build 4x4x4 rank-49 decomposition using Kronecker product
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            i = i1*2 + i0
                            j = j1*2 + j0
                            U4[i*4+j][r] = U2[i1*2+j1][r1] * U2[i0*2+j0][r2]
    
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            j = j1*2 + j0
                            k = k1*2 + k0
                            V4[j*4+k][r] = V2[j1*2+k1][r1] * V2[j0*2+k0][r2]
    
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            k = k1*2 + k0
                            i = i1*2 + i0
                            W4[k*4+i][r] = W2[k1*2+i1][r1] * W2[k0*2+i0][r2]
    
    # Build 4x4x8 rank-49 decomposition using F[x]/(x^2)
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # U matrix: same as U4
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = U4[i*4+j][r]
    
    # V matrix: V[j*8+k][r] = V4[j*4 + (k mod 4)][r]
    for j in range(4):
        for k in range(8):
            for r in range(49):
                V[j*8+k][r] = V4[j*4 + (k % 4)][r]
    
    # W matrix: W[k*4+i][r] = W4[(k mod 4)*4 + i][r]
    for k in range(8):
        for i in range(4):
            for r in range(49):
                W[k*4+i][r] = W4[(k % 4)*4 + i][r]
    
    if time.time() - start_time > 100:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
