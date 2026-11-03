import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    num_blocks = p // 2   # 3 blocks
    
    rank = num_blocks * (7 + 4)  # 3 * 11 = 33
    
    U = np.zeros((n * m, rank), dtype=float)  # (6, 33)
    V = np.zeros((m * p, rank), dtype=float)  # (18, 33)
    W = np.zeros((p * n, rank), dtype=float)  # (12, 33)
    
    # A indices: A[i,j] -> i*m+j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    for t in range(num_blocks):
        base = t * 11
        # B^t: columns 2t and 2t+1
        # B[j, 2t] -> j*p + 2t, B[j, 2t+1] -> j*p + 2t+1
        # B[0,2t]=2t, B[0,2t+1]=2t+1
        # B[1,2t]=6+2t, B[1,2t+1]=7+2t
        # B[2,2t]=12+2t, B[2,2t+1]=13+2t
        
        # C^t output in W: W[k*n+i]
        # C[0,2t] -> W[4t], C[0,2t+1] -> W[4t+2]
        # C[1,2t] -> W[4t+1], C[1,2t+1] -> W[4t+3]
        C11_w = 4 * t
        C12_w = 4 * t + 2
        C21_w = 4 * t + 1
        C22_w = 4 * t + 3
        
        # Part 1: Strassen on A[0:2, 0:2] * B[0:2, 2t:2t+2]
        # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=3, d=A[1,1]=4
        # e=B[0,2t]=2t, f=B[0,2t+1]=2t+1, g=B[1,2t]=6+2t, h=B[1,2t+1]=7+2t
        
        a, b, c, d = 0, 1, 3, 4
        e, f, g, h = 2*t, 2*t+1, 6+2*t, 7+2*t
        
        # M1 = (a+d)(e+h)
        U[a, base+0] = 1.0; U[d, base+0] = 1.0
        V[e, base+0] = 1.0; V[h, base+0] = 1.0
        W[C11_w, base+0] = 1.0; W[C22_w, base+0] = 1.0
        
        # M2 = (c+d)e
        U[c, base+1] = 1.0; U[d, base+1] = 1.0
        V[e, base+1] = 1.0
        W[C21_w, base+1] = 1.0; W[C22_w, base+1] = -1.0
        
        # M3 = a(f-h)
        U[a, base+2] = 1.0
        V[f, base+2] = 1.0; V[h, base+2] = -1.0
        W[C12_w, base+2] = 1.0; W[C22_w, base+2] = 1.0
        
        # M4 = d(g-f)
        U[d, base+3] = 1.0
        V[g, base+3] = 1.0; V[f, base+3] = -1.0
        W[C11_w, base+3] = 1.0; W[C21_w, base+3] = 1.0
        
        # M5 = (a+b)h
        U[a, base+4] = 1.0; U[b, base+4] = 1.0
        V[h, base+4] = 1.0
        W[C11_w, base+4] = -1.0; W[C12_w, base+4] = 1.0
        
        # M6 = (c-d)(g+h)
        U[c, base+5] = 1.0; U[d, base+5] = -1.0
        V[g, base+5] = 1.0; V[h, base+5] = 1.0
        W[C22_w, base+5] = 1.0
        
        # M7 = (a-b)(e+f)
        U[a, base+6] = 1.0; U[b, base+6] = -1.0
        V[e, base+6] = 1.0; V[f, base+6] = 1.0
        W[C11_w, base+6] = 1.0
        
        # Part 2: A[:,2] * B[2, 2t:2t+2] (2x1x2 outer product)
        # A[0,2]=2, A[1,2]=5
        # B[2,2t]=12+2t, B[2,2t+1]=13+2t
        
        # C[0,2t] += A[0,2]*B[2,2t]
        U[2, base+7] = 1.0
        V[12+2*t, base+7] = 1.0
        W[C11_w, base+7] = 1.0
        
        # C[0,2t+1] += A[0,2]*B[2,2t+1]
        U[2, base+8] = 1.0
        V[13+2*t, base+8] = 1.0
        W[C12_w, base+8] = 1.0
        
        # C[1,2t] += A[1,2]*B[2,2t]
        U[5, base+9] = 1.0
        V[12+2*t, base+9] = 1.0
        W[C21_w, base+9] = 1.0
        
        # C[1,2t+1] += A[1,2]*B[2,2t+1]
        U[5, base+10] = 1.0
        V[13+2*t, base+10] = 1.0
        W[C22_w, base+10] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
