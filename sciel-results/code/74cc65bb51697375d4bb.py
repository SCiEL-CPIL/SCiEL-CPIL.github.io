import numpy as np

def construct():
    n, m, p = 4, 4, 4
    
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,0,1,1], [0,0,1,0]),
        ([0,0,0,1], [0,1,0,1], [-1,0,0,1]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([0,1,1,0], [1,0,1,0], [0,1,0,-1]),
        ([0,1,0,1], [0,1,1,0], [0,0,0,1]),
    ]
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for a in range(7):
        ua, va, wa = S[a]
        for b in range(7):
            ub, vb, wb = S[b]
            for i1 in range(2):
                for j1 in range(2):
                    a_coeff = ua[i1*2+j1]
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2*i1+i2
                            j = 2*j1+j2
                            U[i*4+j][col] = a_coeff * ub[i2*2+j2]
                            
            for j1 in range(2):
                for k1 in range(2):
                    b_coeff = va[j1*2+k1]
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2*j1+j2
                            k = 2*k1+k2
                            V[j*4+k][col] = b_coeff * vb[j2*2+k2]
                            
            for k1 in range(2):
                for i1 in range(2):
                    c_coeff = wa[k1*2+i1]
                    for k2 in range(2):
                        for i2 in range(2):
                            k = 2*k1+k2
                            i = 2*i1+i2
                            W[k*4+i][col] = c_coeff * wb[k2*2+i2]
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
