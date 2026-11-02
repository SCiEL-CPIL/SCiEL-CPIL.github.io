import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    # Strassen 2x2x2 factors
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    strassen_U = [
        [1, 0, 1, 0],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: i=1, l=1
        [0, 0, 1, -1],  # M2: k=1, l=-1
        [0, 1, 0, 1],   # M3: j=1, l=1
        [1, 0, 1, 0],   # M4: i=1, k=1
        [-1, 1, 0, 0],  # M5: i=-1, j=1
        [0, 0, 0, 1],   # M6: l=1
        [1, 0, 0, 1],   # M7: i=1, l=1
    ]
    
    for a in range(2):
        for b in range(2):
            if time.time() > deadline:
                break
            
            # Strassen: A[0:2, 2a:2a+2] x B[2a:2a+2, 2b:2b+2] -> C[0:2, 2b:2b+2]
            for s in range(7):
                U[0 * m + 2*a][col] = strassen_U[s][0]
                U[0 * m + 2*a+1][col] = strassen_U[s][1]
                U[1 * m + 2*a][col] = strassen_U[s][2]
                U[1 * m + 2*a+1][col] = strassen_U[s][3]
                
                V[2*a * p + 2*b][col] = strassen_V[s][0]
                V[2*a * p + 2*b+1][col] = strassen_V[s][1]
                V[(2*a+1) * p + 2*b][col] = strassen_V[s][2]
                V[(2*a+1) * p + 2*b+1][col] = strassen_V[s][3]
                
                W[2*b * n + 0][col] = strassen_W[s][0]
                W[(2*b+1) * n + 0][col] = strassen_W[s][1]
                W[2*b * n + 1][col] = strassen_W[s][2]
                W[(2*b+1) * n + 1][col] = strassen_W[s][3]
                
                col += 1
            
            # Trivial: A[2, 2a:2a+2] x B[2a:2a+2, 2b:2b+2] -> C[2, 2b:2b+2]
            for jj in range(2):
                for kk in range(2):
                    U[2 * m + 2*a + jj][col] = 1
                    V[(2*a + jj) * p + 2*b + kk][col] = 1
                    W[(2*b + kk) * n + 2][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
