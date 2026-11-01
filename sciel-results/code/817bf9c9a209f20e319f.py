import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = 2, 2, 12
    rank = 42  # 6 blocks * 7 (Strassen)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # L_r(A) = sum U_ij,r A_ij
    # Q_r(B) = sum V_jk,r B_jk
    # C_ik = sum W_ki,r L_r(A) Q_r(B)
    
    # Local indices for 2x2:
    # A: (0,0), (0,1), (1,0), (1,1) -> U indices 0, 1, 2, 3
    # B: (0,0), (0,1), (1,0), (1,1) -> V indices 0, 1, 2, 3
    # C: (0,0), (0,1), (1,0), (1,1) -> W indices 0, 1, 2, 3 (k*n+i)
    # k=0, i=0 -> 0
    # k=1, i=0 -> 1
    # k=0, i=1 -> 2
    # k=1, i=1 -> 3
    
    # Coefficients:
    # M1: L=a+d, Q=e+h. W: C00+=1, C11+=1
    # M2: L=c+d, Q=e.   W: C10+=1, C11-=1
    # M3: L=a,   Q=f-h. W: C01+=1, C11+=1
    # M4: L=d,   Q=g-e. W: C00+=1, C10+=1
    # M5: L=a+b, Q=h.   W: C00-=1, C01+=1
    # M6: L=c-a, Q=e+f. W: C11+=1
    # M7: L=b-d, Q=g+h. W: C00+=1
    
    # U_coefs[r][i][j]
    U_local = [
        [[1, 0], [0, 1]], # M1: a+d
        [[0, 0], [1, 1]], # M2: c+d
        [[1, 0], [0, 0]], # M3: a
        [[0, 0], [0, 1]], # M4: d
        [[1, 1], [0, 0]], # M5: a+b
        [[-1, 0], [1, 0]],# M6: c-a
        [[0, 1], [0, -1]] # M7: b-d
    ]
    
    # V_coefs[r][j][k]
    V_local = [
        [[1, 0], [0, 1]], # M1: e+h
        [[1, 0], [0, 0]], # M2: e
        [[0, 1], [0, -1]],# M3: f-h
        [[-1, 0], [1, 0]],# M4: g-e
        [[0, 0], [0, 1]], # M5: h
        [[1, 1], [0, 0]], # M6: e+f
        [[0, 0], [1, 1]]  # M7: g+h
    ]
    
    # W_coefs[r][k][i]
    # W index is k*n + i = 2k + i
    # C00 (k=0,i=0) idx 0
    # C01 (k=1,i=0) idx 1
    # C10 (k=0,i=1) idx 2
    # C11 (k=1,i=1) idx 3
    W_local = [
        [1, 0, 0, 1], # M1: C00+1, C11+1
        [0, 0, 1, -1],# M2: C10+1, C11-1
        [0, 1, 0, 1], # M3: C01+1, C11+1
        [1, 0, 1, 0], # M4: C00+1, C10+1
        [-1, 1, 0, 0],# M5: C00-1, C01+1
        [0, 0, 0, 1], # M6: C11+1
        [1, 0, 0, 0]  # M7: C00+1
    ]
    
    num_blocks = 6
    for b in range(num_blocks):
        for r in range(7):
            global_r = b * 7 + r
            
            # Update U
            for i in range(2):
                for j in range(2):
                    val = U_local[r][i][j]
                    if val != 0:
                        global_j = b * 2 + j
                        U[i * 2 + global_j][global_r] += val
                        
            # Update V
            for j in range(2):
                for k in range(2):
                    val = V_local[r][j][k]
                    if val != 0:
                        global_k = b * 2 + k
                        V[j * 12 + global_k][global_r] += val
                        
            # Update W
            for k in range(2):
                for i in range(2):
                    val = W_local[r][k * 2 + i]
                    if val != 0:
                        global_k = b * 2 + k
                        W[global_k * 2 + i][global_r] += val
                        
    return {"U": U, "V": V, "W": W}
