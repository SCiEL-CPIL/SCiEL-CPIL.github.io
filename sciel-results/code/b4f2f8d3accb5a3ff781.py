"""Matrix multiplication 3,4,8 rank-88 decomposition with deadline guard."""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # (12, 88)
    V = [[0] * rank for _ in range(m * p)]  # (32, 88)
    W = [[0] * rank for _ in range(p * n)]  # (24, 88)
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8 -> 32 terms
    for j in range(4):
        for k in range(8):
            U[j][col] = 1          # row i*m+j = 0*4+j = j
            V[j * p + k][col] = 1  # row j*p+k
            W[k * n][col] = 1      # row k*n+0
            col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 2: i=1,2 using Strassen 2x2x2 blocks
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # 8 blocks x 7 Strassen terms = 56 terms
    
    # Strassen 2x2x2 factor matrices
    # Local A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10-B11
        [-1, 0, 1, 0],  # M4: B00-B10
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 1],   # M3: C01+=1, C11+=1
        [1, 0, 1, 0],   # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 0, 1],   # M6: C11+=1
        [1, 0, 0, 0]    # M7: C00+=1
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U: local l -> global i=1+l//2, j=j_start+l%2
                for l in range(4):
                    val = strassen_U[t][l]
                    if val != 0:
                        gi = 1 + l // 2
                        gj = j_start + l % 2
                        U[gi * m + gj][col] += val
                
                # V: local l -> global j=j_start+l//2, k=k_start+l%2
                for l in range(4):
                    val = strassen_V[t][l]
                    if val != 0:
                        gj = j_start + l // 2
                        gk = k_start + l % 2
                        V[gj * p + gk][col] += val
                
                # W: local l -> global k=k_start+l%2, i=1+l//2
                for l in range(4):
                    val = strassen_W[t][l]
                    if val != 0:
                        gi = 1 + l // 2
                        gk = k_start + l % 2
                        W[gk * n + gi][col] += val
                
                col += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
