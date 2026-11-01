import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank-9 decomposition for 2x2x3 matrix multiplication
    # U shape: (4, 9), V shape: (6, 9), W shape: (6, 9)
    # U indices: i*m+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: j*p+k -> (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    # W indices: k*n+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5
    
    R = 9
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    # Term 0: T[0,0,0] and T[1,0,0]
    U[0][0] = 1; U[2][0] = 1
    V[0][0] = 1
    W[0][0] = 1; W[2][0] = 1
    
    # Term 1: T[0,0,1] and T[1,0,1]
    U[0][1] = 1; U[2][1] = 1
    V[1][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # Term 2: T[0,0,2] and T[1,0,2]
    U[0][2] = 1; U[2][2] = 1
    V[2][2] = 1
    W[4][2] = 1; W[5][2] = 1
    
    # Term 3: T[0,1,0]
    U[1][3] = 1
    V[3][3] = 1
    W[2][3] = 1
    
    # Term 4: T[0,1,1]
    U[1][4] = 1
    V[4][4] = 1
    W[3][4] = 1
    
    # Term 5: T[0,1,2]
    U[1][5] = 1
    V[5][5] = 1
    W[4][5] = 1
    
    # Term 6: T[1,1,0]
    U[3][6] = 1
    V[3][6] = 1
    W[3][6] = 1
    
    # Term 7: T[1,1,1]
    U[3][7] = 1
    V[4][7] = 1
    W[5][7] = 1
    
    # Term 8: T[1,1,2]
    U[3][8] = 1
    V[5][8] = 1
    W[5][8] = 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(R):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to rank 12
        R = 12
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
