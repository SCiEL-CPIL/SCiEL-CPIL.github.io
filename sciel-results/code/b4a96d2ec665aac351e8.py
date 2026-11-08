import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 88
    
    # Initialize factor matrices with integers
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 1x4x8 block (i=0)
    # Rank is 1*4*8 = 32
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    if time.time() - start_time > 140:
        return {"U": U, "V": V, "W": W}

    # Part 2: 2x4x8 block (i=1,2)
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # Each 2x2x2 block uses Strassen's rank-7 algorithm
    
    # Strassen factors for 2x2x2
    # Local indices 0..3 for 2x2 matrices
    # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for jb in range(2):
        j0 = jb * 2
        for kb in range(4):
            k0 = kb * 2
            
            for t in range(7):
                # U: 2x2 local A -> global 2x4 A (rows 1,2)
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = j0 + lj
                        U[gi * m + gj][col] += strassen_U[t][l]
                        
                # V: 2x2 local B -> global 4x8 B
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] += strassen_V[t][l]
                        
                # W: 2x2 local C -> global 8x2 C (cols 1,2 of C)
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = k0 + lk
                        W[gk * n + gi][col] += strassen_W[t][l]
                        
                col += 1
                if time.time() - start_time > 140:
                    return {"U": U, "V": V, "W": W}
            if time.time() - start_time > 140:
                return {"U": U, "V": V, "W": W}
        if time.time() - start_time > 140:
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
