import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
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
    
    # Helper to add a 2x2x2 block using Strassen
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        for t in range(7):
            for l in range(4):
                if strassen_U[t][l] != 0:
                    li = l // 2
                    lj = l % 2
                    gi = i_off + li
                    gj = j_off + lj
                    U[gi * m + gj][col] += strassen_U[t][l]
                    
                if strassen_V[t][l] != 0:
                    lj = l // 2
                    lk = l % 2
                    gj = j_off + lj
                    gk = k_off + lk
                    V[gj * p + gk][col] += strassen_V[t][l]
                    
                if strassen_W[t][l] != 0:
                    li = l // 2
                    lk = l % 2
                    gi = i_off + li
                    gk = k_off + lk
                    W[gk * n + gi][col] += strassen_W[t][l]
            col += 1
            if time.time() - start_time > 140:
                return False
        return True

    # Helper to add a 1x2x4 block (trivial rank 8)
    def add_1x2x4_block(i_off, j_off, k_off):
        nonlocal col
        # i is fixed at i_off (only 1 row)
        # j is 2 rows: j_off, j_off+1
        # k is 4 cols: k_off .. k_off+3
        for kk in range(4):
            gk = k_off + kk
            # Term 1: A[i_off, j_off] * B[j_off, gk]
            U[i_off * m + j_off][col] = 1
            V[j_off * p + gk][col] = 1
            W[gk * n + i_off][col] = 1
            col += 1
            if time.time() - start_time > 140:
                return False
            
            # Term 2: A[i_off, j_off+1] * B[j_off+1, gk]
            U[i_off * m + j_off + 1][col] = 1
            V[(j_off + 1) * p + gk][col] = 1
            W[gk * n + i_off][col] = 1
            col += 1
            if time.time() - start_time > 140:
                return False
        return True

    # Block 1: i=0, j=0,1, k=0..3
    if not add_1x2x4_block(0, 0, 0):
        return {"U": U, "V": V, "W": W}
        
    # Block 2: i=0, j=0,1, k=4..7
    if not add_1x2x4_block(0, 0, 4):
        return {"U": U, "V": V, "W": W}
        
    # Block 3: i=1,2, j=0,1, k=0..3
    # Split k into 0,1 and 2,3
    if not add_strassen_block(1, 0, 0):
        return {"U": U, "V": V, "W": W}
    if not add_strassen_block(1, 0, 2):
        return {"U": U, "V": V, "W": W}
        
    # Block 4: i=1,2, j=0,1, k=4..7
    # Split k into 4,5 and 6,7
    if not add_strassen_block(1, 0, 4):
        return {"U": U, "V": V, "W": W}
    if not add_strassen_block(1, 0, 6):
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
