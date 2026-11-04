import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strategy: Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x8) into B1 (2x8) and B2 (1x8)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 x 2x8, use Strassen on each pair of columns (4 pairs * rank 7 = 28)
    # A2*B2: 2x1 x 1x8, rank 2
    # Total rank: 30
    
    total_rank = 30
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 30
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 30
    
    # Strassen for 2x2 x 2x2 (local indices)
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Part 1: A1*B1 (2x2 x 2x8) using Strassen on column pairs
    col_offset = 0
    for pair in range(4):
        k_start = pair * 2
        for c in range(7):
            # U: A1[li, lj] at global index li*3 + lj
            for li in range(2):
                for lj in range(2):
                    u_val = strassen_U[li*2 + lj][c]
                    if u_val != 0:
                        U[li * m + lj][col_offset + c] += u_val
            
            # V: B1[lj, k_start+lk] at global index lj*8 + (k_start+lk)
            for lj in range(2):
                for lk in range(2):
                    v_val = strassen_V[lj*2 + lk][c]
                    if v_val != 0:
                        V[lj * p + (k_start + lk)][col_offset + c] += v_val
            
            # W: C[li, k_start+lk] at global index (k_start+lk)*2 + li
            for lk in range(2):
                for li in range(2):
                    w_val = strassen_W[lk*2 + li][c]
                    if w_val != 0:
                        W[(k_start + lk) * n + li][col_offset + c] += w_val
        
        col_offset += 7
    
    # Part 2: A2*B2 (2x1 x 1x8), rank 2
    # A2[i,0] at global U index i*3 + 2
    # B2[0,k] at global V index 2*8 + k = 16 + k
    # C[i,k] at global W index k*2 + i
    
    # Term 0: handles i=0
    for i in range(2):
        U[i * m + 2][col_offset] = 1 if i == 0 else 0
        U[i * m + 2][col_offset + 1] = 1 if i == 1 else 0
    
    for k in range(8):
        V[2 * p + k][col_offset] = 1
        V[2 * p + k][col_offset + 1] = 1
    
    for k in range(8):
        for i in range(2):
            c = k * n + i
            W[c][col_offset] = 1 if c % 2 == 0 else 0
            W[c][col_offset + 1] = 1 if c % 2 == 1 else 0
    
    return {"U": U, "V": V, "W": W}
