import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Local indices:
    # A: 2x2 -> flat i*2+j
    # B: 2x2 -> flat j*2+k
    # C: 2x2 -> flat i*2+k (local), mapped to W convention k*n+i globally
    
    # Strassen components:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11
    
    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11
    
    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11
    
    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10
    
    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01
    
    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11
    
    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    # For each column-pair j (0..3) and row-pair s (0..1):
    #   A block: rows 0,1; cols 2s, 2s+1
    #   B block: rows 2s, 2s+1; cols 2j, 2j+1
    #   C block: rows 0,1; cols 2j, 2j+1
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # Map local A indices (i_loc*2+j_loc) to global (i_g*4+j_g)
            a_map = [
                0 * 4 + (2 * s + 0),   # i=0, j=2s
                0 * 4 + (2 * s + 1),   # i=0, j=2s+1
                1 * 4 + (2 * s + 0),   # i=1, j=2s
                1 * 4 + (2 * s + 1),   # i=1, j=2s+1
            ]
            
            # Map local B indices (j_loc*2+k_loc) to global (j_g*8+k_g)
            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),   # j_g=2s, k_g=2j
                (2 * s + 0) * 8 + (2 * j + 1),   # j_g=2s, k_g=2j+1
                (2 * s + 1) * 8 + (2 * j + 0),   # j_g=2s+1, k_g=2j
                (2 * s + 1) * 8 + (2 * j + 1),   # j_g=2s+1, k_g=2j+1
            ]
            
            # Map local C indices (i_loc*2+k_loc) to W convention (k_g*n + i_g)
            c_map = [
                (2 * j + 0) * 2 + 0,   # i_g=0, k_g=2j
                (2 * j + 1) * 2 + 0,   # i_g=0, k_g=2j+1
                (2 * j + 0) * 2 + 1,   # i_g=1, k_g=2j
                (2 * j + 1) * 2 + 1,   # i_g=1, k_g=2j+1
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val
                        
    if time.time() - start_time > deadline:
        pass  # Deterministic and fast; this guard is precautionary
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
