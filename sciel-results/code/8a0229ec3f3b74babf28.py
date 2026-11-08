"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def _strassen_2x2():
    """Return (U, V, W) for 2x2 x 2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A11(B21-B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A22(B12-B11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    rank_total = 65  # 49 (Strassen 4x4x4x4) + 16 (4x4x4x1)
    
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank_total for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank_total for _ in range(p * n)]  # 20 x 65
    
    U2, V2, W2 = _strassen_2x2()
    
    # Level-1 Strassen terms for 4x4 x 4x4
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    level1_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # S1
        ({2:1, 3:1}, {0:1}, {2:1}),               # S2
        ({0:1}, {2:1, 3:-1}, {1:1}),              # S3
        ({3:1}, {1:1, 0:-1}, {0:1, 2:1}),         # S4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),         # S5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),         # S6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}),    # S7
    ]
    
    col_offset = 0
    for li, (u_coeffs, v_coeffs, w_coeffs) in enumerate(level1_terms):
        for si in range(7):
            col = col_offset + si
            
            for a_block, coeff in u_coeffs.items():
                br, bc = a_block // 2, a_block % 2
                for di in range(2):
                    for dj in range(2):
                        a_idx = (2*br+di) * m + (2*bc+dj)
                        U[a_idx][col] += coeff * U2[di*2+dj][si]
            
            for b_block, coeff in v_coeffs.items():
                br, bc = b_block // 2, b_block % 2
                for dj in range(2):
                    for dk in range(2):
                        b_idx = (2*br+dj) * p + (2*bc+dk)
                        V[b_idx][col] += coeff * V2[dj*2+dk][si]
            
            for c_block, coeff in w_coeffs.items():
                br, bc = c_block // 2, c_block % 2
                for dk in range(2):
                    for di in range(2):
                        c_idx = (2*bc+dk) * n + (2*br+di)
                        W[c_idx][col] += coeff * W2[dk*2+di][si]
        
        col_offset += 7
    
    # 4x4 x 4x1 part
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
    
    col_offset += 16
    assert col_offset == rank_total
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
