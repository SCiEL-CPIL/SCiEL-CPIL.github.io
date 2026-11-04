"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of the rank-7 Strassen decomposition for 2x2.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen decomposition matrices
    # U2[i, j, c] where index = i*2+j, c in 0..6
    # V2[j, k, c] where index = j*2+k, c in 0..6
    # W2[k, i, c] where index = k*2+i, c in 0..6
    
    # Strassen's 7 products for 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    # C00 += M0, C11 += M0
    # W2 index: k*2+i
    # C00 (k=0, i=0) -> idx 0
    # C11 (k=1, i=1) -> idx 3
    W2[0][0] = 1; W2[3][0] = 1
    
    # M1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    # C10 += M1, C11 -= M1
    # C10 (k=0, i=1) -> idx 1
    # C11 (k=1, i=1) -> idx 3
    W2[1][1] = 1; W2[3][1] = -1
    
    # M2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    # C01 += M2, C11 += M2
    # C01 (k=1, i=0) -> idx 2
    # C11 (k=1, i=1) -> idx 3
    W2[2][2] = 1; W2[3][2] = 1
    
    # M3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    # C00 += M3, C10 += M3
    # C00 (k=0, i=0) -> idx 0
    # C10 (k=0, i=1) -> idx 1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    # C00 -= M4, C01 += M4
    # C00 (k=0, i=0) -> idx 0
    # C01 (k=1, i=0) -> idx 2
    W2[0][4] = -1; W2[2][4] = 1
    
    # M5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    # C11 += M5
    # C11 (k=1, i=1) -> idx 3
    W2[3][5] = 1
    
    # M6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    # C00 += M6
    # C00 (k=0, i=0) -> idx 0
    W2[0][6] = 1
    
    # Construct 4x4 matrices
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Tensor product expansion
    # U index: i*m + j. i = i1*2+i2, j = j1*2+j2
    # Rank index: c1*7 + c2
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            idx_u = i * m + j
            for c1 in range(7):
                u_val1 = U2[i1 * 2 + j1][c1]
                if u_val1 == 0:
                    continue
                for c2 in range(7):
                    u_val2 = U2[i2 * 2 + j2][c2]
                    if u_val2 == 0:
                        continue
                    c = c1 * 7 + c2
                    U[idx_u][c] = u_val1 * u_val2
                        
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            idx_v = j * p + k
            for c1 in range(7):
                v_val1 = V2[j1 * 2 + k1][c1]
                if v_val1 == 0:
                    continue
                for c2 in range(7):
                    v_val2 = V2[j2 * 2 + k2][c2]
                    if v_val2 == 0:
                        continue
                    c = c1 * 7 + c2
                    V[idx_v][c] = v_val1 * v_val2
                        
    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            idx_w = k * n + i
            for c1 in range(7):
                w_val1 = W2[k1 * 2 + i1][c1]
                if w_val1 == 0:
                    continue
                for c2 in range(7):
                    w_val2 = W2[k2 * 2 + i2][c2]
                    if w_val2 == 0:
                        continue
                    c = c1 * 7 + c2
                    W[idx_w][c] = w_val1 * w_val2
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
