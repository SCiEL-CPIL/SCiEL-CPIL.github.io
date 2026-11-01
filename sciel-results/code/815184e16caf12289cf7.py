"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 120.0

def construct():
    # Strassen-based decomposition: apply rank-7 2x2 matmul to each 2x2 block of the 2x10 matrix.
    # Total rank = 5 * 7 = 35 (vs schoolbook 40)
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2 columns each
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # W-side contributions per multiplication:
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 -= 1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C01 += 1, C00 -= 1
    # M6: C11 += 1
    # M7: C00 += 1
    #
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    # Within block b (columns 2b, 2b+1 of B and C):
    #   v_koff 0,1,2,3 -> B[0][2b], B[0][2b+1], B[1][2b], B[1][2b+1]
    #   w_koff 0,1,2,3 -> C[0][2b], C[0][2b+1], C[1][2b], C[1][2b+1]
    
    strassen_decomp = [
        # M1 = (A00+A11) * (B00+B11)
        (
            [(0, 1), (3, 1)],       # U: A[0][0] + A[1][1]
            [(0, 1), (3, 1)],       # V: B[0][2b] + B[1][2b+1]
            [(0, 1), (3, 1)],       # W: C[0][2b]:1, C[1][2b+1]:1
        ),
        # M2 = (A10+A11) * B00
        (
            [(2, 1), (3, 1)],       # U: A[1][0] + A[1][1]
            [(0, 1)],               # V: B[0][2b]
            [(2, 1), (3, -1)],      # W: C[1][2b]:1, C[1][2b+1]:-1
        ),
        # M3 = A00 * (B01-B11)
        (
            [(0, 1)],               # U: A[0][0]
            [(1, 1), (3, -1)],      # V: B[0][2b+1] - B[1][2b+1]
            [(1, 1), (3, 1)],       # W: C[0][2b+1]:1, C[1][2b+1]:1
        ),
        # M4 = A11 * (B10-B00)
        (
            [(3, 1)],               # U: A[1][1]
            [(2, 1), (0, -1)],      # V: B[1][2b] - B[0][2b]
            [(0, 1), (2, 1)],       # W: C[0][2b]:1, C[1][2b]:1
        ),
        # M5 = (A00+A01) * B11
        (
            [(0, 1), (1, 1)],       # U: A[0][0] + A[0][1]
            [(3, 1)],               # V: B[1][2b+1]
            [(1, 1), (0, -1)],      # W: C[0][2b+1]:1, C[0][2b]:-1
        ),
        # M6 = (A10-A00) * (B00+B01)
        (
            [(2, 1), (0, -1)],      # U: A[1][0] - A[0][0]
            [(0, 1), (1, 1)],       # V: B[0][2b] + B[0][2b+1]
            [(3, 1)],               # W: C[1][2b+1]:1
        ),
        # M7 = (A01-A11) * (B10+B11)
        (
            [(1, 1), (3, -1)],      # U: A[0][1] - A[1][1]
            [(2, 1), (3, 1)],       # V: B[1][2b] + B[1][2b+1]
            [(0, 1)],               # W: C[0][2b]:1
        ),
    ]
    
    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_decomp[t]
            
            for u_row, u_coeff in u_side:
                U[u_row][col] = u_coeff
            
            for v_koff, v_coeff in v_side:
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k][col] = v_coeff
            
            for w_koff, w_coeff in w_side:
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i][col] = w_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
