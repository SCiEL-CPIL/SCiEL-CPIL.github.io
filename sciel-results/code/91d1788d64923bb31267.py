"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    def time_check():
        return time.time() - start_time < DEADLINE
    
    # Strategy: Use Strassen-based decomposition for 4x4 x 4x4 (rank 49)
    # plus schoolbook for 4x4 x 4x3 (rank 12), total rank 61
    # Then verify and fall back to schoolbook if invalid
    
    rank_strassen = 49
    rank_extra = 4 * 3  # 12
    total_rank = rank_strassen + rank_extra  # 61
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 61
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 61
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 61
    
    # Build Strassen 4x4 x 4x4 (terms 0-48)
    # Block-level Strassen for 4x4 (2x2 block matrix)
    
    # For each block multiplier b (0-6), define A-blocks, B-blocks, C-blocks
    # A-blocks for block multiplier b: list of (block_row, block_col, sign)
    block_A = [
        [((0,0,1), (1,1,1))],  # M0: A00 + A11
        [((1,0,1), (1,1,1))],  # M1: A10 + A11
        [((0,0,1))],           # M2: A00
        [((1,1,1))],           # M3: A11
        [((0,0,1), (0,1,1))],  # M4: A00 + A01
        [((1,0,1), (0,0,-1))], # M5: A10 - A00
        [((0,1,1), (1,1,-1))], # M6: A01 - A11
    ]
    
    block_B = [
        [((0,0,1), (1,1,1))],  # M0: B00 + B11
        [((0,0,1))],           # M1: B00
        [((0,1,1), (1,1,-1))], # M2: B01 - B11
        [((1,0,1), (0,0,-1))], # M3: B10 - B00
        [((1,1,1))],           # M4: B11
        [((0,0,1), (0,1,1))],  # M5: B00 + B01
        [((1,0,1), (1,1,1))],  # M6: B10 + B11
    ]
    
    block_C = [
        [((0,0,1), (1,1,1))],  # M0: C00 + C11
        [((1,0,1), (1,1,-1))], # M1: C10 - C11
        [((0,1,1), (1,1,1))],  # M2: C01 + C11
        [((0,0,1), (1,0,1))],  # M3: C00 + C10
        [((0,0,-1), (0,1,1))], # M4: -C00 + C01
        [((1,1,1))],           # M5: C11
        [((0,0,1))],           # M6: C00
    ]
    
    # 2x2 Strassen scalar patterns
    # For X*Y=Z (2x2), the 7 scalar multipliers:
    # s0: (X00+X11)(Y00+Y11)
    # s1: (X10+X11)Y00
    # s2: X00(Y01-Y11)
    # s3: X11(Y10-Y00)
    # s4: (X00+X01)Y11
    # s5: (X10-X00)(Y00+Y01)
    # s6: (X01-X11)(Y10+Y11)
    # Z00 = s0 + s3 - s4 + s6
    # Z01 = s2 + s4
    # Z10 = s1 + s3
    # Z11 = s0 - s1 + s2 + s5
    
    # X elements for each s: list of (row, col, sign)
    X_patterns = [
        [((0,0,1), (1,1,1))],  # s0
        [((1,0,1), (1,1,1))],  # s1
        [((0,0,1))],           # s2
        [((1,1,1))],           # s3
        [((0,0,1), (0,1,1))],  # s4
        [((1,0,1), (0,0,-1))], # s5
        [((0,1,1), (1,1,-1))], # s6
    ]
    
    # Y elements for each s
    Y_patterns = [
        [((0,0,1), (1,1,1))],  # s0
        [((0,0,1))],           # s1
        [((0,1,1), (1,1,-1))], # s2
        [((1,0,1), (0,0,-1))], # s3
        [((1,1,1))],           # s4
        [((0,0,1), (0,1,1))],  # s5
        [((1,0,1), (1,1,1))],  # s6
    ]
    
    # Z elements for each s
    Z_patterns = [
        [((0,0,1), (1,1,1))],  # s0
        [((1,0,1), (1,1,-1))], # s1
        [((0,1,1), (1,1,1))],  # s2
        [((0,0,1), (1,0,1))],  # s3
        [((0,0,-1), (0,1,1))], # s4
        [((1,1,1))],           # s5
        [((0,0,1))],           # s6
    ]
    
    # Build the 49 terms
    for b in range(7):
        for s in range(7):
            term_idx = b * 7 + s
            
            # U entries (A elements)
            for (abr, abc, asgn) in block_A[b]:
                for (xr, xc, xsgn) in X_patterns[s]:
                    row = abr * 2 + xr
                    col = abc * 2 + xc
                    sign = asgn * xsgn
                    U[row * m + col][term_idx] += sign
            
            # V entries (B elements)
            for (bbr, bbc, bsgn) in block_B[b]:
                for (yr, yc, ysgn) in Y_patterns[s]:
                    row = bbr * 2 + yr
                    col = bbc * 2 + yc
                    sign = bsgn * ysgn
                    V[row * p + col][term_idx] += sign
            
            # W entries (C elements) - W[k*n+i]
            for (cbr, cbc, csgn) in block_C[b]:
                for (zr, zc, zsgn) in Z_patterns[s]:
                    row = cbr * 2 + zr
                    col = cbc * 2 + zc
                    sign = csgn * zsgn
                    W[col * n + row][term_idx] += sign
    
    # Add schoolbook for extra 3 columns (k=4,5,6)
    for j in range(m):
        for i in range(n):
            for k in range(4, p):
                extra_idx = j * (p - 4) + (k - 4)
                term_idx = rank_strassen + extra_idx
                U[i * m + j][term_idx] = 1
                V[j * p + k][term_idx] = 1
                W[k * n + i][term_idx] = 1
    
    # Verify decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(actual - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

