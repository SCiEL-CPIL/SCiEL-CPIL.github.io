import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16

    # Strassen rank-7 decomposition for 2x2 matrices
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # We need to tile the 2x4 * 4x8 multiplication into 2x2 * 2x2 blocks.
    # A is 2x4, B is 4x8, C is 2x8.
    # We can view A as having 2 blocks of size 2x2 (columns 0-1, 2-3)
    # B as having 4 blocks of size 2x2 (rows 0-1, 2-3; columns 0-1, 2-3, 4-5, 6-7)
    # So C = [A11 B11 + A12 B21, A11 B12 + A12 B22, A11 B13 + A12 B23, A11 B14 + A12 B24]
    # where A11, A12 are 2x2, and B11..B14, B21..B24 are 2x2.
    
    # This gives 4 independent 2x2*2x2 multiplications, each with rank 7.
    # Total rank = 4 * 7 = 28.
    
    # Let's verify the tensor indices.
    # A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B[j, k] where j in {0,1,2,3}, k in {0,...,7}
    # C[i, k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    # Block decomposition:
    # Let A = [A_left | A_right] where A_left = A[:, 0:2], A_right = A[:, 2:4] (both 2x2)
    # Let B = [B_top; B_bottom] where B_top = B[0:2, :], B_bottom = B[2:4, :] (both 2x8)
    # Then C = A_left * B_top + A_right * B_bottom
    
    # Now B_top is 2x8, which we can split into 4 blocks of 2x2: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # Similarly B_bottom is 2x8, split into 4 blocks of 2x2.
    
    # So C = A_left * [B_t1 | B_t2 | B_t3 | B_t4] + A_right * [B_b1 | B_b2 | B_b3 | B_b4]
    # = [A_left*B_t1 + A_right*B_b1 | A_left*B_t2 + A_right*B_b2 | A_left*B_t3 + A_right*B_b3 | A_left*B_t4 + A_right*B_b4]
    
    # Each term like A_left*B_t1 is a 2x2 * 2x2 multiplication.
    # There are 8 such 2x2*2x2 multiplications total (4 for A_left, 4 for A_right).
    # Each has rank 7, so total rank = 8 * 7 = 56.
    
    # Wait, that's the same as the schoolbook rank of 64 reduced by 12.5%? No, 8*7=56 < 64.
    
    # Let me re-examine. The schoolbook rank is n*m*p = 2*4*8 = 64.
    # With Strassen tiling: we have 8 independent 2x2*2x2 multiplications, each rank 7.
    # Total rank = 56.
    
    # Let's construct this.
    # For each of the 8 sub-problems, we use the 7 Strassen multiplications.
    # The sub-problems are:
    # 1. A_left * B_top[:, 0:2]  -> contributes to C[:, 0:2]
    # 2. A_left * B_top[:, 2:4]  -> contributes to C[:, 2:4]
    # 3. A_left * B_top[:, 4:6]  -> contributes to C[:, 4:6]
    # 4. A_left * B_top[:, 6:8]  -> contributes to C[:, 6:8]
    # 5. A_right * B_bottom[:, 0:2]  -> contributes to C[:, 0:2]
    # 6. A_right * B_bottom[:, 2:4]  -> contributes to C[:, 2:4]
    # 7. A_right * B_bottom[:, 4:6]  -> contributes to C[:, 4:6]
    # 8. A_right * B_bottom[:, 6:8]  -> contributes to C[:, 6:8]
    
    # For each 2x2*2x2 multiplication, the Strassen decomposition is:
    # Given A_sub = [[a11, a12], [a21, a22]] and B_sub = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In the tensor decomposition, each "multiplication" M_t corresponds to a rank-1 term.
    # For M1: the "A-factor" is (a11+a22), the "B-factor" is (b11+b22).
    # So U has entries for a11, a22 (each with coefficient 1) in this column.
    # V has entries for b11, b22 (each with coefficient 1) in this column.
    # W has entries for c11 (coeff 1), c22 (coeff 1) in this column.
    
    # Let me map the indices:
    # For sub-problem s (s in 0..7):
    #   A_sub indices: (i, j) where i in {0,1}, j in {0,1}
    #   B_sub indices: (j, k) where j in {0,1}, k in {0,1}
    #   C_sub indices: (i, k) where i in {0,1}, k in {0,1}
    
    # Global A index: i*m + j where i in {0,1}, j in {0,1,2,3}
    #   For A_left: j = j_sub (0 or 1)
    #   For A_right: j = 2 + j_sub (2 or 3)
    # Global B index: j*p + k where j in {0,1,2,3}, k in {0,...,7}
    #   For B_top: j = j_sub (0 or 1)
    #   For B_bottom: j = 2 + j_sub (2 or 3)
    #   k = k_block * 2 + k_sub where k_block in {0,1,2,3}, k_sub in {0,1}
    # Global C index: k*n + i where k in {0,...,7}, i in {0,1}
    #   k = k_block * 2 + k_sub
    
    # Let's build the U, V, W matrices.
    rank = 56
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    
    col = 0
    # Strassen coefficients for each of the 7 multiplications
    # (A_coefficients, B_coefficients, C_coefficients)
    # A_sub = [[a00, a01], [a10, a11]], B_sub = [[b00, b01], [b10, b11]]
    # C_sub = [[c00, c01], [c10, c11]]
    strassen_terms = [
        # M1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (a10+a11)b00 -> c10 += 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3: a00(b01-b11) -> c01 += 1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # M4: a11(b10-b00) -> c00 += 1, c10 += 1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (a00+a01)b11 -> c00 -= 1, c01 += 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (a10-a00)(b00+b01) -> c11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (a01-a11)(b10+b11) -> c00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for s in range(8):
        # s = 0..3: A_left * B_top[:, s*2:(s+1)*2]
        # s = 4..7: A_right * B_bottom[:, (s-4)*2:(s-3)*2]
        if s < 4:
            a_row_offset = 0  # A_left: rows 0,1
            a_col_offset = 0  # A_left: cols 0,1
            b_row_offset = 0  # B_top: rows 0,1
            k_block = s
        else:
            a_row_offset = 0  # A_right: rows 0,1
            a_col_offset = 2  # A_right: cols 2,3
            b_row_offset = 2  # B_bottom: rows 2,3
            k_block = s - 4
        
        for t in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[t]
            for (ai, aj), coeff in a_coeffs.items():
                gi = a_row_offset + ai
                gj = a_col_offset + aj
                U[gi * m + gj][col] += coeff
            for (bi, bk), coeff in b_coeffs.items():
                gj = b_row_offset + bi
                gk = k_block * 2 + bk
                V[gj * p + gk][col] += coeff
            for (ci, ck), coeff in c_coeffs.items():
                gk = k_block * 2 + ck
                gi = ci
                W[gk * n + gi][col] += coeff
            col += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 for all valid (i,j,k).
    
    # Let's verify a few entries
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        r = n * m * p
        U = [[0]*r for _ in range(nm)]
        V = [[0]*r for _ in range(mp)]
        W = [[0]*r for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
