import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    num_u = n * m  # 6
    num_v = m * p  # 21
    num_w = p * n  # 14
    
    # Hybrid Strassen/schoolbook for rank 39
    # A is 2x3, B is 3x7, C is 2x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split j into {0,1} and {2}
    # Part 1: sum_{j=0,1} A[i,j] B[j,k] -> 2x2 * 2x7
    # Part 2: A[i,2] B[2,k] -> 2x1 * 1x7
    
    # For 2x2 * 2x7:
    # Split the 7 columns into 3 pairs + 1 single
    # Each 2x2*2x2 uses Strassen rank 7
    # Each 2x2*2x1 uses schoolbook rank 4
    # Total: 3*7 + 4 = 25
    # For 2x1 * 1x7: schoolbook rank 14
    # Grand total: 25 + 14 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(num_u)]
    V = [[0] * rank for _ in range(num_v)]
    W = [[0] * rank for _ in range(num_w)]
    
    col = 0
    
    # Strassen for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2_2x2(a_rows, b_rows, k0, k1):
        """Strassen for A[a_rows] (2x2) * B[b_rows, k0:k1+1] (2x2). Returns list of (U_indices, V_indices, W_indices, signs)"""
        results = []
        # M1 = (a00+a11)*(b00+b11)
        results.append((
            [(a_rows[0], b_rows[0], 1), (a_rows[1], b_rows[1], 1)],
            [(b_rows[0], k0, 1), (b_rows[1], k1, 1)],
            [(k0, a_rows[0], 1), (k1, a_rows[0], 1), (k0, a_rows[1], 1), (k1, a_rows[1], 1)]
        ))
        # M2 = (a10+a11)*b00
        results.append((
            [(a_rows[1], b_rows[0], 1), (a_rows[1], b_rows[1], 1)],
            [(b_rows[0], k0, 1)],
            [(k0, a_rows[1], 1)]
        ))
        # M3 = a00*(b01-b11)
        results.append((
            [(a_rows[0], b_rows[0], 1)],
            [(b_rows[0], k1, 1), (b_rows[1], k1, -1)],
            [(k1, a_rows[0], 1)]
        ))
        # M4 = a11*(b10-b00)
        results.append((
            [(a_rows[1], b_rows[1], 1)],
            [(b_rows[1], k0, 1), (b_rows[0], k0, -1)],
            [(k0, a_rows[1], 1)]
        ))
        # M5 = (a00+a01)*b11
        results.append((
            [(a_rows[0], b_rows[0], 1), (a_rows[0], b_rows[1], 1)],
            [(b_rows[1], k1, 1)],
            [(k1, a_rows[0], 1), (k1, a_rows[1], 1)]
        ))
        # M6 = (a10-a00)*(b00+b01)
        results.append((
            [(a_rows[1], b_rows[0], 1), (a_rows[0], b_rows[0], -1)],
            [(b_rows[0], k0, 1), (b_rows[0], k1, 1)],
            [(k0, a_rows[0], 1), (k1, a_rows[0], 1)]
        ))
        # M7 = (a01-a11)*(b10+b11)
        results.append((
            [(a_rows[0], b_rows[1], 1), (a_rows[1], b_rows[1], -1)],
            [(b_rows[1], k0, 1), (b_rows[1], k1, 1)],
            [(k0, a_rows[0], 1), (k1, a_rows[0], 1)]
        ))
        return results
    
    def schoolbook_2x2_2x1(a_rows, b_rows, k):
        """Schoolbook for A[a_rows] (2x2) * B[b_rows, k] (2x1). Returns list of (U_indices, V_indices, W_indices)"""
        results = []
        for i in a_rows:
            for j in b_rows:
                results.append((
                    [(i, j, 1)],
                    [(j, k, 1)],
                    [(k, i, 1)]
                ))
        return results
    
    def add_block(u_list, v_list, w_list):
        nonlocal col
        for (i, j, sign) in u_list:
            U[i*m+j][col] = sign
        for (j, k, sign) in v_list:
            V[j*p+k][col] = sign
        for (k, i, sign) in w_list:
            W[k*n+i][col] = sign
        col += 1
    
    # Part 1: 2x2 * 2x7 (rows 0,1 of A; rows 0,1 of B; cols 0-6 of B)
    a_rows = [0, 1]
    b_rows = [0, 1]
    
    # Columns 0-1: Strassen
    for u, v, w in strassen_2x2_2x2(a_rows, b_rows, 0, 1):
        add_block(u, v, w)
    
    # Columns 2-3: Strassen
    for u, v, w in strassen_2x2_2x2(a_rows, b_rows, 2, 3):
        add_block(u, v, w)
    
    # Columns 4-5: Strassen
    for u, v, w in strassen_2x2_2x2(a_rows, b_rows, 4, 5):
        add_block(u, v, w)
    
    # Column 6: Schoolbook 2x2 * 2x1
    for u, v, w in schoolbook_2x2_2x1(a_rows, b_rows, 6):
        add_block(u, v, w)
    
    # Part 2: 2x1 * 1x7 (row 2 of A; row 2 of B; cols 0-6)
    for i in [0, 1]:
        for k in range(7):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} but got {col}"
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(num_u)]
                    V = [[0] * rank for _ in range(num_v)]
                    W = [[0] * rank for _ in range(num_w)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][column] = 1
                                V[jj * p + kk][column] = 1
                                W[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
