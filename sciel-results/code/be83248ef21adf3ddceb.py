"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Deadline guard: 140s safety margin within 150s limit
    start_time = time.time()
    deadline = start_time + 140
    
    # Build schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 80
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_schoolbook
    
    # Helper: verify a decomposition
    def verify(U_try, V_try, W_try, tol=1e-9):
        n, m, p = DIMENSIONS
        # Check on-support entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_try[a, :], V_try[b, :] * W_try[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check off-support entries: exhaustive check
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k2 = divmod(b, p)
                    i2, k1 = divmod(c, n)
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue
                    val = np.dot(U_try[a, :], V_try[b, :] * W_try[c, :])
                    if abs(val) > tol:
                        return False
        return True
    
    # Define the 7 Strassen factors for 2x2x2
    # 2x2 indexing: row*2 + col
    # X entries: X00=0, X01=1, X10=2, X11=3
    # Y entries: Y00=0, Y01=1, Y10=2, Y11=3
    # Z entries: Z00=0, Z01=1, Z10=2, Z11=3
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # s=0: M1 = (X11 - X21)(Y21 - Y22), contributes to Z11 and Z21
    U2[3, 0] = 1    # X11
    U2[2, 0] = -1   # X21
    V2[2, 0] = -1   # Y21
    V2[3, 0] = 1    # Y22
    W2[0, 0] = 1    # Z11
    W2[2, 0] = 1    # Z21
    
    # s=1: M2 = (X11 + X12)Y22, contributes to Z12
    U2[3, 1] = 1    # X11
    U2[1, 1] = 1    # X12
    V2[3, 1] = 1    # Y22
    W2[1, 1] = 1    # Z12
    
    # s=2: M3 = X11(Y11 - Y22), contributes to Z12 and Z22
    U2[3, 2] = 1    # X11
    V2[0, 2] = 1    # Y11
    V2[3, 2] = -1   # Y22
    W2[1, 2] = 1    # Z12
    W2[3, 2] = 1    # Z22
    
    # s=3: M4 = X22(Y11 + Y22), contributes to Z11 and Z22
    U2[3, 3] = 1    # X22
    V2[0, 3] = 1    # Y11
    V2[3, 3] = 1    # Y22
    W2[0, 3] = 1    # Z11
    W2[3, 3] = 1    # Z22
    
    # s=4: M5 = (X12 - X22)(Y11 + Y21), contributes to Z11 and Z22
    U2[1, 4] = 1    # X12
    U2[3, 4] = -1   # X22
    V2[0, 4] = 1    # Y11
    V2[2, 4] = 1    # Y21
    W2[0, 4] = -1   # Z11
    W2[3, 4] = 1    # Z22
    
    # s=5: M6 = (X11 + X21)(Y22 - Y21), contributes to Z21
    U2[3, 5] = 1    # X11
    U2[2, 5] = 1    # X21
    V2[3, 5] = 1    # Y22
    V2[2, 5] = -1   # Y21
    W2[2, 5] = 1    # Z21
    
    # s=6: M7 = (X21 - X11)(Y11 + Y12), contributes to Z11
    U2[2, 6] = 1    # X21
    U2[3, 6] = -1   # X11
    V2[0, 6] = 1    # Y11
    V2[1, 6] = 1    # Y12
    W2[0, 6] = 1    # Z11
    
    # Verify the 2x2x2 Strassen decomposition
    def verify_2x2(U2_try, V2_try, W2_try, tol=1e-9):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    i_row, i_col = divmod(i, 2)
                    j_row, j_col = divmod(j, 2)
                    k_row, k_col = divmod(k, 2)
                    if i_col == j_row and i_row == k_row and j_col == k_col:
                        expected = 1.0
                    else:
                        expected = 0.0
                    val = np.dot(U2_try[i, :], V2_try[j, :] * W2_try[k, :])
                    if abs(val - expected) > tol:
                        return False
        return True
    
    if not verify_2x2(U2, V2, W2):
        # Fallback to schoolbook if Strassen factors are wrong
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Build the 4x4x4 rank-49 decomposition using Kronecker product of Strassen factors
    # The 4x4 matrix A has blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # For block product r (0..6) and 2x2 product s (0..6):
    # col = r * 7 + s
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            for block_a in range(4):
                for u_a in range(4):
                    if U2[block_a, r] != 0 and U2[u_a, s] != 0:
                        block_row_a = block_a // 2
                        block_col_a = block_a % 2
                        u_row_a = u_a // 2
                        u_col_a = u_a % 2
                        a_row = block_row_a * 2 + u_row_a
                        a_col = block_col_a * 2 + u_col_a
                        a_idx = a_row * 4 + a_col
                        U4[a_idx, col] = U2[block_a, r] * U2[u_a, s]
            
            for block_b in range(4):
                for u_b in range(4):
                    if V2[block_b, r] != 0 and V2[u_b, s] != 0:
                        block_row_b = block_b // 2
                        block_col_b = block_b % 2
                        u_row_b = u_b // 2
                        u_col_b = u_b % 2
                        b_row = block_row_b * 2 + u_row_b
                        b_col = block_col_b * 2 + u_col_b
                        b_idx = b_row * 4 + b_col
                        V4[b_idx, col] = V2[block_b, r] * V2[u_b, s]
            
            for block_c in range(4):
                for u_c in range(4):
                    if W2[block_c, r] != 0 and W2[u_c, s] != 0:
                        block_row_c = block_c // 2
                        block_col_c = block_c % 2
                        u_row_c = u_c // 2
                        u_col_c = u_c % 2
                        c_row = block_row_c * 2 + u_row_c
                        c_col = block_col_c * 2 + u_col_c
                        c_idx = c_row * 4 + c_col
                        W4[c_idx, col] = W2[block_c, r] * W2[u_c, s]
    
    # Verify the 4x4x4 Strassen decomposition
    def verify_4x4(U4_try, V4_try, W4_try, tol=1e-9):
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    i_row, i_col = divmod(i, 4)
                    j_row, j_col = divmod(j, 4)
                    k_row, k_col = divmod(k, 4)
                    if i_col == j_row and i_row == k_row and j_col == k_col:
                        expected = 1.0
                    else:
                        expected = 0.0
                    val = np.dot(U4_try[i, :], V4_try[j, :] * W4_try[k, :])
                    if abs(val - expected) > tol:
                        return False
        return True
    
    if not verify_4x4(U4, V4, W4):
        # Fallback to schoolbook if 4x4x4 Strassen is wrong
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Now extend to 4x4x5
    # C[4x5] = A[4x4] * B[4x5]
    # Split B into B1[4x4] and B2[4x1]
    # C = A*B1 + A*B2
    # A*B1: use rank-49 Strassen (above)
    # A*B2: use schoolbook rank 16
    
    # Total rank: 49 + 16 = 65
    
    U65 = np.zeros((16, 65))
    V65 = np.zeros((20, 65))  # m*p = 4*5 = 20
    W65 = np.zeros((20, 65))  # p*n = 5*4 = 20
    
    # Copy the 4x4x4 Strassen part
    U65[:, :49] = U4
    # V4 is 16x49, corresponds to B[:, 0:4]
    # B index: b = row*5 + col for 4x5 matrix
    # For B[:, 0:4]: b = row*5 + col where col=0..3
    # V4[b_idx, col] where b_idx = row*4 + col (for 4x4)
    # Need to map to 4x5 indexing: b_45 = row*5 + col
    for b_idx in range(16):
        b_row = b_idx // 4
        b_col = b_idx % 4
        b_45 = b_row * 5 + b_col
        V65[b_45, :49] = V4[b_idx, :49]
    
    # W4 is 16x49, corresponds to C[:, 0:4]
    # C index: c = row*4 + col for 4x4
    # For C[:, 0:4]: c = row*4 + col where col=0..3
    # W65 index: c_54 = col*4 + row for 5x4 (p*n = 5*4)
    # The convention is W[k*n+i, r] where k is the output column, i is the output row
    # So W index = k*4 + i where k=0..4 (output col), i=0..3 (output row)
    # For C[:, 0:4]: k=0..3, i=0..3
    # C 4x4 index: c = i*4 + k (row i, col k)
    # W65 index: k*4 + i
    for c_idx in range(16):
        c_row = c_idx // 4  # i
        c_col = c_idx % 4   # k
        w_idx = c_col * 4 + c_row
        W65[w_idx, :49] = W4[c_idx, :49]
    
    # Now add the 4x4x1 schoolbook part for columns 49..64
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # For each (i, j): one column
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U65[i * 4 + j, col] = 1  # A[i, j]
            V65[j * 5 + 4, col] = 1  # B[j, 4]
            W65[4 * 4 + i, col] = 1  # C[i, 4] -> k=4, i=i -> index = 4*4 + i = 16 + i
    
    # Verify the 65-rank decomposition
    if verify(U65, V65, W65):
        best_U, best_V, best_W, best_rank = U65.copy(), V65.copy(), W65.copy(), 65
    else:
        # If verification fails, fall back to schoolbook
        best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_schoolbook
    
    # If we have time, try to find a lower-rank decomposition
    if time.time() < deadline - 30:
        # Try rank 64, 63, etc. with random search
        for test_rank in range(64, 40, -1):
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                # Initialize with the 65-rank decomposition, zero out one column
                U_try = U65.copy()
                V_try = V65.copy()
                W_try = W65.copy()
                
                # Remove the last column and try to repair
                U_try[:, -1] = 0
                V_try[:, -1] = 0
                W_try[:, -1] = 0
                
                # Try random perturbation of the last column
                rng = np.random.RandomState(attempt)
                U_try[:, -1] = rng.randn(16) * 0.1
                V_try[:, -1] = rng.randn(20) * 0.1
                W_try[:, -1] = rng.randn(20) * 0.1
                
                if verify(U_try, V_try, W_try):
                    best_U, best_V, best_W, best_rank = U_try.copy(), V_try.copy(), W_try.copy(), test_rank
                    found = True
                    break
            
            if found:
                break
    
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    return result
# EVOLVE-BLOCK-END
