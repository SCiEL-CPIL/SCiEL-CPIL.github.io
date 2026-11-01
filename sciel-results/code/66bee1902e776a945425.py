"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 28 via corrected Strassen blocks."""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    # Build the standard rank-32 schoolbook decomposition
    def build_schoolbook():
        rank = n * m * p  # 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Build rank-28 Strassen decomposition (based on 2x2 blocks)
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A0 (2x2, cols 0-1) and A1 (2x2, cols 2-3)
    # Split B into B0 (rows 0-1) and B1 (rows 2-3), each 2x4
    # Split B0 into B00 (2x2, cols 0-1) and B01 (2x2, cols 2-3)
    # Split B1 into B10 (2x2, cols 0-1) and B11 (2x2, cols 2-3)
    # C = A0*B0 + A1*B1
    # C0 (cols 0-1) = A0*B00 + A1*B10
    # C1 (cols 2-3) = A0*B01 + A1*B11
    
    # Use Strassen for each 2x2 * 2x2 block
    # Strassen for X (2x2) * Y (2x2):
    # M0 = (X00+X11)*(Y00+Y11)
    # M1 = (X10+X11)*Y00
    # M2 = X00*(Y01-Y11)
    # M3 = X11*(Y10-Y00)
    # M4 = (X00+X01)*Y11
    # M5 = (X10-X00)*(Y00+Y01)
    # M6 = (X11-X01)*(Y10+Y11)
    # Result:
    # R00 = M0+M3-M4+M6
    # R01 = M2+M4
    # R10 = M1+M3
    # R11 = M0-M1+M2+M5

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28
        
        # We have two Strassen computations:
        # 1. A0 * B0 -> contributes to C0 (cols 0-1)
        # 2. A1 * B1 -> contributes to C1 (cols 2-3)
        # Wait, that's only 14 terms. But we need A0*B00, A1*B10 for C0, and A0*B01, A1*B11 for C1.
        # So we need 4 Strassen computations? No, Strassen is for 2x2*2x2.
        # Let's re-evaluate.
        
        # A is 2x4. B is 4x4.
        # We can view this as two 2x2 * 2x4 multiplications? No, Strassen is for square matrices.
        # We can split B into two 2x2 blocks vertically? No, B is 4x4.
        # Let's split the 4x4 B into four 2x2 blocks: B00, B01, B10, B11.
        # A is 2x4. Split into A0 (2x2) and A1 (2x2).
        # C = A0 * [B00 B01] + A1 * [B10 B11]
        # C = [A0*B00 + A1*B10, A0*B01 + A1*B11]
        
        # So we need to compute:
        # T1 = A0 * B00 (2x2 * 2x2) -> 7 terms
        # T2 = A1 * B10 (2x2 * 2x2) -> 7 terms
        # T3 = A0 * B01 (2x2 * 2x2) -> 7 terms
        # T4 = A1 * B11 (2x2 * 2x2) -> 7 terms
        # Total 28 terms.
        
        # Let's define the Strassen terms for a generic 2x2 * 2x2.
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        # M0 = (x00+x11)*(y00+y11)
        # M1 = (x10+x11)*y00
        # M2 = x00*(y01-y11)
        # M3 = x11*(y10-y00)
        # M4 = (x00+x01)*y11
        # M5 = (x10-x00)*(y00+y01)
        # M6 = (x11-x01)*(y10+y11)
        # R00 = M0+M3-M4+M6
        # R01 = M2+M4
        # R10 = M1+M3
        # R11 = M0-M1+M2+M5
        
        # For each of the 4 blocks, we assign 7 columns in U, V, W.
        
        # Block 1: A0 * B00 -> C00 (top-left 2x2 of C, which is cols 0-1, rows 0-1)
        # A0 indices in A: i in {0,1}, j in {0,1}
        # B00 indices in B: j in {0,1}, k in {0,1}
        # C00 indices in C: i in {0,1}, k in {0,1}
        
        # Block 2: A1 * B10 -> C10 (bottom-left 2x2 of C, which is cols 0-1, rows 0-1)
        # Wait, C is 2x4. Rows are i (0,1). Cols are k (0,1,2,3).
        # C00 is C[0:2, 0:2]
        # C10 is C[0:2, 0:2] from A1*B10? No.
        # C = [A0*B00 + A1*B10, A0*B01 + A1*B11]
        # So C[0:2, 0:2] = A0*B00 + A1*B10
        # C[0:2, 2:4] = A0*B01 + A1*B11
        
        # So for C[0:2, 0:2], we sum contributions from Block 1 and Block 2.
        # For C[0:2, 2:4], we sum contributions from Block 3 and Block 4.
        
        # Let's define a helper to add Strassen terms for a specific block.
        
        def add_strassen_block(col_start, A_i_offset, A_j_offset, B_j_offset, B_k_offset, C_k_offset):
            # A_i_offset: 0 or 2 (row offset in A)
            # A_j_offset: 0 or 2 (col offset in A)
            # B_j_offset: 0 or 2 (row offset in B)
            # B_k_offset: 0 or 2 (col offset in B)
            # C_k_offset: 0 or 2 (col offset in C)
            
            # Strassen terms
            # M0 = (x00+x11)*(y00+y11)
            # M1 = (x10+x11)*y00
            # M2 = x00*(y01-y11)
            # M3 = x11*(y10-y00)
            # M4 = (x00+x01)*y11
            # M5 = (x10-x00)*(y00+y01)
            # M6 = (x11-x01)*(y10+y11)
            
            # U factors (A side)
            # x00 = A[A_i_offset + 0, A_j_offset + 0]
            # x01 = A[A_i_offset + 0, A_j_offset + 1]
            # x10 = A[A_i_offset + 1, A_j_offset + 0]
            # x11 = A[A_i_offset + 1, A_j_offset + 1]
            
            # V factors (B side)
            # y00 = B[B_j_offset + 0, B_k_offset + 0]
            # y01 = B[B_j_offset + 0, B_k_offset + 1]
            # y10 = B[B_j_offset + 1, B_k_offset + 0]
            # y11 = B[B_j_offset + 1, B_k_offset + 1]
            
            # W factors (C side)
            # R00 = C[C_k_offset + 0, 0] and C[C_k_offset + 0, 1]? No, C is 2x4.
            # C rows are i (0,1). C cols are k (0,1,2,3).
            # The result of a 2x2 * 2x2 is a 2x2 matrix.
            # R00 -> C[0, C_k_offset + 0]
            # R01 -> C[0, C_k_offset + 1]
            # R10 -> C[1, C_k_offset + 0]
            # R11 -> C[1, C_k_offset + 1]
            
            # Let's define the contributions for each of the 7 terms.
            # Each term has a U factor, V factor, and W factor.
            # U factor is a vector in R^8 (flattened A)
            # V factor is a vector in R^16 (flattened B)
            # W factor is a vector in R^8 (flattened C)
            
            # Term 0: M0 = (x00+x11)*(y00+y11)
            # U: x00 + x11
            # V: y00 + y11
            # W: R00, R11 (both get M0)
            # R00 = M0+M3-M4+M6 -> W[0, C_k_offset] += M0
            # R11 = M0-M1+M2+M5 -> W[1, C_k_offset+1] += M0
            
            # Term 1: M1 = (x10+x11)*y00
            # U: x10 + x11
            # V: y00
            # W: R10, R11
            # R10 = M1+M3 -> W[1, C_k_offset] += M1
            # R11 = M0-M1+M2+M5 -> W[1, C_k_offset+1] -= M1
            
            # Term 2: M2 = x00*(y01-y11)
            # U: x00
            # V: y01 - y11
            # W: R01, R11
            # R01 = M2+M4 -> W[0, C_k_offset+1] += M2
            # R11 = M0-M1+M2+M5 -> W[1, C_k_offset+1] += M2
            
            # Term 3: M3 = x11*(y10-y00)
            # U: x11
            # V: y10 - y00
            # W: R00, R10
            # R00 = M0+M3-M4+M6 -> W[0, C_k_offset] += M3
            # R10 = M1+M3 -> W[1, C_k_offset] += M3
            
            # Term 4: M4 = (x00+x01)*y11
            # U: x00 + x01
            # V: y11
            # W: R00, R01
            # R00 = M0+M3-M4+M6 -> W[0, C_k_offset] -= M4
            # R01 = M2+M4 -> W[0, C_k_offset+1] += M4
            
            # Term 5: M5 = (x10-x00)*(y00+y01)
            # U: x10 - x00
            # V: y00 + y01
            # W: R11
            # R11 = M0-M1+M2+M5 -> W[1, C_k_offset+1] += M5
            
            # Term 6: M6 = (x11-x01)*(y10+y11)
            # U: x11 - x01
            # V: y10 + y11
            # W: R00
            # R00 = M0+M3-M4+M6 -> W[0, C_k_offset] += M6
            
            # Now, let's implement this.
            # We will add 7 columns to U, V, W starting at col_start.
            
            for t in range(7):
                col = col_start + t
                
                # U contributions
                if t == 0:
                    U[A_i_offset + 0 + A_j_offset][col] += 1  # x00
                    U[A_i_offset + 1 + A_j_offset][col] += 1  # x11
                elif t == 1:
                    U[A_i_offset + 1 + A_j_offset][col] += 1  # x10
                    U[A_i_offset + 1 + A_j_offset][col] += 1  # x11 -> wait, x10 is row 1, col 0. x11 is row 1, col 1.
                    # Correction:
                    U[A_i_offset + 1 + A_j_offset][col] = 0 # Reset
                    U[A_i_offset + 1 + A_j_offset][col] += 1 # x10 (row 1, col 0)
                    U[A_i_offset + 1 + A_j_offset + 1][col] += 1 # x11 (row 1, col 1)
                elif t == 2:
                    U[A_i_offset + 0 + A_j_offset][col] += 1 # x00
                elif t == 3:
                    U[A_i_offset + 1 + A_j_offset + 1][col] += 1 # x11
                elif t == 4:
                    U[A_i_offset + 0 + A_j_offset][col] += 1 # x00
                    U[A_i_offset + 0 + A_j_offset + 1][col] += 1 # x01
                elif t == 5:
                    U[A_i_offset + 1 + A_j_offset][col] += 1 # x10
                    U[A_i_offset + 0 + A_j_offset][col] -= 1 # x00
                elif t == 6:
                    U[A_i_offset + 1 + A_j_offset + 1][col] += 1 # x11
                    U[A_i_offset + 0 + A_j_offset + 1][col] -= 1 # x01
                
                # V contributions
                if t == 0:
                    V[B_j_offset + 0 + B_k_offset][col] += 1 # y00
                    V[B_j_offset + 1 + B_k_offset + 1][col] += 1 # y11
                elif t == 1:
                    V[B_j_offset + 0 + B_k_offset][col] += 1 # y00
                elif t == 2:
                    V[B_j_offset + 0 + B_k_offset + 1][col] += 1 # y01
                    V[B_j_offset + 1 + B_k_offset + 1][col] -= 1 # y11
                elif t == 3:
                    V[B_j_offset + 1 + B_k_offset][col] += 1 # y10
                    V[B_j_offset + 0 + B_k_offset][col] -= 1 # y00
                elif t == 4:
                    V[B_j_offset + 1 + B_k_offset + 1][col] += 1 # y11
                elif t == 5:
                    V[B_j_offset + 0 + B_k_offset][col] += 1 # y00
                    V[B_j_offset + 0 + B_k_offset + 1][col] += 1 # y01
                elif t == 6:
                    V[B_j_offset + 1 + B_k_offset][col] += 1 # y10
                    V[B_j_offset + 1 + B_k_offset + 1][col] += 1 # y11
                
                # W contributions
                # C flat index: k * n + i = k * 2 + i
                # R00 -> C[0, C_k_offset] -> index (C_k_offset) * 2 + 0
                # R01 -> C[0, C_k_offset+1] -> index (C_k_offset+1) * 2 + 0
                # R10 -> C[1, C_k_offset] -> index (C_k_offset) * 2 + 1
                # R11 -> C[1, C_k_offset+1] -> index (C_k_offset+1) * 2 + 1
                
                w00_idx = C_k_offset * 2 + 0
                w01_idx = (C_k_offset + 1) * 2 + 0
                w10_idx = C_k_offset * 2 + 1
                w11_idx = (C_k_offset + 1) * 2 + 1
                
                if t == 0:
                    W[w00_idx][col] += 1 # R00
                    W[w11_idx][col] += 1 # R11
                elif t == 1:
                    W[w10_idx][col] += 1 # R10
                    W[w11_idx][col] -= 1 # R11
                elif t == 2:
                    W[w01_idx][col] += 1 # R01
                    W[w11_idx][col] += 1 # R11
                elif t == 3:
                    W[w00_idx][col] += 1 # R00
                    W[w10_idx][col] += 1 # R10
                elif t == 4:
                    W[w00_idx][col] -= 1 # R00
                    W[w01_idx][col] += 1 # R01
                elif t == 5:
                    W[w11_idx][col] += 1 # R11
                elif t == 6:
                    W[w00_idx][col] += 1 # R00

        # Block 1: A0 * B00 -> C[0:2, 0:2]
        # A0: i in {0,1}, j in {0,1} -> A_i_offset=0, A_j_offset=0
        # B00: j in {0,1}, k in {0,1} -> B_j_offset=0, B_k_offset=0
        # C[0:2, 0:2] -> C_k_offset=0
        add_strassen_block(0, 0, 0, 0, 0, 0)
        
        # Block 2: A1 * B10 -> C[0:2, 0:2]
        # A1: i in {0,1}, j in {2,3} -> A_i_offset=0, A_j_offset=2
        # B10: j in {2,3}, k in {0,1} -> B_j_offset=2, B_k_offset=0
        # C[0:2, 0:2] -> C_k_offset=0
        add_strassen_block(7, 0, 2, 2, 0, 0)
        
        # Block 3: A0 * B01 -> C[0:2, 2:4]
        # A0: i in {0,1}, j in {0,1} -> A_i_offset=0, A_j_offset=0
        # B01: j in {0,1}, k in {2,3} -> B_j_offset=0, B_k_offset=2
        # C[0:2, 2:4] -> C_k_offset=2
        add_strassen_block(14, 0, 0, 0, 2, 2)
        
        # Block 4: A1 * B11 -> C[0:2, 2:4]
        # A1: i in {0,1}, j in {2,3} -> A_i_offset=0, A_j_offset=2
        # B11: j in {2,3}, k in {2,3} -> B_j_offset=2, B_k_offset=2
        # C[0:2, 2:4] -> C_k_offset=2
        add_strassen_block(21, 0, 2, 2, 2, 2)
        
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        # Check diagonal entries (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal entries (should be 0)
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b_idx in range(m*p):
                j2 = b_idx // p
                k1 = b_idx % p
                for c_idx in range(p*n):
                    k2 = c_idx // n
                    i2 = c_idx % n
                    
                    # Skip diagonal
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    
                    val = 0.0
                    for r in range(rank):
                        val += U[a][r] * V[b_idx][r] * W[c_idx][r]
                    if abs(val) > 1e-9:
                        return False
        return True

    def is_half_integer(x, tol=1e-6):
        """Check if x is a half-integer (i.e., 2*x is an integer)."""
        val = 2 * x
        rounded = round(val)
        return abs(val - rounded) < tol

    def check_half_integer_constraint(U, V, W):
        """Check if all entries in U, V, W are half-integers."""
        for row in U:
            for x in row:
                if not is_half_integer(x):
                    return False
        for row in V:
            for x in row:
                if not is_half_integer(x):
                    return False
        for row in W:
            for x in row:
                if not is_half_integer(x):
                    return False
        return True

    # Try to find a rank-27 decomposition by searching for a null vector in the space of rank-28 decompositions
    def try_rank_27_search():
        if time.time() > deadline:
            return None
        
        base = build_strassen_28()
        rank28 = 28
        U28 = np.array(base["U"], dtype=float)
        V28 = np.array(base["V"], dtype=float)
        W28 = np.array(base["W"], dtype=float)
        
        # Construct the 28 tensor vectors in the full space
        dim_total = (n*m) * (m*p) * (p*n)  # 1024
        T_flat = np.zeros((rank28, dim_total))
        
        for r in range(rank28):
            idx = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        T_flat[r, idx] = U28[i, r] * V28[j, r] * W28[k, r]
                        idx += 1
        
        # Check rank of T_flat
        try:
            rank_T = np.linalg.matrix_rank(T_flat, tol=1e-10)
        except:
            return None
            
        if rank_T < rank28:
            # Find null space
            U_svd, s, Vh = np.linalg.svd(T_flat)
            null_mask = s < 1e-10
            if np.any(null_mask):
                for r_null in range(len(null_mask)):
                    if not null_mask[r_null]:
                        continue
                    
                    c = Vh[r_null, :]
                    max_c = np.max(np.abs(c))
                    if max_c == 0:
                        continue
                    c = c / max_c
                    
                    best_alpha = None
                    best_k = -1
                    
                    for k in range(rank28):
                        if abs(c[k]) < 1e-6:
                            continue
                        
                        alpha = 1.0 - c / c[k]
                        alpha[k] = 0.0
                        
                        valid = True
                        for r in range(rank28):
                            if r == k:
                                continue
                            if abs(alpha[r]) < 1e-6:
                                continue
                            if not is_half_integer(alpha[r], tol=1e-5):
                                valid = False
                                break
                        
                        if valid:
                            best_alpha = alpha
                            best_k = k
                            break
                    
                    if best_alpha is not None:
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        new_rank = rank28 - 1
                        col_map = []
                        for r in range(rank28):
                            if r != best_k:
                                col_map.append(r)
                        
                        for i in range(n*m):
                            row = []
                            for r_new, r_old in enumerate(col_map):
                                row.append(best_alpha[r_old] * U28[i, r_old])
                            U_new.append(row)
                        
                        for i in range(m*p):
                            row = []
                            for r_new, r_old in enumerate(col_map):
                                row.append(V28[i, r_old])
                            V_new.append(row)
                        
                        for i in range(p*n):
                            row = []
                            for r_new, r_old in enumerate(col_map):
                                row.append(W28[i, r_old])
                            W_new.append(row)
                        
                        if validate(U_new, V_new, W_new, new_rank):
                            if check_half_integer_constraint(U_new, V_new, W_new):
                                return {"U": U_new, "V": V_new, "W": W_new}
        
        return None

    best_result = None
    best_rank = 999

    # Try rank 27
    res27 = try_rank_27_search()
    if res27:
        best_result = res27
        best_rank = 27
    else:
        # Fallback to rank 28
        res28 = build_strassen_28()
        if validate(res28["U"], res28["V"], res28["W"], 28):
            best_result = res28
            best_rank = 28
        else:
            # Fallback to rank 32
            res32 = build_schoolbook()
            if validate(res32["U"], res32["V"], res32["W"], 32):
                best_result = res32
                best_rank = 32
            else:
                raise RuntimeError("Failed to find valid decomposition")

    return best_result

# EVOLVE-BLOCK-END
