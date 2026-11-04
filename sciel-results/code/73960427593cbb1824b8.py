"""Tensor decomposition of 5x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 110.0  # safety margin before 120s limit

    n = 5
    m = 5
    p = 5

    U_cols = []
    V_cols = []
    W_cols = []

    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        if u_val == 0 or v_val == 0 or w_val == 0:
            return
        U_cols.append({u_idx: u_val})
        V_cols.append({v_idx: v_val})
        W_cols.append({w_idx: w_val})

    # --- Part 1: 3x3 Core (Laderman's Algorithm, Rank 23) ---
    # Indices for 3x3 block (0..2)
    # A[i,j], B[j,k], C[i,k]
    # Laderman's 23 products for 3x3 matrix multiplication
    
    # Define the 23 terms (u_idx, v_idx, w_idx, coeff)
    # u_idx = i*3+j, v_idx = j*3+k, w_idx = k*3+i
    # Note: The global indices for 5x5 will be mapped later.
    # For now, we generate local indices and map them.
    
    laderman_terms = [
        # (i, j, k, coeff) for U, (j, k, i, coeff) for V, (k, i, j, coeff) for W
        # M1: A00 B00
        ((0,0), (0,0), (0,0), 1),
        # M2: A11 B11
        ((1,1), (1,1), (1,1), 1),
        # M3: A22 B22
        ((2,2), (2,2), (2,2), 1),
        # M4: (A00+A11) B00
        ((0,0), (0,0), (0,0), 1), # Wait, Laderman is specific.
        # I will use a verified list of 23 terms for 3x3.
        # Source: https://en.wikipedia.org/wiki/Laderman%27s_algorithm
        # The algorithm uses 23 multiplications.
        
        # Let's use the standard 23-term decomposition.
        # M1 = A00 B00
        # M2 = A11 B11
        # M3 = A22 B22
        # M4 = (A00+A11) B00
        # M5 = (A00+A11) B11
        # M6 = (A01+A10) B01
        # M7 = (A01+A10) B10
        # M8 = (A02+A12) B02
        # M9 = (A02+A12) B20
        # M10 = (A10+A20) B10
        # M11 = (A10+A20) B20
        # M12 = (A12+A22) B12
        # M13 = (A12+A22) B21
        # M14 = (A20+A02) B20
        # M15 = (A20+A02) B01
        # M16 = (A21+A10) B21
        # M17 = (A21+A10) B10
        # M18 = (A22+A00) B22
        # M19 = (A22+A00) B00
        # M20 = (A01+A12+A22) B01
        # M21 = (A10+A21+A02) B10
        # M22 = (A12+A20+A01) B21
        # M23 = (A21+A00+A11) B20
    ]
    
    # Correct Laderman 23 terms (coefficients for U, V, W)
    # U[i,j], V[j,k], W[k,i]
    # Term: sum c_r * U_r[i,j] V_r[j,k] W_r[k,i]
    
    # I will construct the 23 terms explicitly.
    # This is a known decomposition.
    # 
    # M1: A00 B00 -> C00
    # M2: A11 B11 -> C11
    # M3: A22 B22 -> C22
    # M4: (A00+A11) B00
    # M5: (A00+A11) B11
    # M6: (A01+A10) B01
    # M7: (A01+A10) B10
    # M8: (A02+A12) B02
    # M9: (A02+A12) B20
    # M10: (A10+A20) B10
    # M11: (A10+A20) B20
    # M12: (A12+A22) B12
    # M13: (A12+A22) B21
    # M14: (A20+A02) B20
    # M15: (A20+A02) B01
    # M16: (A21+A10) B21
    # M17: (A21+A10) B10
    # M18: (A22+A00) B22
    # M19: (A22+A00) B00
    # M20: (A01+A12+A22) B01
    # M21: (A10+A21+A02) B10
    # M22: (A12+A20+A01) B21
    # M23: (A21+A00+A11) B20
    
    # Let's verify one: C00 = M1 + M4 - M5 - M19?
    # C00 = A00B00 + A01B10 + A02B20
    # M1 = A00B00
    # M4 = (A00+A11)B00 = A00B00 + A11B00
    # M5 = (A00+A11)B11 = A00B11 + A11B11
    # M19 = (A22+A00)B00 = A22B00 + A00B00
    # This doesn't look right. I will use the standard 23-term list from literature.
    
    # Actually, to avoid bugs in hardcoding 23 complex terms, I will use the 
    # Kronecker product of Strassen (2x2) and a 1x1 term? No, that gave 24.
    # I will use the 24-term decomposition for 3x3 derived from 2+1 split.
    # It is safer to implement the 2+1 split for 3x3 (Rank 24) than risk a bug in Laderman 23.
    # 24 is still much better than 125.
    # Total rank will be 24 + 7 + 9 + 9 + 9 + 9 = 67.
    
    # --- Re-Plan: Use 2+1 split for 3x3 (Rank 24) ---
    # A33 = [A22 A21; A12 A11] where A22 is 2x2, A11 is 1x1, etc.
    # Actually, split 3 into 2+1.
    # Blocks: 0,1 (2x2) and 2 (1x1).
    # C00 (2x2) = A00 B00 + A01 B10
    # C01 (2x1) = A00 B01 + A01 B11
    # C10 (1x2) = A10 B00 + A11 B10
    # C11 (1x1) = A10 B01 + A11 B11
    
    # R(A00 B00) = 7 (Strassen)
    # R(A01 B10) = R(2x1 x 1x2) = 2? No, 2x1 x 1x2 is 2x1x2? No.
    # A01 is 2x1. B10 is 1x2. Product is 2x2.
    # R(2x1 x 1x2) = 2?
    # C_ij = A_i1 B_1j.
    # C_00 = A_01 B_10
    # C_01 = A_01 B_11
    # C_10 = A_11 B_10
    # C_11 = A_11 B_11
    # 4 multiplications.
    
    # R(A00 B01) = R(2x2 x 2x1) = 4?
    # C_ij = A_ik B_kj.
    # A is 2x2, B is 2x1.
    # C_00 = A_00 B_00 + A_01 B_10
    # C_10 = A_10 B_00 + A_11 B_10
    # 4 multiplications.
    
    # R(A10 B00) = R(1x2 x 2x2) = 4.
    # R(A11 B11) = R(1x1 x 1x1) = 1.
    
    # R(A10 B01) = R(1x2 x 2x1) = 2?
    # C_00 = A_00 B_00 + A_01 B_10.
    # 2 multiplications? No, 2 terms.
    
    # R(A01 B11) = R(2x1 x 1x1) = 2.
    # R(A10 B10) = R(1x1 x 1x2) = 2.
    
    # Total for 3x3:
    # C00: 7 (A00B00) + 4 (A01B10) = 11
    # C01: 4 (A00B01) + 2 (A01B11) = 6
    # C10: 4 (A10B00) + 2 (A10B01) = 6
    # C11: 1 (A11B11)
    # Total: 11 + 6 + 6 + 1 = 24.
    
    # So Rank(3x3) = 24.
    # Rank(2x2) = 7.
    # Rank(3x2x3) = 9.
    # Rank(2x3x2) = 9.
    # Rank(3x3x2) = 9.
    # Rank(2x3x3) = 9.
    
    # Total Rank = 24 + 7 + 9 + 9 + 9 + 9 = 67.

    # --- Implementation of 3x3 (Rank 24) ---
    # Indices: 0,1,2
    # Block 0: 2x2 (indices 0,1)
    # Block 1: 1x1 (index 2)
    
    # Strassen 2x2 terms
    def strassen_2x2(u_offset, v_offset, w_offset):
        # Returns list of (u_idx, u_val, v_idx, v_val, w_idx, w_val)
        # u_idx = i*2+j, v_idx = j*2+k, w_idx = k*2+i
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)(B00)
        # M3: (A00)(B10+B11)
        # M4: (A11)(B01-B11)
        # M5: (A00+A10)(B11)
        # M6: (A01-A11)(B00+A10)
        # M7: (A10-A00)(B01+A11)
        
        terms = []
        # M1
        terms.append(((0,0), 1, (0,0), 1, (0,0), 1))
        terms.append(((0,0), 1, (0,1), 1, (0,0), 0)) # Wait, Strassen is specific.
        
        # I will use the standard Strassen 7 terms for 2x2.
        # U, V, W for 2x2.
        # M1: U00=1, U11=1; V00=1, V11=1; W00=1, W11=1
        # M2: U10=1, U11=1; V00=1; W10=1, W11=0?
        
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10+B11)
        # M4 = A11(B01-B11)
        # M5 = (A00+A10)B11
        # M6 = (A01-A11)(B00+A10)
        # M7 = (A10-A00)(B01+A11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # I will implement this directly.
        pass

    # This is getting complex to hardcode. I will use the 66-rank Laderman if I can find a reliable list.
    # Given the time, I will implement the 67-rank solution using the 24-term 3x3 decomposition.
    
    # Helper to add Strassen 2x2 terms
    def add_strassen_2x2(base_i, base_j, base_k, scale=1):
        # Adds 7 terms for 2x2 multiplication at offsets
        # i,j,k are base indices for the 2x2 block
        # U index: (base_i + i, base_j + j)
        # V index: (base_j + j, base_k + k)
        # W index: (base_k + k, base_i + i)
        
        # M1
        add_term((base_i+0)*5+(base_j+0), scale, (base_j+0)*5+(base_k+0), scale, (base_k+0)*5+(base_i+0), scale)
        add_term((base_i+0)*5+(base_j+1), scale, (base_j+1)*5+(base_k+0), scale, (base_k+0)*5+(base_i+0), 0) # Placeholder
        # This is too error-prone.
        
        # I will use the 125-rank schoolbook if I can't guarantee correctness of the 67-rank.
        # But 67 is a huge improvement.
        
        # Let's try a simpler improvement: 4x4 (Rank 49) + 45 Frame = 94.
        # The current code claims 94 but fails.
        # I will fix the 4x4 Kronecker product.
        
    # --- Fixed 4x4 Kronecker Product ---
    # U2, V2, W2 for Strassen 2x2
    U2 = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0]
    ]
    
    # Generate 49 terms for 4x4
    for r1 in range(7):
        for r2 in range(7):
            u_col = {}
            v_col = {}
            w_col = {}
            
            for i2 in range(2):
                for j2 in range(2):
                    idx2_u = i2 * 2 + j2
                    u_val1 = U2[idx2_u][r1]
                    if u_val1 == 0: continue
                    for i2_p in range(2):
                        for j2_p in range(2):
                            idx2_u_p = i2_p * 2 + j2_p
                            u_val2 = U2[idx2_u_p][r2]
                            if u_val2 == 0: continue
                            u_val = u_val1 * u_val2
                            if u_val == 0: continue
                            i4 = 2 * i2 + i2_p
                            j4 = 2 * j2 + j2_p
                            idx4_u = i4 * 4 + j4
                            u_col[idx4_u] = u_col.get(idx4_u, 0) + u_val
                            
            for j2 in range(2):
                for k2 in range(2):
                    idx2_v = j2 * 2 + k2
                    v_val1 = V2[idx2_v][r1]
                    if v_val1 == 0: continue
                    for j2_p in range(2):
                        for k2_p in range(2):
                            idx2_v_p = j2_p * 2 + k2_p
                            v_val2 = V2[idx2_v_p][r2]
                            if v_val2 == 0: continue
                            v_val = v_val1 * v_val2
                            if v_val == 0: continue
                            j4 = 2 * j2 + j2_p
                            k4 = 2 * k2 + k2_p
                            idx4_v = j4 * 4 + k4
                            v_col[idx4_v] = v_col.get(idx4_v, 0) + v_val
                            
            for k2 in range(2):
                for i2 in range(2):
                    idx2_w = k2 * 2 + i2
                    w_val1 = W2[idx2_w][r1]
                    if w_val1 == 0: continue
                    for k2_p in range(2):
                        for i2_p in range(2):
                            idx2_w_p = k2_p * 2 + i2_p
                            w_val2 = W2[idx2_w_p][r2]
                            if w_val2 == 0: continue
                            w_val = w_val1 * w_val2
                            if w_val == 0: continue
                            k4 = 2 * k2 + k2_p
                            i4 = 2 * i2 + i2_p
                            idx4_w = k4 * 4 + i4
                            w_col[idx4_w] = w_col.get(idx4_w, 0) + w_val
                            
            # Embed 4x4 into 5x5 (indices 0..3)
            U5_col = {}
            V5_col = {}
            W5_col = {}
            
            for idx4_u, val in u_col.items():
                if val != 0:
                    i4, j4 = divmod(idx4_u, 4)
                    idx5_u = i4 * 5 + j4
                    U5_col[idx5_u] = val
                    
            for idx4_v, val in v_col.items():
                if val != 0:
                    j4, k4 = divmod(idx4_v, 4)
                    idx5_v = j4 * 5 + k4
                    V5_col[idx5_v] = val
                    
            for idx4_w, val in w_col.items():
                if val != 0:
                    k4, i4 = divmod(idx4_w, 4)
                    idx5_w = k4 * 5 + i4
                    W5_col[idx5_w] = val
                    
            U_cols.append(U5_col)
            V_cols.append(V5_col)
            W_cols.append(W5_col)

    # Add Frame terms (45)
    for ii in range(5):
        for jj in range(5):
            if ii == 4 or jj == 4:
                for kk in range(5):
                    U_cols.append({ii * 5 + jj: 1})
                    V_cols.append({jj * 5 + kk: 1})
                    W_cols.append({kk * 5 + ii: 1})

    rank = len(U_cols)
    
    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify
    try:
        import numpy as np
        rng = np.random.RandomState(12345)
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        for trial in range(100):
            if time.time() > tmax:
                break
            A = rng.randint(0, 6, size=(n, m))
            B = rng.randint(0, 6, size=(m, p))
            C_true = A @ B
            C_test = np.zeros((n, p), dtype=np.int64)
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                w_col = W_np[:, r]
                u_mat = u_col.reshape(n, m)
                v_mat = v_col.reshape(m, p)
                w_mat = w_col.reshape(p, n)
                temp = u_mat @ v_mat
                C_test += temp * w_mat.T
            if not np.array_equal(C_test, C_true):
                return _schoolbook()
    except ImportError:
        return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
