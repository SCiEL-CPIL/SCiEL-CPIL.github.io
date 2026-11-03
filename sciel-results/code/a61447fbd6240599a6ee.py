"""
Rewritten program for Matrix Multiplication 2x4x5.
Implements a Rank-36 decomposition using Strassen blocks for 2x2x2 parts
and schoolbook for 2x2x1 parts, constructed via vectorized NumPy operations.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback: Schoolbook Rank 40
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, idx] = 1
                V_school[j*p + k, idx] = 1
                W_school[k*n + i, idx] = 1
                idx += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Target Rank 36
    # Decomposition:
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x5) = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    
    # For each 2x2 * 2x5 block:
    # Split 2x5 into 2x2, 2x2, 2x1
    # Rank(2x2 * 2x2) = 7 (Strassen)
    # Rank(2x2 * 2x1) = 4 (Schoolbook)
    # Total per 2x2*2x5 = 11 * 2 + 4 = 26? No.
    # Wait. 2x2 * 2x5.
    # B = [B_col0, B_col1, B_col2, B_col3, B_col4]
    # C_col0 = A * B_col0
    # C_col1 = A * B_col1
    # ...
    # If we group B_cols (0,1) into a 2x2 block, we use Strassen (Rank 7) for those 2 columns of C.
    # If we group B_cols (2,3) into a 2x2 block, we use Strassen (Rank 7) for those 2 columns of C.
    # If we take B_col4 (2x1), we use Schoolbook (Rank 4) for that 1 column of C.
    # Total rank for one A(2x2)*B(2x5) block is 7 + 7 + 4 = 18.
    # We have two such blocks (A1*B1 and A2*B2).
    # Total Rank = 18 + 18 = 36.

    total_rank = 36
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))

    # Strassen 2x2x2 Patterns
    # Indices for 2x2 matrix flattened row-major: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U (A side), V (B side), W (C side)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C00 -= M2 ? 
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7

    # Let's define the 7 rank-1 terms for a generic 2x2x2 block
    # U_vec, V_vec, W_vec are of length 4 (representing the 4 elements of the 2x2 matrix)
    strassen_terms = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
        # M4
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        # M6
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
    ]
    
    # Schoolbook 2x2x1 Patterns (for 2x2 A * 2x1 B -> 2x1 C)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Terms:
    # 1. A00*B0 -> C0
    # 2. A01*B1 -> C0
    # 3. A10*B0 -> C1
    # 4. A11*B1 -> C1
    # U_vec (len 4), V_vec (len 2), W_vec (len 2)
    schoolbook_2x2x1_terms = [
        (np.array([1, 0, 0, 0]), np.array([1, 0]), np.array([1, 0])),
        (np.array([0, 1, 0, 0]), np.array([0, 1]), np.array([1, 0])),
        (np.array([0, 0, 1, 0]), np.array([1, 0]), np.array([0, 1])),
        (np.array([0, 0, 0, 1]), np.array([0, 1]), np.array([0, 1])),
    ]

    def add_block_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, col_start):
        """
        Adds Strassen decomposition for A[a_row_off:a_row_off+2, a_col_off:a_col_off+2] * 
        B[b_row_off:b_row_off+2, b_col_off:b_col_off+2] contributing to 
        C[c_row_off:c_row_off+2, c_col_off:c_col_off+2]
        """
        col = col_start
        for u_pat, v_pat, w_pat in strassen_terms:
            # U: 2x2 block in A
            # Flatten u_pat to indices
            for local_i in range(2):
                for local_j in range(2):
                    val = u_pat[local_i * 2 + local_j]
                    if val != 0:
                        global_i = a_row_off + local_i
                        global_j = a_col_off + local_j
                        U[global_i * m + global_j, col] += val
            
            # V: 2x2 block in B
            for local_i in range(2):
                for local_j in range(2):
                    val = v_pat[local_i * 2 + local_j]
                    if val != 0:
                        global_i = b_row_off + local_i
                        global_j = b_col_off + local_j
                        V[global_i * p + global_j, col] += val
            
            # W: 2x2 block in C
            # W index is [k*n + i]
            for local_i in range(2):
                for local_j in range(2):
                    val = w_pat[local_i * 2 + local_j]
                    if val != 0:
                        global_i = c_row_off + local_i
                        global_j = c_col_off + local_j
                        W[global_j * n + global_i, col] += val
            
            col += 1
        return col

    def add_block_2x2x1(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, col_start):
        """
        Adds Schoolbook decomposition for A[2x2] * B[2x1] contributing to C[2x1]
        """
        col = col_start
        for u_pat, v_pat, w_pat in schoolbook_2x2x1_terms:
            # U: 2x2 block in A
            for local_i in range(2):
                for local_j in range(2):
                    val = u_pat[local_i * 2 + local_j]
                    if val != 0:
                        global_i = a_row_off + local_i
                        global_j = a_col_off + local_j
                        U[global_i * m + global_j, col] += val
            
            # V: 2x1 block in B
            for local_i in range(2):
                val = v_pat[local_i]
                if val != 0:
                    global_i = b_row_off + local_i
                    global_j = b_col_off
                    V[global_i * p + global_j, col] += val
            
            # W: 2x1 block in C
            for local_i in range(2):
                val = w_pat[local_i]
                if val != 0:
                    global_i = c_row_off + local_i
                    global_j = c_col_off
                    W[global_j * n + global_i, col] += val
            
            col += 1
        return col

    # Construct the decomposition
    col = 0
    
    # Block A1 (rows 0-1, cols 0-1 of A) * B1 (rows 0-1 of B)
    # B1 is 2x5. Split into cols 0-1 (2x2), 2-3 (2x2), 4 (2x1)
    col = add_block_2x2x2(0, 0, 0, 0, 0, 0, col) # A1 * B1[:, 0:2] -> C[:, 0:2]
    col = add_block_2x2x2(0, 0, 0, 2, 0, 2, col) # A1 * B1[:, 2:4] -> C[:, 2:4]
    col = add_block_2x2x1(0, 0, 0, 4, 0, 4, col) # A1 * B1[:, 4]   -> C[:, 4]
    
    # Block A2 (rows 0-1, cols 2-3 of A) * B2 (rows 2-3 of B)
    # B2 is 2x5. Split into cols 0-1 (2x2), 2-3 (2x2), 4 (2x1)
    col = add_block_2x2x2(0, 2, 2, 0, 0, 0, col) # A2 * B2[:, 0:2] -> C[:, 0:2]
    col = add_block_2x2x2(0, 2, 2, 2, 0, 2, col) # A2 * B2[:, 2:4] -> C[:, 2:4]
    col = add_block_2x2x1(0, 2, 2, 4, 0, 4, col) # A2 * B2[:, 4]   -> C[:, 4]
    
    if col != total_rank:
        raise RuntimeError(f"Rank mismatch: expected {total_rank}, got {col}")

    # Verification
    def verify_decomposition(U_mat, V_mat, W_mat):
        # Convert to numpy arrays for fast dot products
        U_arr = np.array(U_mat)
        V_arr = np.array(V_mat)
        W_arr = np.array(W_mat)
        
        # Diagonal check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over rank r: U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    s = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Off-diagonal check
        # We check a random subset or full set if fast enough
        # nm=8, mp=20, pn=10.
        # For each (u, v, w) not on diagonal, sum should be 0.
        # Diagonal defined by j_u == j_v and k_v == k_w and i_u == i_w
        for u in range(nm):
            i_u = u // m
            j_u = u % m
            for v in range(mp):
                j_v = v // p
                k_v = v % p
                if j_u != j_v:
                    continue
                for w in range(pn):
                    k_w = w // n
                    i_w = w % n
                    if k_v != k_w or i_u != i_w:
                        s = np.sum(U_arr[u, :] * V_arr[v, :] * W_arr[w, :])
                        if abs(s) > 1e-9:
                            return False
        return True

    if time.time() < deadline:
        # Convert to lists for return
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Clean up floating point noise (should be integers or 0.5)
        # Round to nearest 0.5
        def clean_val(x):
            return round(x * 2) / 2
        
        U_clean = [[clean_val(x) for x in row] for row in U_list]
        V_clean = [[clean_val(x) for x in row] for row in V_list]
        W_clean = [[clean_val(x) for x in row] for row in W_list]
        
        # Verify cleaned version to be sure
        if verify_decomposition(U_clean, V_clean, W_clean):
            return {"U": U_clean, "V": V_clean, "W": W_clean}
            
    return best_result

# EVOLVE-BLOCK-END
