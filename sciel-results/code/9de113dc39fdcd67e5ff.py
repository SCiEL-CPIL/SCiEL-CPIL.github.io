"""Rank-28 exact tensor decomposition for 2x4 by 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-28 exact tensor decomposition for (2x4)*(4x4) -> (2x4).
    
    Structure:
      A is 2x4, B is 4x4, C is 2x4.
      Split A into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3).
      Split B into four 2x2 blocks:
        B[0:2, 0:2] = B00, B[0:2, 2:4] = B01
        B[2:4, 0:2] = B10, B[2:4, 2:4] = B11
      C_left = A_left * B00 + A_right * B10  (2x2)
      C_right = A_left * B01 + A_right * B11 (2x2)
    
    Each 2x2*2x2 uses Strassen (rank 7). Total rank = 4*7 = 28.
    
    Index conventions:
      U[i, t]: i = row_A * 4 + col_A  (0..7),  A is 2x4
      V[j, t]: j = row_B * 4 + col_B  (0..15), B is 4x4
      W[k, t]: k = col_C * 2 + row_C  (0..7),  C is 2x4, W[k*n+i,r] with n=2
                So W index = col_C * 2 + row_C.
    
    Strassen for 2x2 * 2x2:
      A = [[a, b], [c, d]], B = [[e, f], [g, h]]
      p1 = (a+d)(e+h)
      p2 = (c+d)e
      p3 = a(f-h)
      p4 = d(g-e)
      p5 = (a+b)h
      p6 = (c-a)(e+g)
      p7 = (b-d)(f+h)
      
      C00 = p1 + p4 - p5 + p7
      C01 = p3 + p5
      C10 = p2 + p4
      C11 = p1 - p2 + p3 + p6
    
    For each Strassen term t in a block, we need:
      U contributions: which A entries participate
      V contributions: which B entries participate
      W contributions: which C entries receive the result (with sign)
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 8 x 28
    
    # Strassen term definitions for 2x2*2x2:
    # Each term: (U_entries, V_entries, W_entries)
    # U_entries: dict {local_A_index: coeff}
    # V_entries: dict {local_B_index: coeff}
    # W_entries: dict {local_C_index: coeff}
    # Local indices: A: 00,01,10,11 -> 0,1,2,3
    #                B: 00,01,10,11 -> 0,1,2,3
    #                C: 00,01,10,11 -> 0,1,2,3 (C[row,col])
    
    # p1 = (a+d)(e+h): A has 00,11; B has 00,11; C: 00(+1), 11(+1)
    # p2 = (c+d)e: A has 10,11; B has 00; C: 10(+1), 11(-1)
    # p3 = a(f-h): A has 00; B has 01,11; C: 01(+1), 11(+1)
    # p4 = d(g-e): A has 11; B has 10,00; C: 00(+1), 10(+1)
    # p5 = (a+b)h: A has 00,01; B has 11; C: 00(-1), 01(+1)
    # p6 = (c-a)(e+g): A has 10,00; B has 00,10; C: 11(+1)
    # p7 = (b-d)(f+h): A has 01,11; B has 01,11; C: 00(+1)
    
    strassen_terms = [
        # (U_dict, V_dict, W_dict)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # p1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # p2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # p3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # p4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # p5
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {3: 1}),            # p6
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {0: 1}),            # p7
    ]
    
    # Four blocks:
    # Block 0: A_left * B00 -> C_left (C rows 0,1; C cols 0,1)
    #   A cols: 0,1; B rows: 0,1, cols: 0,1; C: row 0-1, col 0-1
    # Block 1: A_right * B10 -> C_left (C rows 0,1; C cols 0,1)
    #   A cols: 2,3; B rows: 2,3, cols: 0,1; C: row 0-1, col 0-1
    # Block 2: A_left * B01 -> C_right (C rows 0,1; C cols 2,3)
    #   A cols: 0,1; B rows: 0,1, cols: 2,3; C: row 0-1, col 2-3
    # Block 3: A_right * B11 -> C_right (C rows 0,1; C cols 2,3)
    #   A cols: 2,3; B rows: 2,3, cols: 2,3; C: row 0-1, col 2-3
    
    # For each block:
    #   a_row_offset: row offset in A (always 0 since A has 2 rows)
    #   a_col_offset: column offset in A (0 or 2)
    #   b_row_offset: row offset in B (0 or 2)
    #   b_col_offset: col offset in B (0 or 2)
    #   c_col_offset: col offset in C (0 or 2)
    
    blocks = [
        # (a_col_off, b_row_off, b_col_off, c_col_off)
        (0, 0, 0, 0),   # Block 0: A_left * B00 -> C_left
        (2, 2, 0, 0),   # Block 1: A_right * B10 -> C_left
        (0, 0, 2, 2),   # Block 2: A_left * B01 -> C_right
        (2, 2, 2, 2),   # Block 3: A_right * B11 -> C_right
    ]
    
    for b_idx, (a_co, b_ro, b_co, c_co) in enumerate(blocks):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
            t = b_idx * 7 + t_idx
            
            # Fill U: A index = row * 4 + col
            for loc_a, coeff in u_dict.items():
                row_a = loc_a // 2  # 0 or 1
                col_a = loc_a % 2   # 0 or 1
                global_a = row_a * 4 + (col_a + a_co)
                U[global_a, t] += coeff
            
            # Fill V: B index = row * 4 + col
            for loc_b, coeff in v_dict.items():
                row_b = loc_b // 2  # 0 or 1
                col_b = loc_b % 2   # 0 or 1
                global_b = (row_b + b_ro) * 4 + (col_b + b_co)
                V[global_b, t] += coeff
            
            # Fill W: C index = col_C * 2 + row_C
            for loc_c, coeff in w_dict.items():
                row_c = loc_c // 2  # 0 or 1
                col_c = loc_c % 2   # 0 or 1
                global_c = (col_c + c_co) * 2 + row_c
                W[global_c, t] += coeff
    
    # Verify with random test
    np.random.seed(42)
    A_test = np.random.randn(2, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition:
    # C[k*n+i] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    # More precisely: C = sum_t outer(W[:,t], ...) 
    # Actually: T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    # C[k,i] = sum_{i,j} A[i,j] B[j,k] T[i,j,k]
    #         = sum_t W[k,i,t] * (sum_j U[i,j,t] A[i,j]) * (sum_j V[j,k,t] B[j,k])
    
    C_computed = np.zeros((2, 4))
    for t in range(rank):
        # For each output position (row_c, col_c):
        # W index = col_c * 2 + row_c
        # The term contributes: W[col_c*2+row_c, t] * (sum over A entries) * (sum over B entries)
        # But we need to be careful: the tensor decomposition gives
        # C[row_c, col_c] = sum_t W[col_c*2+row_c, t] * sum_{i,j} U[i*4+j, t] * A[i,j] * V[j*4+col_c, t] * B[j, col_c]
        # Wait, that's not right either. Let me think more carefully.
        
        # The tensor is T[i, j, k] where i indexes A (2*4=8), j indexes B (4*4=16), k indexes C (4*2=8).
        # T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
        # C[k] = sum_{i,j} A[i] * B[j] * T[i,j,k]
        #      = sum_t W[k,t] * sum_i (U[i,t]*A[i]) * sum_j (V[j,t]*B[j])
        
        a_proj = U[:, t] @ A_test.ravel()  # scalar: sum_i U[i,t]*A[i]
        b_proj = V[:, t] @ B_test.ravel()  # scalar: sum_j V[j,t]*B[j]
        for k in range(8):
            C_computed.ravel()[k] += W[k, t] * a_proj * b_proj
    
    # Check
    max_err = np.max(np.abs(C_computed - C_expected))
    if max_err > 1e-10:
        # Fallback: schoolbook rank 32
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
