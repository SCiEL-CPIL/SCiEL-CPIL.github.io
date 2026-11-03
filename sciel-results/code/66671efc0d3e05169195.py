import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to add a Strassen block
    # A is 2x2 block at (row_offset, col_offset) in A (2x4)
    # B is 2x2 block at (row_offset, col_offset) in B (4x4)
    # C is 2x2 block at (row_offset, col_offset) in C (2x4)
    def add_strassen_block(col_start, a_row, a_col, b_row, b_col, c_row, c_col):
        # Indices for A (2x2 block)
        # A is 2x4. Index i*m + j.
        # a_row is 0 or 1 (row in 2x4 A)
        # a_col is 0, 1, 2, 3 (col in 2x4 A)
        # The 2x2 block covers rows a_row, a_row+1? No, A is only 2 rows high.
        # Wait, the Strassen block operates on 2x2 matrices.
        # A is 2x4. We split columns into 2x2 blocks.
        # So A_block is rows 0,1 and cols a_col, a_col+1.
        # B is 4x4. We split rows and cols into 2x2 blocks.
        # B_block is rows b_row, b_row+1 and cols b_col, b_col+1.
        # C is 2x4. We split columns into 2x2 blocks.
        # C_block is rows c_row, c_row+1? No, C is 2 rows high.
        # C_block is rows 0,1 and cols c_col, c_col+1.
        
        # Let's re-map indices carefully.
        # U index: i*m + j. i in [0,1], j in [0,3].
        # V index: j*p + k. j in [0,3], k in [0,3].
        # W index: k*n + i. k in [0,3], i in [0,1].
        
        # A indices for the 2x2 block:
        # Local (0,0) -> Global (a_row, a_col)
        # Local (0,1) -> Global (a_row, a_col+1)
        # Local (1,0) -> Global (a_row+1, a_col)  -- Wait, a_row+1 must be < 2. So a_row must be 0.
        # Actually, A is 2x4. The "2x2" blocks in Strassen for 2x4x4 decomposition 
        # are formed by taking 2 rows of A (which is all of A) and 2 cols of A.
        # So a_row is always 0? No, the Strassen algorithm on 2x2 matrices 
        # assumes the matrix is 2x2. Here we are multiplying a 2x4 by 4x4.
        # We decomposed it into 4 products of 2x2 by 2x2.
        # Product 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2] part 1
        # Product 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2] part 2
        # Product 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4] part 1
        # Product 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4] part 2
        
        # So for each of these 4 products, we use the 7 Strassen multiplications.
        
        # A block cols: a_col, a_col+1
        # B block rows: b_row, b_row+1; cols: b_col, b_col+1
        # C block cols: c_col, c_col+1 (rows are 0,1)
        
        # U indices (A part)
        # Local A[0,0] -> Global A[0, a_col]
        # Local A[0,1] -> Global A[0, a_col+1]
        # Local A[1,0] -> Global A[1, a_col]
        # Local A[1,1] -> Global A[1, a_col+1]
        
        u_00 = 0 * m + a_col
        u_01 = 0 * m + (a_col + 1)
        u_10 = 1 * m + a_col
        u_11 = 1 * m + (a_col + 1)
        
        # V indices (B part)
        # Local B[0,0] -> Global B[b_row, b_col]
        # Local B[0,1] -> Global B[b_row, b_col+1]
        # Local B[1,0] -> Global B[b_row+1, b_col]
        # Local B[1,1] -> Global B[b_row+1, b_col+1]
        
        v_00 = b_row * p + b_col
        v_01 = b_row * p + (b_col + 1)
        v_10 = (b_row + 1) * p + b_col
        v_11 = (b_row + 1) * p + (b_col + 1)
        
        # W indices (C part)
        # Local C[0,0] -> Global C[0, c_col] -> W[c_col * n + 0]
        # Local C[0,1] -> Global C[0, c_col+1] -> W[(c_col+1) * n + 0]
        # Local C[1,0] -> Global C[1, c_col] -> W[c_col * n + 1]
        # Local C[1,1] -> Global C[1, c_col+1] -> W[(c_col+1) * n + 1]
        
        w_00 = c_col * n + 0
        w_01 = (c_col + 1) * n + 0
        w_10 = c_col * n + 1
        w_11 = (c_col + 1) * n + 1
        
        # Strassen Coefficients for 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Output:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        terms = [
            # M1
            {'U': [(u_00, 1), (u_11, 1)], 'V': [(v_00, 1), (v_11, 1)], 'W': [(w_00, 1), (w_11, 1)]},
            # M2
            {'U': [(u_10, 1), (u_11, 1)], 'V': [(v_00, 1)], 'W': [(w_10, 1), (w_11, -1)]},
            # M3
            {'U': [(u_00, 1)], 'V': [(v_01, 1), (v_11, -1)], 'W': [(w_01, 1), (w_11, 1)]},
            # M4
            {'U': [(u_11, 1)], 'V': [(v_10, 1), (v_00, -1)], 'W': [(w_00, 1), (w_10, 1)]},
            # M5
            {'U': [(u_00, 1), (u_01, 1)], 'V': [(v_11, 1)], 'W': [(w_00, -1), (w_01, 1)]},
            # M6
            {'U': [(u_10, 1), (u_00, -1)], 'V': [(v_00, 1), (v_01, 1)], 'W': [(w_11, 1)]},
            # M7
            {'U': [(u_01, 1), (u_11, -1)], 'V': [(v_10, 1), (v_11, 1)], 'W': [(w_00, 1)]}
        ]
        
        for t in terms:
            col = col_start
            for idx, val in t['U']:
                U[idx, col] += val
            for idx, val in t['V']:
                V[idx, col] += val
            for idx, val in t['W']:
                W[idx, col] += val
            col_start += 1

    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # a_col=0, b_row=0, b_col=0, c_col=0
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # a_col=2, b_row=2, b_col=0, c_col=0
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # a_col=0, b_row=0, b_col=2, c_col=2
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # a_col=2, b_row=2, b_col=2, c_col=2
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)

    # Verification and potential reduction
    def verify(U_arr, V_arr, W_arr, rank):
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    T_recon[i, j, k] = s
        return np.max(np.abs(T_recon - 1.0)) < 1e-9

    if not verify(U, V, W, rank):
        # Fallback to naive if Strassen construction fails (should not happen)
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1

    # Attempt greedy rank reduction (low chance of success, but included per directive)
    start_time = time.time()
    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    # Only try if we have time
    if time.time() - start_time < 10:
        for _ in range(50): # Limited iterations
            if time.time() - start_time > 110: break
            removed = False
            for r in range(current_rank):
                # Skip if time is running out
                if time.time() - start_time > 115: break
                
                # Try removing column r
                mask = [i for i in range(current_rank) if i != r]
                new_U = current_U[:, mask]
                new_V = current_V[:, mask]
                new_W = current_W[:, mask]
                
                if verify(new_U, new_V, new_W, len(mask)):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank = len(mask)
                    removed = True
                    break
            if not removed:
                break # If no term could be removed in this pass, stop

    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
