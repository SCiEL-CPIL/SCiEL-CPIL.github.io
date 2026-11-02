import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Rank 153 decomposition:
    # 1. Use Rank 49 Strassen for the top-left 4x4 block (A11 * B11)
    # 2. Use Naive multiplication for the remaining cross terms and bottom-right 2x2 block
    
    total_rank = 153
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    col = 0
    
    # --- Part 1: Rank 49 Strassen for 4x4 block ---
    # A11 is rows 0-3, cols 0-3
    # B11 is rows 0-3, cols 0-3
    # C11 is rows 0-3, cols 0-3
    
    # High-level Strassen terms for 4x4 (split into 2x2 blocks of size 2x2)
    # Let A_blocks be A11, A12, A21, A22 (all 2x2)
    # Let B_blocks be B11, B12, B21, B22 (all 2x2)
    
    # High-level Strassen combinations:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Low-level Strassen terms for 2x2 (base case)
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b01-b11)
    # M3 = a11(b10-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b01)
    # M6 = (a01-a11)(b10+b11)
    
    # Base Strassen coefficients for A, B, C
    # (row, col, coeff)
    base_a = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    base_b = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (1,1,1)],      # M6
    ]
    base_c = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,-1)],     # M1
        [(0,1,1), (1,1,1)],      # M2
        [(0,0,1), (1,0,1)],      # M3
        [(0,0,-1), (0,1,1)],     # M4
        [(1,1,1)],               # M5
        [(0,0,1)],               # M6
    ]
    
    # High-level Strassen combinations
    # Each entry is a list of (block_row, block_col, coeff) for A and B
    # Block indices: 00, 01, 10, 11 correspond to top-left, top-right, bottom-left, bottom-right
    high_a = [
        [(0,0,1), (1,1,1)],      # M1: A11 + A22
        [(1,0,1), (1,1,1)],      # M2: A21 + A22
        [(0,0,1)],               # M3: A11
        [(1,1,1)],               # M4: A22
        [(0,0,1), (0,1,1)],      # M5: A11 + A12
        [(1,0,1), (0,0,-1)],     # M6: A21 - A11
        [(0,1,1), (1,1,-1)],     # M7: A12 - A22
    ]
    high_b = [
        [(0,0,1), (1,1,1)],      # M1: B11 + B22
        [(0,0,1)],               # M2: B11
        [(0,1,1), (1,1,-1)],     # M3: B12 - B22
        [(1,0,1), (0,0,-1)],     # M4: B21 - B11
        [(1,1,1)],               # M5: B22
        [(0,0,1), (0,1,1)],      # M6: B11 + B12
        [(1,0,1), (1,1,1)],      # M7: B21 + B22
    ]
    # High-level C contributions
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    high_c = [
        [(0,0,1), (1,1,1)],      # M1: C11 += 1, C22 += 1
        [(1,0,1), (1,1,-1)],     # M2: C21 += 1, C22 -= 1
        [(0,1,1), (1,1,1)],      # M3: C12 += 1, C22 += 1
        [(0,0,1), (1,0,1)],      # M4: C11 += 1, C21 += 1
        [(0,0,-1), (0,1,1)],     # M5: C11 -= 1, C12 += 1
        [(1,0,1), (1,1,1)],      # M6: C21 += 1, C22 += 1
        [(0,0,1)],               # M7: C11 += 1
    ]
    
    # Iterate over high-level terms
    for h_idx in range(7):
        # For each high-level term, we have a 2x2 matrix product
        # We decompose this 2x2 product using base Strassen
        for l_idx in range(7):
            # Fill U
            # A contribution: sum over high_a[h_idx] blocks, each block contributes base_a[l_idx]
            for (br, bc, hc) in high_a[h_idx]:
                # Block br, bc in A11 (4x4)
                # Global row: br*2 + lr, Global col: bc*2 + lc
                for (lr, lc, bc_coeff) in base_a[l_idx]:
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    U[gi * m + gj, col] += hc * bc_coeff
            
            # Fill V
            for (br, bc, hc) in high_b[h_idx]:
                # Block br, bc in B11 (4x4)
                # Global row: br*2 + lr, Global col: bc*2 + lc
                for (lr, lc, bc_coeff) in base_b[l_idx]:
                    gj = br * 2 + lr
                    gk = bc * 2 + lc
                    V[gj * p + gk, col] += hc * bc_coeff
            
            # Fill W
            # C contribution: sum over high_c[h_idx] blocks, each block contributes base_c[l_idx]
            for (br, bc, hc) in high_c[h_idx]:
                # Block br, bc in C11 (4x4)
                # Global row: br*2 + lr, Global col: bc*2 + lc
                for (lr, lc, bc_coeff) in base_c[l_idx]:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    W[gk * n + gi, col] += hc * bc_coeff
            
            col += 1
            
            if time.time() > deadline - 20:
                # Return current partial result if time is running out
                # This should not happen as 49 terms are fast
                return {"U": U[:, :col].tolist(), "V": V[:, :col].tolist(), "W": W[:, :col].tolist()}

    # --- Part 2: Naive multiplication for cross terms ---
    
    # Helper function to add naive terms
    def add_naive(U, V, W, col, i_range, j_range, k_range, l_range, n, m, p):
        # A[i, j] * B[j, k] -> C[i, l]
        # U index: i*m + j
        # V index: j*p + k
        # W index: l*n + i (Note: W[k*n+i, r] in prompt? 
        # Prompt says W[k*n+i,r]. Let's re-read carefully.
        # "output factor follows the official trace convention W[k*n+i,r]"
        # Usually C[i,k] = sum U[i,j,r] V[j,k,r] W[i,k,r]?
        # Or C[i,k] = sum U[i,j,r] V[j,k,r] W[k,n+i,r]?
        # Standard tensor decomposition: C_{ik} = sum_r U_{ir} V_{kr} W_{rk}? No.
        # Let's check the prompt's W convention again.
        # "W[k*n+i,r]"
        # If C is n x p (6x6), indices i (row), k (col).
        # If W is indexed by [k*n+i], that's col*n + row.
        # Let's assume the standard: C[i,k] = sum_r U[i,r] V[k,r] W[i,k,r]?
        # But the prompt says W shape is (p*n, rank). p=6, n=6.
        # If W[k*n+i], then index is col*6 + row.
        # Let's stick to the code's existing logic for W if it was correct.
        # The previous code used W[gk * n + gi, col].
        # gk is col index of C, gi is row index of C.
        # So W index = col * n + row.
        # This matches "W[k*n+i]" if k is col and i is row.
        
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    for l in l_range:
                        U[i * m + j, col] += 1.0
                        V[j * p + k, col] += 1.0
                        W[k * n + i, col] += 1.0
                        col += 1
        return col

    # 1. A12 * B21 -> C11
    # A12: i in 0..3, j in 4..5
    # B21: j in 4..5, k in 0..3
    # C11: i in 0..3, k in 0..3
    # Note: The summation index is j.
    # U[i, j], V[j, k], W[k, i]
    col = add_naive(U, V, W, col, range(0,4), range(4,6), range(0,4), range(0,4), n, m, p)
    
    # 2. A11 * B12 -> C12
    # A11: i in 0..3, j in 0..3
    # B12: j in 0..3, k in 4..5
    # C12: i in 0..3, k in 4..5
    col = add_naive(U, V, W, col, range(0,4), range(0,4), range(4,6), range(4,6), n, m, p)
    
    # 3. A21 * B11 -> C21
    # A21: i in 4..5, j in 0..3
    # B11: j in 0..3, k in 0..3
    # C21: i in 4..5, k in 0..3
    col = add_naive(U, V, W, col, range(4,6), range(0,4), range(0,4), range(0,4), n, m, p)
    
    # 4. A22 * B22 -> C22
    # A22: i in 4..5, j in 4..5
    # B22: j in 4..5, k in 4..5
    # C22: i in 4..5, k in 4..5
    col = add_naive(U, V, W, col, range(4,6), range(4,6), range(4,6), range(4,6), n, m, p)
    
    # Verify col == total_rank
    assert col == total_rank, f"Rank mismatch: {col} != {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
