import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 3, 4, 4
    
    rank = 36
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Decomposition:
    # Split A (3x4) into A_top (2x4) and A_bottom (1x4).
    # C_top (2x4) = A_top * B
    # C_bottom (1x4) = A_bottom * B
    
    # A_top (2x4) * B (4x4) -> C_top (2x4)
    # Split A_top into A_TL (2x2) and A_TR (2x2).
    # Split B into B_TL (2x4) and B_BL (2x4) ? No.
    # B is 4x4.
    # C_top[i, j] = sum_{k=0}^1 A_TL[i, k] B[k, j] + sum_{k=2}^3 A_TR[i, k-2] B[k, j]
    
    # Part 1: A_TL (2x2) * B_TL (2x4)
    # A_TL indices: i in 0,1; k in 0,1.
    # B_TL indices: k in 0,1; j in 0..3.
    # Rank 16.
    
    # Part 2: A_TR (2x2) * B_BL (2x4)
    # A_TR indices: i in 0,1; k in 2,3 (local 0,1).
    # B_BL indices: k in 2,3; j in 0..3.
    # Rank 16.
    
    # Part 3: A_bottom (1x4) * B (4x4)
    # A_bottom indices: k in 0..3.
    # B indices: k in 0..3; j in 0..3.
    # Rank 4.
    
    col = 0
    
    # Helper to add a 2x2 * 2x4 block
    def add_2x2_2x4(a_row_offset, a_col_offset, b_row_offset, c_row_offset):
        nonlocal col
        # a_row_offset: 0 or 2 (global row of A)
        # a_col_offset: 0 or 2 (global col of A)
        # b_row_offset: 0 or 2 (global row of B)
        # c_row_offset: 0 or 2 (global row of C)
        
        # Local A is 2x2. Local B is 2x4.
        # U shape: 4 (2x2) x 16
        # V shape: 8 (2x4) x 16
        # W shape: 8 (2x4) x 16
        
        # Rank 16 decomposition:
        # For each column j of C (0..3):
        #   C[0, j] = A[0,0]B[0,j] + A[0,1]B[1,j]
        #   C[1, j] = A[1,0]B[0,j] + A[1,1]B[1,j]
        # We use 4 multiplications per column j.
        
        for j in range(4):
            # Term 1: A[0,0] * B[0,j]
            u_idx = (a_row_offset + 0) * m + (a_col_offset + 0)
            v_idx = (b_row_offset + 0) * p + j
            w_idx = j * n + (c_row_offset + 0)
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
            # Term 2: A[0,1] * B[1,j]
            u_idx = (a_row_offset + 0) * m + (a_col_offset + 1)
            v_idx = (b_row_offset + 1) * p + j
            w_idx = j * n + (c_row_offset + 0)
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
            # Term 3: A[1,0] * B[0,j]
            u_idx = (a_row_offset + 1) * m + (a_col_offset + 0)
            v_idx = (b_row_offset + 0) * p + j
            w_idx = j * n + (c_row_offset + 1)
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
            # Term 4: A[1,1] * B[1,j]
            u_idx = (a_row_offset + 1) * m + (a_col_offset + 1)
            v_idx = (b_row_offset + 1) * p + j
            w_idx = j * n + (c_row_offset + 1)
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part 1: A rows 0,1 cols 0,1 * B rows 0,1 cols 0,1,2,3 -> C rows 0,1 cols 0,1,2,3
    add_2x2_2x4(0, 0, 0, 0)
    
    # Part 2: A rows 0,1 cols 2,3 * B rows 2,3 cols 0,1,2,3 -> C rows 0,1 cols 0,1,2,3
    add_2x2_2x4(0, 2, 2, 0)
    
    # Part 3: A row 2 cols 0,1,2,3 * B rows 0,1,2,3 cols 0,1,2,3 -> C row 2 cols 0,1,2,3
    # Rank 4.
    for j in range(4):
        # C[2, j] = sum_{k=0}^3 A[2, k] B[k, j]
        # We use 4 multiplications.
        for k in range(4):
            u_idx = 2 * m + k
            v_idx = k * p + j
            w_idx = j * n + 2
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    if col != rank:
        # Adjust rank if needed
        rank = col
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]
        
    # Verify
    def verify_full():
        for r in range(n):
            for c in range(p):
                for i in range(m):
                    a_idx = r * m + i
                    b_idx = i * p + c
                    w_idx = c * n + r
                    expected = 1.0
                    actual = sum(U[a_idx, k] * V[b_idx, k] * W[w_idx, k] for k in range(rank))
                    if abs(actual - expected) > 1e-10:
                        return False
        # Check zeros
        for r in range(n):
            for c in range(p):
                for i in range(m):
                    for j in range(m):
                        if i != j:
                            a_idx = r * m + j
                            b_idx = i * p + c
                            w_idx = c * n + r
                            actual = sum(U[a_idx, k] * V[b_idx, k] * W[w_idx, k] for k in range(rank))
                            if abs(actual) > 1e-10:
                                return False
        return True
        
    if not verify_full():
        # Fallback to rank 48
        rank = 48
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r in range(n):
            for i in range(m):
                for c in range(p):
                    k = r * m * p + i * p + c
                    U[r * m + i, k] = 1.0
                    V[i * p + c, k] = 1.0
                    W[c * n + r, k] = 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
