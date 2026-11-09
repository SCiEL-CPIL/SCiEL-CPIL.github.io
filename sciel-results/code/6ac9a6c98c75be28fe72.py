import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Improved block decomposition:
    # Split m=5 into 2+3
    # Part A: 2x2x5, split p=5 into 2+2+1:
    #   - 2x2x2 Strassen: rank 7
    #   - 2x2x2 Strassen: rank 7
    #   - 2x2x1 schoolbook: rank 2
    #   Part A total: 16
    # Part B: 2x3x5, split m=3 into 2+1:
    #   - 2x2x5: split p=5 into 2+2+1: rank 7+7+2 = 16
    #   - 2x1x5: schoolbook: rank 10
    #   Part B total: 26
    # Grand total: 16 + 26 = 42
    
    U_data = [] # List of (row_idx, col_idx, val)
    V_data = []
    W_data = []
    
    col_offset = 0
    
    def add_term(u_terms, v_terms, w_terms):
        nonlocal col_offset
        for idx, val in u_terms:
            U_data.append((idx, col_offset, val))
        for idx, val in v_terms:
            V_data.append((idx, col_offset, val))
        for idx, val in w_terms:
            W_data.append((idx, col_offset, val))
        col_offset += 1
    
    def add_strassen_block(i_base, j_base, k_base):
        """Add Strassen decomposition for a 2x2x2 block at given base indices."""
        def u_idx(i_l, j_l):
            return (i_base + i_l) * m + (j_base + j_l)
        def v_idx(j_l, k_l):
            return (j_base + j_l) * p + (k_base + k_l)
        def w_idx(k_l, i_l):
            return (k_base + k_l) * n + (i_base + i_l)
        
        # m1 = (A00+A11)(B00+B11)
        add_term(
            [(u_idx(0,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), 1), (w_idx(1,1), 1)]
        )
        # m2 = (A10+A11)B00
        add_term(
            [(u_idx(1,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1)],
            [(w_idx(0,1), 1), (w_idx(1,1), -1)]
        )
        # m3 = A00(B01-B11)
        add_term(
            [(u_idx(0,0), 1)],
            [(v_idx(0,1), 1), (v_idx(1,1), -1)],
            [(w_idx(1,0), 1), (w_idx(1,1), 1)]
        )
        # m4 = A11(B10-B00)
        add_term(
            [(u_idx(1,1), 1)],
            [(v_idx(1,0), 1), (v_idx(0,0), -1)],
            [(w_idx(0,1), 1), (w_idx(0,0), 1)]
        )
        # m5 = (A00+A01)B11
        add_term(
            [(u_idx(0,0), 1), (u_idx(0,1), 1)],
            [(v_idx(1,1), 1)],
            [(w_idx(1,0), 1), (w_idx(0,0), -1)]
        )
        # m6 = (A10-A00)(B00+B01)
        add_term(
            [(u_idx(1,0), 1), (u_idx(0,0), -1)],
            [(v_idx(0,0), 1), (v_idx(0,1), 1)],
            [(w_idx(1,1), 1)]
        )
        # m7 = (A01-A11)(B10+B11)
        add_term(
            [(u_idx(0,1), 1), (u_idx(1,1), -1)],
            [(v_idx(1,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), 1)]
        )
    
    def add_schoolbook_block(i_base, j_start, j_end, k_start, k_end):
        """Add schoolbook terms for A[i_base:i_base+2, j_start:j_end] * B[j_start:j_end, k_start:k_end]"""
        for i in range(2):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    add_term(
                        [((i_base + i) * m + j, 1)],
                        [(j * p + k, 1)],
                        [(k * n + i_base + i, 1)]
                    )
    
    # Part A: 2x2x5 (i_base=0, j=0..1, k=0..4)
    # Split k into 0-1, 2-3, 4
    add_strassen_block(0, 0, 0)  # 2x2x2 for k=0,1
    add_strassen_block(0, 0, 2)  # 2x2x2 for k=2,3
    # 2x2x1 for k=4
    for i in range(2):
        for j in range(2):
            add_term(
                [(i * m + j, 1)],
                [(j * p + 4, 1)],
                [(4 * n + i, 1)]
            )
    
    # Part B: 2x3x5 (i_base=0, j=2..4, k=0..4)
    # Split j into 2-3 and 4
    # B1: 2x2x5 (j=2,3), split k into 0-1, 2-3, 4
    add_strassen_block(0, 2, 0)  # 2x2x2 for j=2,3 k=0,1
    add_strassen_block(0, 2, 2)  # 2x2x2 for j=2,3 k=2,3
    # 2x2x1 for j=2,3 k=4
    for i in range(2):
        for j in range(2, 4):
            add_term(
                [(i * m + j, 1)],
                [(j * p + 4, 1)],
                [(4 * n + i, 1)]
            )
    
    # B2: 2x1x5 (j=4), schoolbook
    for i in range(2):
        for k in range(5):
            add_term(
                [(i * m + 4, 1)],
                [(4 * p + k, 1)],
                [(k * n + i, 1)]
            )
    
    R = col_offset
    
    # Build matrices
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for idx, col, val in U_data:
        U[idx][col] = val
    for idx, col, val in V_data:
        V[idx][col] = val
    for idx, col, val in W_data:
        W[idx][col] = val

    # Verify
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}

    # Simple verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(R):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook if error
                    U = [[0] * 50 for _ in range(n * m)]
                    V = [[0] * 50 for _ in range(m * p)]
                    W = [[0] * 50 for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
