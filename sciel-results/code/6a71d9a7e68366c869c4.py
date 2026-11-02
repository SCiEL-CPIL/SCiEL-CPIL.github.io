import numpy as np
import time

def construct():
    n, m, p = 5, 5, 5
    deadline = time.time() + 110  # Safety margin
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        u_vec[u_idx] = u_val
        v_vec[v_idx] = v_val
        w_vec[w_idx] = w_val
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Split 5 into 1 + 4
    # Blocks: index 0 (size 1), index 1 (size 4)
    block_sizes = [1, 4]
    block_offsets = [0, 1]
    
    # Add schoolbook terms for small blocks
    def add_schoolbook_block(bi, bj, bk):
        ni = block_sizes[bi]
        mj = block_sizes[bj]
        pk = block_sizes[bk]
        off_i = block_offsets[bi]
        off_j = block_offsets[bj]
        off_k = block_offsets[bk]
        
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    if time.time() > deadline:
                        return
                    u_idx = idx_u(off_i + i, off_j + j)
                    v_idx = idx_v(off_j + j, off_k + k)
                    w_idx = idx_w(off_k + k, off_i + i)
                    add_term(u_idx, v_idx, w_idx)
    
    # Add Strassen terms for 4x4x4
    def add_strassen_4x4(bi, bj, bk):
        off_i = block_offsets[bi]
        off_j = block_offsets[bj]
        off_k = block_offsets[bk]
        
        # Strassen for 4x4: split into 2x2 blocks of 2x2 matrices
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mx is a 2x2 matrix product, which has rank 7 (Strassen)
        # But we can use schoolbook for 2x2: rank 4
        # So total rank for 4x4 = 7 * 4 = 28
        
        # Define 2x2 Strassen helper
        def add_2x2_strassen(ai_off, bj_off, ck_off):
            # ai_off, bj_off, ck_off are base indices for 2x2 blocks
            # A2x2 at (ai_off, bj_off), B2x2 at (bj_off, ck_off), C2x2 at (ck_off, ai_off)
            
            # Schoolbook for 2x2: rank 4
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        if time.time() > deadline:
                            return
                        u_idx = idx_u(ai_off + i, bj_off + j)
                        v_idx = idx_v(bj_off + j, ck_off + k)
                        w_idx = idx_w(ck_off + k, ai_off + i)
                        add_term(u_idx, v_idx, w_idx)
        
        # Block positions within the 4x4 block:
        # A11: rows 0-1, cols 0-1 -> base (0, 0)
        # A12: rows 0-1, cols 2-3 -> base (0, 2)
        # A21: rows 2-3, cols 0-1 -> base (2, 0)
        # A22: rows 2-3, cols 2-3 -> base (2, 2)
        
        # B11: rows 0-1, cols 0-1 -> base (0, 0)
        # B12: rows 0-1, cols 2-3 -> base (0, 2)
        # B21: rows 2-3, cols 0-1 -> base (2, 0)
        # B22: rows 2-3, cols 2-3 -> base (2, 2)
        
        # C11: rows 0-1, cols 0-1 -> base (0, 0)
        # C12: rows 0-1, cols 2-3 -> base (0, 2)
        # C21: rows 2-3, cols 0-1 -> base (2, 0)
        # C22: rows 2-3, cols 2-3 -> base (2, 2)
        
        # M1 = (A11+A22)(B11+B22)
        # This is a 2x2 product where:
        # A term: A11 + A22 (sum of two 2x2 matrices)
        # B term: B11 + B22
        # C term: contributes to C11 and C22
        
        # We need to express (A11+A22)(B11+B22) as sum of rank-1 terms
        # (A11+A22)B11 + (A11+A22)B22
        # = A11*B11 + A22*B11 + A11*B22 + A22*B22
        # Each is a 2x2 product, rank 4 each, total 16 terms
        
        # But we can do better with Strassen for the outer level
        # Actually, let's just use schoolbook for all 2x2 products within Strassen
        # That gives 7 * 4 = 28 terms for the 4x4 block
        
        # M1: (A11+A22)(B11+B22)
        # = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # Each 2x2 product: 4 terms
        # Total for M1: 16 terms
        
        # Similarly for other Mx terms
        
        # Let me implement this directly
        
        # Helper to add a 2x2 product with possible signs
        def add_2x2_product(a_base, b_base, c_base, sign=1):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        if time.time() > deadline:
                            return
                        u_idx = idx_u(off_i + a_base[0] + i, off_j + b_base[1] + j)
                        v_idx = idx_v(off_j + b_base[0] + j, off_k + b_base[1] + k)
                        w_idx = idx_w(off_k + b_base[1] + k, off_i + a_base[0] + i)
                        add_term(u_idx, v_idx, w_idx, u_val=sign, v_val=sign, w_val=1)
        
        # This is getting complex. Let me use a different approach.
        # Just use schoolbook for the entire 4x4 block: rank 64
        # Total rank: 1 (1x1) + 4 (1x4) + 4 (4x1) + 64 (4x4) = 73
        # That's better than 125!
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if time.time() > deadline:
                        return
                    u_idx = idx_u(off_i + i, off_j + j)
                    v_idx = idx_v(off_j + j, off_k + k)
                    w_idx = idx_w(off_k + k, off_i + i)
                    add_term(u_idx, v_idx, w_idx)
    
    # Add all block combinations
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                if time.time() > deadline:
                    break
                if block_sizes[bi] == 4 and block_sizes[bj] == 4 and block_sizes[bk] == 4:
                    add_strassen_4x4(bi, bj, bk)
                else:
                    add_schoolbook_block(bi, bj, bk)
    
    # Wait, I'm double-counting. Let me reconsider.
    # The block decomposition for matrix multiplication C = AB:
    # If A is partitioned as [[A11, A12], [A21, A22]] (1x1, 1x4, 4x1, 4x4)
    # B is partitioned as [[B11, B12], [B21, B22]]
    # C is partitioned as [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 (1x1 * 1x1 = 1x1)
    # C12 = A11*B12 + A12*B22 (1x1 * 1x4 + 1x4 * 4x4 = 1x4)
    # C21 = A21*B11 + A22*B21 (4x1 * 1x1 + 4x4 * 4x1 = 4x1)
    # C22 = A21*B12 + A22*B22 (4x1 * 1x4 + 4x4 * 4x4 = 4x4)
    
    # So the terms are:
    # A11*B11: 1 term
    # A11*B12: 4 terms (1x4)
    # A12*B22: 4*4 = 16 terms
    # A21*B11: 4 terms
    # A22*B21: 16 terms
    # A21*B12: 4*4 = 16 terms
    # A22*B22: 16*4 = 64 terms (schoolbook) or 28 (Strassen)
    
    # Total with schoolbook: 1 + 4 + 16 + 4 + 16 + 16 + 64 = 121
    # That's better than 125!
    
    # Let me restart with this correct decomposition
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # A11: (0,0), A12: (0,1:5), A21: (1:5,0), A22: (1:5,1:5)
    # B11: (0,0), B12: (0,1:5), B21: (1:5,0), B22: (1:5,1:5)
    # C11: (0,0), C12: (0,1:5), C21: (1:5,0), C22: (1:5,1:5)
    
    # C11 = A11 * B11
    add_term(idx_u(0, 0), idx_v(0, 0), idx_w(0, 0))
    
    # C12 = A11 * B12 + A12 * B22
    # A11 * B12: A(0,0) * B(0,k) for k=1..4 -> C(0,k)
    for k in range(1, 5):
        if time.time() > deadline:
            return {"U": [], "V": [], "W": []}
        add_term(idx_u(0, 0), idx_v(0, k), idx_w(k, 0))
    
    # A12 * B22: A(0,j) * B(j,k) for j=1..4, k=1..4 -> C(0,k)
    for j in range(1, 5):
        for k in range(1, 5):
            if time.time() > deadline:
                return {"U": [], "V": [], "W": []}
            add_term(idx_u(0, j), idx_v(j, k), idx_w(k, 0))
    
    # C21 = A21 * B11 + A22 * B21
    # A21 * B11: A(i,0) * B(0,0) for i=1..4 -> C(i,0)
    for i in range(1, 5):
        if time.time() > deadline:
            return {"U": [], "V": [], "W": []}
        add_term(idx_u(i, 0), idx_v(0, 0), idx_w(0, i))
    
    # A22 * B21: A(i,j) * B(j,0) for i=1..4, j=1..4 -> C(i,0)
    for i in range(1, 5):
        for j in range(1, 5):
            if time.time() > deadline:
                return {"U": [], "V": [], "W": []}
            add_term(idx_u(i, j), idx_v(j, 0), idx_w(0, i))
    
    # C22 = A21 * B12 + A22 * B22
    # A21 * B12: A(i,0) * B(0,k) for i=1..4, k=1..4 -> C(i,k)
    for i in range(1, 5):
        for k in range(1, 5):
            if time.time() > deadline:
                return {"U": [], "V": [], "W": []}
            add_term(idx_u(i, 0), idx_v(0, k), idx_w(k, i))
    
    # A22 * B22: A(i,j) * B(j,k) for i=1..4, j=1..4, k=1..4 -> C(i,k)
    # Use schoolbook: 64 terms
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                if time.time() > deadline:
                    return {"U": [], "V": [], "W": []}
                add_term(idx_u(i, j), idx_v(j, k), idx_w(k, i))
    
    U = np.column_stack(U_cols) if U_cols else np.zeros((25, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((25, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((25, 0))
    
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                row_list.append(rounded)
            result.append(row_list)
        return result
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
