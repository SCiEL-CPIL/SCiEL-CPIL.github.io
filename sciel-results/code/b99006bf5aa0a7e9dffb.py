import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    n, m, p = 3, 5, 5
    dim_u = n * m  # 15
    dim_v = m * p  # 25
    dim_w = p * n  # 15
    
    # Partition indices:
    # A: 3x5 -> rows split as [2, 1], cols split as [2, 2, 1]
    # B: 5x5 -> rows split as [2, 2, 1], cols split as [2, 2, 1]
    # C: 5x3 -> rows split as [2, 2, 1], cols split as [2, 1]
    
    # Row partitions for A (3 rows): [0,1], [2]
    a_row_parts = [(0, 2), (2, 3)]  # sizes 2, 1
    
    # Col partitions for A / Row partitions for B (5): [0,1], [2,3], [4]
    a_col_parts = [(0, 2), (2, 4), (4, 5)]  # sizes 2, 2, 1
    
    # Col partitions for B / Row partitions for C (5): [0,1], [2,3], [4]
    b_col_parts = [(0, 2), (2, 4), (4, 5)]  # sizes 2, 2, 1
    
    # Col partitions for C (3 cols): [0,1], [2]
    c_col_parts = [(0, 2), (2, 3)]  # sizes 2, 1
    
    # Collect all rank-1 terms
    U_rows = []  # each element is a list of length dim_u
    V_rows = []  # each element is a list of length dim_v
    W_rows = []  # each element is a list of length dim_w
    
    def add_term(u_entry, v_entry, w_entry):
        """Add a rank-1 term. u_entry: dict mapping (i,j) -> val, etc."""
        U_row = [0.0] * dim_u
        for (i, j), val in u_entry.items():
            if val != 0:
                U_row[i * m + j] = val
        V_row = [0.0] * dim_v
        for (j, k), val in v_entry.items():
            if val != 0:
                V_row[j * p + k] = val
        W_row = [0.0] * dim_w
        for (k, i), val in w_entry.items():
            if val != 0:
                W_row[k * n + i] = val
        U_rows.append(U_row)
        V_rows.append(V_row)
        W_rows.append(W_row)
    
    def strassen_222(a_row_range, a_col_range, b_col_range):
        """Add Strassen decomposition for a 2x2 @ 2x2 block.
        a_row_range: (start, end) for A rows (size 2)
        a_col_range: (start, end) for A cols / B rows (size 2)
        b_col_range: (start, end) for B cols / C rows (size 2)
        C cols correspond to a_row_range (size 2)
        """
        ar0, ar1 = a_row_range[0], a_row_range[1] - 1  # A row indices
        ac0, ac1 = a_col_range[0], a_col_range[1] - 1  # A col / B row indices
        bc0, bc1 = b_col_range[0], b_col_range[1] - 1  # B col / C row indices
        cr0, cr1 = a_row_range[0], a_row_range[1] - 1  # C col indices (same as A rows)
        
        # Strassen 7 multiplications for 2x2 @ 2x2:
        # P1 = A00 * (B00 - B11)
        # P2 = (A00 + A01) * B11
        # P3 = (A10 + A11) * B00
        # P4 = A11 * (B01 - B00)
        # P5 = (A00 + A11) * (B00 + B11)
        # P6 = (A01 - A11) * (B10 + B11)
        # P7 = (A00 - A10) * (B00 + B01)
        
        # C00 = P1 + P4 - P6 + P7
        # C01 = P2 + P3 - P5 + P7
        # C10 = P1 + P5 - P2 + P4
        # C11 = P3 + P6 - P1 + P5
        
        # Each P is a product of a 2x1 "matrix" (from A side) and 1x2 "matrix" (from B side)
        # For each term, we need U entries (coefficients on A), V entries (coefficients on B),
        # and W entries (coefficients on C output)
        
        # P1: A00 * (B00 - B11)
        # U: {(ar0,ac0): 1}, V: {(ac0,bc0): 1, (ac1,bc1): -1}, W: {(bc0,cr0): 1, (bc0,cr1): 0, (bc1,cr0): 0, (bc1,cr1): 0}
        # Wait, W maps (k,i) -> coefficient. Here k is C row (= B col), i is C col (= A row)
        # C00 gets P1, so W[(bc0, cr0)] += 1
        # C10 gets P1, so W[(bc0, cr1)] += 1  -- NO, C10 means C row 1, col 0
        # Let me re-index: C[ar_idx, bc_idx] where ar_idx in {0,1}, bc_idx in {0,1}
        # C00: ar_idx=0, bc_idx=0 -> global C row = bc0, C col = cr0
        # C01: ar_idx=0, bc_idx=1 -> global C row = bc1, C col = cr0
        # C10: ar_idx=1, bc_idx=0 -> global C row = bc0, C col = cr1
        # C11: ar_idx=1, bc_idx=1 -> global C row = bc1, C col = cr1
        
        # W entry: W[(k, i)] where k = C row index, i = C col index
        # For P1: contributes to C00 (k=bc0, i=cr0) with +1, C10 (k=bc0, i=cr1) with +1
        
        # Let me define the 7 Strassen terms properly:
        # Term 1: u = {(ar0,ac0):1}, v = {(ac0,bc0):1, (ac1,bc1):-1}, w = {(bc0,cr0):1, (bc0,cr1):1}
        # Term 2: u = {(ar0,ac0):1, (ar0,ac1):1}, v = {(ac1,bc1):1}, w = {(bc1,cr0):1, (bc1,cr1):0, (bc0,cr0):0, (bc0,cr1):0}
        # Hmm, this is getting confusing. Let me use a cleaner formulation.
        
        # For 2x2 matrices:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = A @ B
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        
        # Strassen:
        # p1 = a00*(b00 - b11)
        # p2 = (a00 + a01)*b11
        # p3 = (a10 + a11)*b00
        # p4 = a11*(b01 - b00)
        # p5 = (a00 + a11)*(b00 + b11)
        # p6 = (a01 - a11)*(b10 + b11)
        # p7 = (a00 - a10)*(b00 + b01)
        
        # c00 = p1 + p4 - p6 + p7
        # c01 = p2 + p3 - p5 + p7
        # c10 = p1 + p5 - p2 + p4
        # c11 = p3 + p6 - p1 + p5
        
        # For each pi, the U part is coefficients on A elements, V part on B elements,
        # W part on C elements.
        
        # p1: U={a00:1}, V={b00:1, b11:-1}, W={c00:1, c10:1}
        # p2: U={a00:1, a01:1}, V={b11:1}, W={c01:1}
        # p3: U={a10:1, a11:1}, V={b00:1}, W={c01:1, c11:1}
        # p4: U={a11:1}, V={b01:1, b00:-1}, W={c00:1, c10:1}
        # p5: U={a00:1, a11:1}, V={b00:1, b11:1}, W={c01:-1, c10:1, c11:1}
        # p6: U={a01:1, a11:-1}, V={b10:1, b11:1}, W={c00:-1, c11:1}
        # p7: U={a00:1, a10:-1}, V={b00:1, b01:1}, W={c00:1, c01:1}
        
        # Map local indices to global:
        # a00 -> (ar0, ac0), a01 -> (ar0, ac1)
        # a10 -> (ar1, ac0), a11 -> (ar1, ac1)
        # b00 -> (ac0, bc0), b01 -> (ac0, bc1)
        # b10 -> (ac1, bc0), b11 -> (ac1, bc1)
        # c00 -> (bc0, cr0), c01 -> (bc1, cr0)
        # c10 -> (bc0, cr1), c11 -> (bc1, cr1)
        
        # p1
        add_term(
            {(ar0, ac0): 1.0},
            {(ac0, bc0): 1.0, (ac1, bc1): -1.0},
            {(bc0, cr0): 1.0, (bc0, cr1): 1.0}
        )
        # p2
        add_term(
            {(ar0, ac0): 1.0, (ar0, ac1): 1.0},
            {(ac1, bc1): 1.0},
            {(bc1, cr0): 1.0}
        )
        # p3
        add_term(
            {(ar1, ac0): 1.0, (ar1, ac1): 1.0},
            {(ac0, bc0): 1.0},
            {(bc1, cr0): 1.0, (bc1, cr1): 1.0}
        )
        # p4
        add_term(
            {(ar1, ac1): 1.0},
            {(ac0, bc1): 1.0, (ac0, bc0): -1.0},
            {(bc0, cr0): 1.0, (bc0, cr1): 1.0}
        )
        # p5
        add_term(
            {(ar0, ac0): 1.0, (ar1, ac1): 1.0},
            {(ac0, bc0): 1.0, (ac1, bc1): 1.0},
            {(bc1, cr0): -1.0, (bc0, cr1): 1.0, (bc1, cr1): 1.0}
        )
        # p6
        add_term(
            {(ar0, ac1): 1.0, (ar1, ac1): -1.0},
            {(ac1, bc0): 1.0, (ac1, bc1): 1.0},
            {(bc0, cr0): -1.0, (bc1, cr1): 1.0}
        )
        # p7
        add_term(
            {(ar0, ac0): 1.0, (ar1, ac0): -1.0},
            {(ac0, bc0): 1.0, (ac0, bc1): 1.0},
            {(bc0, cr0): 1.0, (bc1, cr0): 1.0}
        )
    
    def trivial_1x1(a_row_idx, a_col_idx, b_col_idx):
        """Add a trivial rank-1 term for a 1x1 @ 1x1 block."""
        add_term(
            {(a_row_idx, a_col_idx): 1.0},
            {(a_col_idx, b_col_idx): 1.0},
            {(b_col_idx, a_row_idx): 1.0}
        )
    
    # Iterate over all blocks
    for ai, (ar_start, ar_end) in enumerate(a_row_parts):
        for bi, (ac_start, ac_end) in enumerate(a_col_parts):
            for ci, (bc_start, bc_end) in enumerate(b_col_parts):
                ar_size = ar_end - ar_start
                ac_size = ac_end - ac_start
                bc_size = bc_end - bc_start
                
                if ar_size == 2 and ac_size == 2 and bc_size == 2:
                    # Use Strassen for 2x2 @ 2x2 block
                    strassen_222((ar_start, ar_end), (ac_start, ac_end), (bc_start, bc_end))
                elif ar_size == 1 and ac_size == 1 and bc_size == 1:
                    # Trivial 1x1 @ 1x1
                    trivial_1x1(ar_start, ac_start, bc_start)
                else:
                    # Mixed sizes: use trivial decomposition for this block
                    for dr in range(ar_size):
                        for dc in range(ac_size):
                            for dk in range(bc_size):
                                a_row = ar_start + dr
                                a_col = ac_start + dc
                                b_col = bc_start + dk
                                trivial_1x1(a_row, a_col, b_col)
    
    # Check time
    if time.time() - start_time > deadline - 5:
        # Convert and return
        rank = len(U_rows)
        U = [list(row) for row in U_rows]
        V = [list(row) for row in V_rows]
        W = [list(row) for row in W_rows]
        return {'U': U, 'V': V, 'W': W}
    
    rank = len(U_rows)
    U = [list(row) for row in U_rows]
    V = [list(row) for row in V_rows]
    W = [list(row) for row in W_rows]
    
    return {'U': U, 'V': V, 'W': W}
