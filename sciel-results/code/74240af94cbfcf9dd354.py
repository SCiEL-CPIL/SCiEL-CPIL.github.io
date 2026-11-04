"""Matrix multiplication 4,4,9 tensor decomposition - Strassen block rank-114."""
import time

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2x2: 7 rank-1 terms
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),        # M7
    ]
    
    def add_strassen_2x2(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        """Add 7 Strassen terms for a 2x2 * 2x2 -> 2x2 block."""
        for u_c, v_c, w_c in strassen_2x2:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for loc, coeff in u_c.items():
                li = loc // 2
                lj = loc % 2
                gi = A_row_off + li
                gj = A_col_off + lj
                u_vec[gi * m + gj] += coeff
            
            for loc, coeff in v_c.items():
                lj = loc // 2
                lk = loc % 2
                gj = B_row_off + lj
                gk = B_col_off + lk
                v_vec[gj * p + gk] += coeff
            
            for loc, coeff in w_c.items():
                li = loc // 2
                lk = loc % 2
                gi = C_row_off + li
                gk = C_col_off + lk
                w_vec[gk * n + gi] += coeff
            
            add_term(u_vec, v_vec, w_vec)
    
    def add_strassen_4x4(B_col_offset):
        """Add 49 terms for 4x4 * 4x4 using Strassen at two levels."""
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            """Add 7 Strassen terms for the inner 2x2 multiplication."""
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # A side
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = br * 2 + li
                        gj = bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                # B side
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = br * 2 + lj
                        gk = B_col_offset + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                # C side
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = br * 2 + li
                        gk = B_col_offset + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # Outer Strassen for 4x4:
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)B00      -> C10, C11
        # M3 = A00(B01-B11)      -> C01, C11
        # M4 = A11(B10-B00)      -> C00, C10
        # M5 = (A00+A01)B11      -> C00, C01
        # M6 = (A10-A00)(B00+B01) -> C11
        # M7 = (A01-A11)(B10+B11) -> C00
        
        # M1
        add_inner(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        )
        # M2
        add_inner(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        )
        # M3
        add_inner(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        )
        # M4
        add_inner(
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        )
        # M5
        add_inner(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        )
        # M6
        add_inner(
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        )
        # M7
        add_inner(
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        )
    
    # Part 1: A * B[:, 0:4]
    add_strassen_4x4(0)
    
    # Part 2: A * B[:, 4:8]
    add_strassen_4x4(4)
    
    # Part 3: A * B[:, 8:9] - schoolbook, rank 16
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
