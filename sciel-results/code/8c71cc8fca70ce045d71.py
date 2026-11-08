import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin within 120s

    n, m, p = 4, 4, 7

    terms = []

    def add_term(u, v, w):
        # Filter out zero terms just in case
        if not any(u) or not any(v) or not any(w):
            return
        terms.append((u, v, w))

    # Helper to generate Strassen terms for 2x2 * 2x2
    def strassen_2x2_terms(i_off, j_off, k_off):
        # u: A indices, v: B indices, w: C indices
        # A is n x m (4x4), B is m x p (4x7), C is p x n (7x4)
        # Local 2x2 block of A: rows i_off, i_off+1; cols j_off, j_off+1
        # Local 2x2 block of B: rows j_off, j_off+1; cols k_off, k_off+1
        # Local 2x2 block of C: rows k_off, k_off+1; cols i_off, i_off+1
        
        # Indices in global vectors
        # A: row * m + col
        # B: row * p + col
        # C: col * n + row (Trace convention W[k*n+i, r])
        
        def a_idx(r, c): return (i_off + r) * m + (j_off + c)
        def b_idx(r, c): return (j_off + r) * p + (k_off + c)
        def c_idx(r, c): return (k_off + r) * n + (i_off + c)

        # 7 Strassen terms
        # 1. M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_idx(0,0)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(0,0)] = 1; v[b_idx(1,1)] = 1
        w[c_idx(0,0)] = 1; w[c_idx(1,1)] = 1
        add_term(u, v, w)

        # 2. M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_idx(1,0)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(0,0)] = 1
        w[c_idx(0,1)] = 1; w[c_idx(1,1)] = -1
        add_term(u, v, w)

        # 3. M3 = A00(B01+B11) -> C01 += M3, C11 += M3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_idx(0,0)] = 1
        v[b_idx(0,1)] = 1; v[b_idx(1,1)] = 1
        w[c_idx(1,0)] = 1; w[c_idx(1,1)] = 1
        add_term(u, v, w)

        # 4. M4 = A11(B10+B00) -> C00 += M4, C10 += M4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_idx(1,1)] = 1
        v[b_idx(1,0)] = 1; v[b_idx(0,0)] = 1
        w[c_idx(0,0)] = 1; w[c_idx(0,1)] = 1
        add_term(u, v, w)

        # 5. M5 = (A00+A01)B11 -> C01 += M5, C00 -= M5
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_idx(0,0)] = 1; u[a_idx(0,1)] = 1
        v[b_idx(1,1)] = 1
        w[c_idx(1,0)] = 1; w[c_idx(0,0)] = -1
        add_term(u, v, w)

        # 6. M6 = (A10+A00)B01 -> C11 += M6
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_idx(1,0)] = 1; u[a_idx(0,0)] = 1
        v[b_idx(0,1)] = 1
        w[c_idx(1,1)] = 1
        add_term(u, v, w)

        # 7. M7 = (A01+A11)B10 -> C00 += M7
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[a_idx(0,1)] = 1; u[a_idx(1,1)] = 1
        v[b_idx(1,0)] = 1
        w[c_idx(0,0)] = 1
        add_term(u, v, w)

    # Head: 4x4 * 4x4 (Cols 0-3 of B)
    # Use 4 blocks of 2x2 * 2x2 Strassen
    for i_block in range(2):
        for j_block in range(2):
            i_off = i_block * 2
            j_off = j_block * 2
            k_off = 0 # Cols 0-1 for first two, 2-3 for next two?
            # Wait, B cols 0-3.
            # Block (0,0) of A uses B rows 0-1. B cols 0-1?
            # No, Strassen for 4x4*4x4 uses 2x2 blocks.
            # A blocks: (0,0), (0,1), (1,0), (1,1).
            # B blocks: (0,0), (0,1), (1,0), (1,1).
            # C blocks: (0,0), (0,1), (1,0), (1,1).
            # The 7 terms combine these.
            # My strassen_2x2_terms function generates terms for a SINGLE 2x2*2x2 multiplication.
            # To get 4x4*4x4, I need to combine 4 such multiplications?
            # No, Strassen for 4x4*4x4 is NOT 4 independent 2x2*2x2 Strassen.
            # It is a recursive Strassen.
            # The current program's `strassen_4x4_head` uses a nested loop of 7x7 terms.
            # This corresponds to the recursive application.
            # Let's stick to the recursive structure for the head.
            
            pass # Skipping this block logic for now, will use the nested approach from before but cleaner.

    # Let's revert to the nested approach for the head, as it is known to be 49 terms.
    # And the tail approach.
    
    # Re-implementing the head with the nested 7x7 structure
    def strassen_4x4_head():
        # Outer Strassen terms (7)
        # Inner Strassen terms (7)
        # This generates 49 terms.
        
        # Indices for 2x2 blocks in 4x4 matrix
        # Block 0: (0,0), Block 1: (0,1), Block 2: (1,0), Block 3: (1,1)
        
        outer_terms = [
            # (A_blocks, B_blocks, C_blocks) with coefficients
            # Format: ([(block_idx, coeff), ...], ...)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def get_global_idx(block_idx, local_idx, dim, col_offset=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + col_offset
            
            if dim == 'A':
                return row * m + col
            elif dim == 'B':
                return row * p + col
            elif dim == 'C':
                return col * n + row

        for oa, ob, oc in outer_terms:
            for ia, ib, ic in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for a_block, a_coeff in oa:
                    for a_local, a_l_coeff in ia:
                        idx = get_global_idx(a_block, a_local, 'A')
                        u[idx] += a_coeff * a_l_coeff
                        
                for b_block, b_coeff in ob:
                    for b_local, b_l_coeff in ib:
                        idx = get_global_idx(b_block, b_local, 'B', col_offset=0)
                        v[idx] += b_coeff * b_l_coeff
                        
                for c_block, c_coeff in oc:
                    for c_local, c_l_coeff in ic:
                        idx = get_global_idx(c_block, c_local, 'C', col_offset=0)
                        w[idx] += c_coeff * c_l_coeff
                        
                add_term(u, v, w)

    def strassen_4x4x2_tail():
        # Tail cols 4-5.
        # Use 4 blocks of 2x2 * 2x2 Strassen.
        # B rows are split 0-1, 2-3.
        # B cols are 4-5.
        # A rows/cols split 0-1, 2-3.
        # C rows 4-5, cols 0-3.
        
        for i_block in range(2):
            for j_block in range(2):
                i_off = i_block * 2
                j_off = j_block * 2
                k_off = 4
                strassen_2x2_terms(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        # Tail col 6.
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
