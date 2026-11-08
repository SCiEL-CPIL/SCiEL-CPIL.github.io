import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        """
        Adds 7 Strassen terms for a 2x2 * 2x2 block multiplication.
        """
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)

    def strassen_4x4_head():
        """
        Generates 49 terms for 4x4 * 4x4 using recursive Strassen.
        """
        # Outer Strassen terms for 4x4*4x4
        outer_terms = [
            # M1 = (A00+A11)(B00+B11) -> C00 + C11
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2 = (A10+A11)B00 -> C10 - C11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3 = A00(B01-B11) -> C01 + C11
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4 = A11(B10-B00) -> C00 + C10
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5 = (A00+A01)B11 -> -C00 + C01
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6 = (A10-A00)(B00+B01) -> C11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7 = (A01-A11)(B10+B11) -> C00
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        # Inner Strassen terms for 2x2 * 2x2
        inner_terms = [
            # M1 = (x00+x11)(y00+y11) -> z00 + z11
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2 = (x10+x11)y00 -> z10 - z11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3 = x00(y01-y11) -> z01 + z11
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4 = x11(y10-y00) -> z00 + z10
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5 = (x00+x01)y11 -> -z00 + z01
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6 = (x10-x00)(y00+y01) -> z11
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7 = (x01-x11)(y10+y11) -> z00
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, matrix_type):
            """
            Get global flat indices for a block entry.
            """
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            
            row = br * 2 + lr
            col = bc * 2 + lc
            
            if matrix_type == 'A':
                return row * m + col
            elif matrix_type == 'B':
                return row * p + col
            elif matrix_type == 'C':
                # W index: k * n + i, where k is col, i is row
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                # Accumulate U
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                        
                # Accumulate V
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                        
                # Accumulate W
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                        
                add_term(u, v, w)

    def strassen_4x4x2_tail():
        """
        Generates 28 terms for 4x4 * 4x2 (cols 4,5) using block Strassen.
        """
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        """
        Generates 16 terms for 4x4 * 4x1 (col 6) using schoolbook.
        """
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    # Execute generation
    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

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
