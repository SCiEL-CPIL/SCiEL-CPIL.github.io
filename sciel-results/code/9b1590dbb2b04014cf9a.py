import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_3x3(i_off, j_off, k_off):
        """
        Adds 7 Strassen terms for 3x3 * 3x3 block.
        A block rows [i_off, i_off+2], cols [j_off, j_off+2]
        B block rows [j_off, j_off+2], cols [k_off, k_off+2]
        C block rows [i_off, i_off+2], cols [k_off, k_off+2]
        """
        # M1 = (A00+A22)(B00+B22) -> C00 + C22
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+2) * m + (j_off+2)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+2) * p + (k_off+2)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+2) * n + (i_off+2)] = 1
        add_term(u, v, w)

        # M2 = (A20+A22)B00 -> C20 - C22
        u = [0] * (n * m)
        u[(i_off+2) * m + j_off] = 1
        u[(i_off+2) * m + (j_off+2)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+2)] = 1
        w[(k_off+2) * n + (i_off+2)] = -1
        add_term(u, v, w)

        # M3 = A00(B02-B22) -> C02 + C22
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+2)] = 1
        v[(j_off+2) * p + (k_off+2)] = -1
        w = [0] * (p * n)
        w[(k_off+2) * n + i_off] = 1
        w[(k_off+2) * n + (i_off+2)] = 1
        add_term(u, v, w)

        # M4 = A22(B20-B00) -> C00 + C20
        u = [0] * (n * m)
        u[(i_off+2) * m + (j_off+2)] = 1
        v = [0] * (m * p)
        v[(j_off+2) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+2)] = 1
        add_term(u, v, w)

        # M5 = (A00+A02)B22 -> -C00 + C02
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+2)] = 1
        v = [0] * (m * p)
        v[(j_off+2) * p + (k_off+2)] = 1
        w = [0] * (p * n)
        w[(k_off+2) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        add_term(u, v, w)

        # M6 = (A20-A00)(B00+B02) -> C22
        u = [0] * (n * m)
        u[(i_off+2) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+2)] = 1
        w = [0] * (p * n)
        w[(k_off+2) * n + (i_off+2)] = 1
        add_term(u, v, w)

        # M7 = (A02-A22)(B20+B22) -> C00
        u = [0] * (n * m)
        u[i_off * m + (j_off+2)] = 1
        u[(i_off+2) * m + (j_off+2)] = -1
        v = [0] * (m * p)
        v[(j_off+2) * p + k_off] = 1
        v[(j_off+2) * p + (k_off+2)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        add_term(u, v, w)

    def strassen_4x4_head():
        """
        Generates 49 terms for 4x4 * 4x4 using recursive Strassen.
        """
        # Outer Strassen terms
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        # Inner Strassen terms for 2x2 * 2x2
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, dim_a, dim_b, offset_k=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            
            row = br * 2 + lr
            col = bc * 2 + lc + offset_k
            
            if dim_a == 'A':
                return row * m + col
            elif dim_a == 'B':
                return row * p + col
            elif dim_a == 'C':
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A', None)
                        u[idx] += a_coeff * x_coeff
                        
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B', None, offset_k=0)
                        v[idx] += b_coeff * y_coeff
                        
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C', None, offset_k=0)
                        w[idx] += c_coeff * z_coeff
                        
                add_term(u, v, w)

    def specialized_tail():
        """
        Generates 12 terms for 4x4 * 4x3 (cols 4,5,6).
        A is 4x4, B is 4x3 (cols 4,5,6 of original B).
        Partition A into 3x3 TL, 3x1 TR, 1x3 BL, 1x1 BR.
        Partition B into 3x3 T, 1x3 B.
        """
        # 1. A_TL * B_T (3x3 * 3x3) -> C_TL (3x3)
        # This is Strassen on the top-left 3x3 block of A and top-left 3x3 block of B (cols 4,5,6).
        # A_TL is rows 0-2, cols 0-2 of A.
        # B_T is rows 0-2, cols 4-6 of B.
        # C_TL is rows 0-2, cols 4-6 of C.
        strassen_3x3(0, 0, 4)

        # 2. A_BL * B_T (1x3 * 3x3) -> C_BL (1x3)
        # A_BL is row 3, cols 0-2 of A.
        # B_T is rows 0-2, cols 4-6 of B.
        # C_BL is row 3, cols 4-6 of C.
        # C[3, 4+k] = sum_{j=0}^2 A[3, j] B[j, 4+k]
        # We need 3 terms, one for each j.
        for j in range(3):
            u = [0] * (n * m)
            u[3 * m + j] = 1
            v = [0] * (m * p)
            v[j * p + 4] = 1
            v[j * p + 5] = 1
            v[j * p + 6] = 1
            w = [0] * (p * n)
            w[4 * n + 3] = 1
            w[5 * n + 3] = 1
            w[6 * n + 3] = 1
            add_term(u, v, w)

        # 3. A_TR * B_B (3x1 * 1x3) -> C_TR (3x3)
        # A_TR is rows 0-2, col 3 of A.
        # B_B is row 3, cols 4-6 of B.
        # C_TR is rows 0-2, cols 4-6 of C.
        # C[i, 4+k] = A[i, 3] B[3, 4+k]
        # This is a single term.
        u = [0] * (n * m)
        u[0 * m + 3] = 1
        u[1 * m + 3] = 1
        u[2 * m + 3] = 1
        v = [0] * (m * p)
        v[3 * p + 4] = 1
        v[3 * p + 5] = 1
        v[3 * p + 6] = 1
        w = [0] * (p * n)
        for i in range(3):
            for k in range(3):
                w[(4+k) * n + i] = 1
        add_term(u, v, w)

        # 4. A_BR * B_B (1x1 * 1x3) -> C_BR (1x3)
        # A_BR is row 3, col 3 of A.
        # B_B is row 3, cols 4-6 of B.
        # C_BR is row 3, cols 4-6 of C.
        # C[3, 4+k] = A[3, 3] B[3, 4+k]
        # This is a single term.
        u = [0] * (n * m)
        u[3 * m + 3] = 1
        v = [0] * (m * p)
        v[3 * p + 4] = 1
        v[3 * p + 5] = 1
        v[3 * p + 6] = 1
        w = [0] * (p * n)
        w[4 * n + 3] = 1
        w[5 * n + 3] = 1
        w[6 * n + 3] = 1
        add_term(u, v, w)

    # Execute generation
    strassen_4x4_head()
    specialized_tail()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    # Check for linearly dependent terms
    independent_indices = [0]
    for i in range(1, len(terms)):
        u_i, v_i, w_i = terms[i]
        is_dep = False
        for j in independent_indices:
            u_j, v_j, w_j = terms[j]
            scale = None
            for r in range(n * m):
                if u_j[r] != 0:
                    if u_i[r] == 0:
                        scale = None
                        break
                    scale = u_i[r] / u_j[r]
                    break
            if scale is None:
                continue
            match = True
            for r in range(n * m):
                if abs(u_i[r] - scale * u_j[r]) > 1e-9:
                    match = False
                    break
            if not match:
                continue
            for r in range(m * p):
                if abs(v_i[r] - scale * v_j[r]) > 1e-9:
                    match = False
                    break
            if not match:
                continue
            for r in range(p * n):
                if abs(w_i[r] - scale * w_j[r]) > 1e-9:
                    match = False
                    break
            if match:
                is_dep = True
                break
        if not is_dep:
            independent_indices.append(i)
    
    terms = [terms[i] for i in independent_indices]
    
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
