import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 130.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        """
        Adds 7 Strassen terms for a 2x2 * 2x2 block multiplication.
        A block at rows [i_off, i_off+1], cols [j_off, j_off+1]
        B block at rows [j_off, j_off+1], cols [k_off, k_off+1]
        C block at rows [i_off, i_off+1], cols [k_off, k_off+1]
        """
        terms = []
        
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
        terms.append((u, v, w))

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        terms.append((u, v, w))

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        terms.append((u, v, w))

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        terms.append((u, v, w))

        return terms

    def build_4x4x7_optimized():
        all_terms = []
        
        # Part 1: 4x4 * 4x4 (columns 0-3 of B)
        # Use recursive Strassen: 7 * 7 = 49 terms
        
        # Outer Strassen terms for 4x4 * 4x4
        # A = [[A00, A01], [A10, A11]] where each block is 2x2
        # B = [[B00, B01], [B10, B11]] where each block is 2x2
        # C = [[C00, C01], [C10, C11]] where each block is 2x2
        
        # The 7 outer multiplications:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # For each outer multiplication, we need to compute a 2x2 * 2x2 product
        # which itself requires 7 Strassen terms.
        # Total: 7 * 7 = 49 terms.
        
        # Let's define the block offsets for A, B, C
        # A blocks: A00 at (0,0), A01 at (0,2), A10 at (2,0), A11 at (2,2)
        # B blocks: B00 at (0,0), B01 at (0,2), B10 at (2,0), B11 at (2,2)
        # C blocks: C00 at (0,0), C01 at (0,2), C10 at (2,0), C11 at (2,2)
        
        # For each of the 7 outer terms, we have a combination of A blocks and B blocks.
        # We need to generate the 7 inner Strassen terms for each outer term.
        
        # This is complex to do generically. Let's use the explicit construction from inspiration 1.
        
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

        def get_block_indices(block_idx, local_idx, dim_a, offset_k=0):
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
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                all_terms.append((u, v, w))

        # Part 2: 4x4 * 4x3 (columns 4-6 of B)
        # We can split this into:
        # - 4x4 * 4x2 (columns 4-5) using Strassen-like block decomposition
        # - 4x4 * 4x1 (column 6) using schoolbook
        
        # For 4x4 * 4x2:
        # Split A into 2x2 blocks, B into 2x2x2 blocks (2 columns of 2x2)
        # Actually, B is 4x2, so we have two 2x2 blocks vertically? No, B is 4x2.
        # B = [[B_top], [B_bottom]] where B_top is 2x2, B_bottom is 2x2.
        # A = [[A00, A01], [A10, A11]]
        # C = A * B = [[A00*B_top + A01*B_bottom], [A10*B_top + A11*B_bottom]]
        # Each term like A00*B_top is 2x2 * 2x2, which can be done with 7 Strassen terms.
        # There are 4 such terms: A00*B_top, A01*B_bottom, A10*B_top, A11*B_bottom.
        # Total: 4 * 7 = 28 terms.
        
        # But wait, we need to be careful about the indices.
        # B_top is rows 0-1, cols 4-5 of B. So in B's flat index, that's j*7+4 and j*7+5 for j=0,1.
        # B_bottom is rows 2-3, cols 4-5 of B. So j*7+4 and j*7+5 for j=2,3.
        
        # Let's use the strassen_2x2_block function for each of the 4 sub-products.
        # A00 * B_top: A block at (0,0), B block at (0,4) [rows 0-1, cols 4-5]
        # A01 * B_bottom: A block at (0,2), B block at (2,4) [rows 2-3, cols 4-5]
        # A10 * B_top: A block at (2,0), B block at (0,4)
        # A11 * B_bottom: A block at (2,2), B block at (2,4)
        
        # Wait, the strassen_2x2_block function assumes A and B blocks are aligned in the same way.
        # Let's just call it for each of the 4 combinations.
        
        # A00 * B_top
        all_terms.extend(strassen_2x2_block(0, 0, 4))
        # A01 * B_bottom
        all_terms.extend(strassen_2x2_block(0, 2, 4))
        # A10 * B_top
        all_terms.extend(strassen_2x2_block(2, 0, 4))
        # A11 * B_bottom
        all_terms.extend(strassen_2x2_block(2, 2, 4))
        
        # Part 3: 4x4 * 4x1 (column 6 of B)
        # This is just schoolbook: 16 terms.
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                all_terms.append((u, v, w))

        rank = len(all_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(all_terms):
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

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7_optimized()
        
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        target = np.ones((n, m, p), dtype=float)
        if not np.allclose(tensor_decomp, target, atol=1e-9):
            return build_schoolbook()

        return result
    except Exception:
        return build_schoolbook()
