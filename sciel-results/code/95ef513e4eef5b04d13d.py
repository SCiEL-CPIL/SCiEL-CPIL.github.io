"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strategy: 4x4x6 = 4x4x4 (rank 49) + 4x4x2 (rank 28) = rank 77
    # A is 4x4, B is 4x6, C is 4x6
    # C = A @ B = [A @ B_left (4x4)] [A @ B_right (4x2)]

    # Build 4x4x4 Strassen (rank 49)
    # and 4x4x2 decomposition (rank 28)

    # Helper: 2x2x2 Strassen (rank 7)
    # A: 2x2 (indices 0,1,2,3 row-major), B: 2x2 (indices 0,1,2,3 row-major)
    # C: 2x2, W index = col*2+row (col-major)
    # C00=0, C10=1, C01=2, C11=3
    def strassen_2x2x2():
        return [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),   # M1=(A00+A11)(B00+B11)
            ([(2,1),(3,1)], [(0,1)], [(1,1)]),                # M2=(A10+A11)B00
            ([(0,1)], [(1,1),(3,-1)], [(2,1)]),               # M3=A00(B01-B11)
            ([(3,1)], [(2,1),(0,-1)], [(1,1),(3,-1)]),        # M4=A11(B10-B00)
            ([(0,1),(1,1)], [(3,1)], [(2,1)]),                # M5=(A00+A01)B11
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,-1)]),        # M6=(A10-A00)(B00+B01)
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(2,-1)]),        # M7=(A01-A11)(B10+B11)
        ]

    # Build 4x4x4 Strassen (rank 49) by composing 2x2x2 Strassen
    # A: 4x4 (indices 0..15 row-major), B: 4x4 (indices 0..15 row-major)
    # C: 4x4, W index = col*4+row (col-major)
    #
    # Split into 2x2 blocks:
    # A00: rows 0-1, cols 0-1 -> A indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> A indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> A indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> A indices 10,11,14,15
    #
    # B00: rows 0-1, cols 0-1 -> B indices 0,1,4,5
    # B01: rows 0-1, cols 2-3 -> B indices 2,3,6,7
    # B10: rows 2-3, cols 0-1 -> B indices 8,9,12,13
    # B11: rows 2-3, cols 2-3 -> B indices 10,11,14,15
    #
    # C00: rows 0-1, cols 0-1 -> W indices 0,1,4,5
    # C01: rows 0-1, cols 2-3 -> W indices 8,9,12,13
    # C10: rows 2-3, cols 0-1 -> W indices 2,3,6,7
    # C11: rows 2-3, cols 2-3 -> W indices 10,11,14,15

    def build_4x4x4():
        sub = strassen_2x2x2()  # 7 multiplications

        # Mapping functions: sub-index (r*2+c) -> parent index
        # For A blocks (row-major within 4x4):
        a00 = [0,1,4,5]   # A00[r*2+c] -> a00[r*2+c]
        a01 = [2,3,6,7]
        a10 = [8,9,12,13]
        a11 = [10,11,14,15]

        b00 = [0,1,4,5]
        b01 = [2,3,6,7]
        b10 = [8,9,12,13]
        b11 = [10,11,14,15]

        # For W blocks (col-major within 4x4: col*4+row):
        # C00: col 0-1, row 0-1 -> [0,1,4,5]
        # C01: col 2-3, row 0-1 -> [8,9,12,13]
        # C10: col 0-1, row 2-3 -> [2,3,6,7]
        # C11: col 2-3, row 2-3 -> [10,11,14,15]
        w00 = [0,1,4,5]
        w01 = [8,9,12,13]
        w10 = [2,3,6,7]
        w11 = [10,11,14,15]

        results = []

        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # Sub U: indices 0-3 map to A00+A11 combined
        # Sub V: indices 0-3 map to B00+B11 combined
        # Sub W: indices 0-3 map to C00+C11 combined
        for (su, sv, sw) in sub:
            u = []
            for (idx, coeff) in su:
                u.append((a00[idx], coeff))
                u.append((a11[idx], coeff))
            v = []
            for (idx, coeff) in sv:
                v.append((b00[idx], coeff))
                v.append((b11[idx], coeff))
            w = []
            for (idx, coeff) in sw:
                w.append((w00[idx], coeff))
                w.append((w11[idx], coeff))
            results.append((u, v, w))

        # M2 = (A10+A11)B00 -> C10 += M2
        for (su, sv, sw) in sub:
            u = []
            for (idx, coeff) in su:
                u.append((a10[idx], coeff))
                u.append((a11[idx], coeff))
            v = []
            for (idx, coeff) in sv:
                v.append((b00[idx], coeff))
            w = []
            for (idx, coeff) in sw:
                w.append((w10[idx], coeff))
            results.append((u, v, w))

        # M3 = A00(B01-B11) -> C01 += M3
        for (su, sv, sw) in sub:
            u = []
            for (idx, coeff) in su:
                u.append((a00[idx], coeff))
            v = []
            for (idx, coeff) in sv:
                v.append((b01[idx], coeff))
                v.append((b11[idx], -coeff))
            w = []
            for (idx, coeff) in sw:
                w.append((w01[idx], coeff))
            results.append((u, v, w))

        # M4 = A11(B10-B00) -> C10 += M4, C11 -= M4
        for (su, sv, sw) in sub:
            u = []
            for (idx, coeff) in su:
                u.append((a11[idx], coeff))
            v = []
            for (idx, coeff) in sv:
                v.append((b10[idx], coeff))
                v.append((b00[idx], -coeff))
            w = []
            for (idx, coeff) in sw:
                w.append((w10[idx], coeff))
                w.append((w11[idx], -coeff))
            results.append((u, v, w))

        # M5 = (A00+A01)B11 -> C01 += M5
        for (su, sv, sw) in sub:
            u = []
            for (idx, coeff) in su:
                u.append((a00[idx], coeff))
                u.append((a01[idx], coeff))
            v = []
            for (idx, coeff) in sv:
                v.append((b11[idx], coeff))
            w = []
            for (idx, coeff) in sw:
                w.append((w01[idx], coeff))
            results.append((u, v, w))

        # M6 = (A10-A00)(B00+B01) -> C10 -= M6
        for (su, sv, sw) in sub:
            u = []
            for (idx, coeff) in su:
                u.append((a10[idx], coeff))
                u.append((a00[idx], -coeff))
            v = []
            for (idx, coeff) in sv:
                v.append((b00[idx], coeff))
                v.append((b01[idx], coeff))
            w = []
            for (idx, coeff) in sw:
                w.append((w10[idx], -coeff))
            results.append((u, v, w))

        # M7 = (A01-A11)(B10+B11) -> C01 -= M7
        for (su, sv, sw) in sub:
            u = []
            for (idx, coeff) in su:
                u.append((a01[idx], coeff))
                u.append((a11[idx], -coeff))
            v = []
            for (idx, coeff) in sv:
                v.append((b10[idx], coeff))
                v.append((b11[idx], coeff))
            w = []
            for (idx, coeff) in sw:
                w.append((w01[idx], -coeff))
            results.append((u, v, w))

        return results  # 49 multiplications

    # Build 4x4x2 decomposition (rank 28)
    # A: 4x4 (indices 0..15 row-major), B: 4x2 (indices 0..7 row-major: row*2+col)
    # C: 4x2, W index = col*4+row (col-major: col 0-1, row 0-3)
    #
    # Split A into 2x2 blocks:
    # A00: rows 0-1, cols 0-1 -> [0,1,4,5]
    # A01: rows 0-1, cols 2-3 -> [2,3,6,7]
    # A10: rows 2-3, cols 0-1 -> [8,9,12,13]
    # A11: rows 2-3, cols 2-3 -> [10,11,14,15]
    #
    # B: 4x2, split into 2x2 blocks:
    # B00: rows 0-1, cols 0-1 -> B indices 0,1,2,3 (row*2+col)
    # B10: rows 2-3, cols 0-1 -> B indices 4,5,6,7
    #
    # C: 4x2, split into 2x2 blocks:
    # C00: rows 0-1, cols 0-1 -> W indices: col*4+row, col 0-1, row 0-1 -> [0,1,4,5]
    # C10: rows 2-3, cols 0-1 -> W indices: col*4+row, col 0-1, row 2-3 -> [2,3,6,7]

    def build_4x4x2():
        sub = strassen_2x2x2()  # 7 multiplications

        a00 = [0,1,4,5]
        a01 = [2,3,6,7]
        a10 = [8,9,12,13]
        a11 = [10,11,14,15]

        b00 = [0,1,2,3]  # B00: rows 0-1, cols 0-1
        b10 = [4,5,6,7]  # B10: rows 2-3, cols 0-1

        w00 = [0,1,4,5]  # C00: col 0-1, row 0-1
        w10 = [2,3,6,7]  # C10: col 0-1, row 2-3

        results = []

        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        #
        # For C00: use 2x2x2 Strassen on (A00+A01, B00+B10) etc.
        # Actually, C00 = A00*B00 + A01*B10
        # This is a 2x2 = 2x2 * 2x2 + 2x2 * 2x2
        # We can use Strassen on the combined problem.
        #
        # Let me think of it as: C00 = [A00 A01] * [B00; B10]
        # This is a 2x4 * 4x2 = 2x2 multiplication.
        # We can do this with 7 multiplications using Strassen-like approach.
        #
        # Actually, the simplest is to just do it naively: 4 multiplications for C00
        # and 4 for C10, total 8. But we want rank 28, so we need 14 for each 2x2 output.
        #
        # Wait, 4x4x2 should have rank at most 4*4*2=32 naively.
        # With Strassen-like: split into 2 2x4x2 sub-problems, each rank 14.
        # 2x4x2: A is 2x4, B is 4x2, C is 2x2.
        # This can be done with 7 multiplications (Strassen for 2x2 output).
        #
        # Let me just do it naively for simplicity: 8 multiplications for 4x4x2.
        # Total: 49 + 8 = 57. That's better than 84!
        #
        # Actually, let me do it properly with rank 14 per 2x4x2.
        # 2x4x2: A (2x4), B (4x2), C (2x2)
        # C00 = A[0,:] @ B[:,0] = sum_j A[0,j]*B[j,0]
        # C01 = A[0,:] @ B[:,1]
        # C10 = A[1,:] @ B[:,0]
        # C11 = A[1,:] @ B[:,1]
        #
        # This is just 4 dot products, but we can use Strassen:
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C00+=M1, C11+=M1
        # M2 = (A[1,0]+A[1,1])*B[0,0] -> C10+=M2
        # M3 = A[0,0]*(B[0,1]-B[1,1]) -> C01+=M3
        # M4 = A[1,1]*(B[1,0]-B[0,0]) -> C10+=M4, C11-=M4
        # M5 = (A[0,0]+A[0,1])*B[1,1] -> C01+=M5
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C10-=M6
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C01-=M7
        #
        # Wait, that's for 2x2x2. For 2x4x2, we need to handle the middle dimension.
        # A is 2x4, B is 4x2.
        # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
        #
        # This is like multiplying a 2x4 by 4x2. We can split A into [A_left(2x2) A_right(2x2)]
        # and B into [B_top(2x2); B_bottom(2x2)].
        # C = A_left @ B_top + A_right @ B_bottom
        # Each is a 2x2x2 multiplication (rank 7).
        # Total: 14 multiplications for 2x4x2.
        #
        # So for 4x4x2: split A into top 2 rows and bottom 2 rows.
        # C_top (2x2) = A_top(2x4) @ B(4x2) -> 14 mults
        # C_bottom (2x2) = A_bottom(2x4) @ B(4x2) -> 14 mults
        # Total: 28 mults.

        # For C_top = A_top(2x4) @ B(4x2):
        # A_top: rows 0-1 of A, all 4 cols -> A indices 0,1,2,3,4,5,6,7
        #   A_top_left (2x2): rows 0-1, cols 0-1 -> [0,1,4,5]
        #   A_top_right (2x2): rows 0-1, cols 2-3 -> [2,3,6,7]
        # B: 4x2 -> B indices 0,1,2,3,4,5,6,7
        #   B_top (2x2): rows 0-1, cols 0-1 -> [0,1,2,3]
        #   B_bottom (2x2): rows 2-3, cols 0-1 -> [4,5,6,7]
        # C_top (2x2): rows 0-1, cols 0-1 of C -> W indices [0,1,4,5]

        # For C_bottom = A_bottom(2x4) @ B(4x2):
        # A_bottom: rows 2-3 of A, all 4 cols -> A indices 8,9,10,11,12,13,14,15
        #   A_bottom_left (2x2): rows 2-3, cols 0-1 -> [8,9,12,13]
        #   A_bottom_right (2x2): rows 2-3, cols 2-3 -> [10,11,14,15]
        # B: same as above
        # C_bottom (2x2): rows 2-3, cols 0-1 of C -> W indices [2,3,6,7]

        # Helper: 2x2x2 Strassen for (2x2 A, 2x2 B, 2x2 C)
        # A indices: a0,a1,a2,a3 (row-major)
        # B indices: b0,b1,b2,b3 (row-major)
        # C indices (W): w0,w1,w2,w3 (col-major: col*2+row)
        def strassen_2x2x2_mapped(a_map, b_map, w_map):
            sub = strassen_2x2x2()
            results = []
            for (su, sv, sw) in sub:
                u = [(a_map[idx], coeff) for (idx, coeff) in su]
                v = [(b_map[idx], coeff) for (idx, coeff) in sv]
                w = [(w_map[idx], coeff) for (idx, coeff) in sw]
                results.append((u, v, w))
            return results

        # C_top = A_top_left @ B_top + A_top_right @ B_bottom
        # M1: (A_top_left + A_top_right) @ (B_top + B_bottom) -> C_top += M1
        # M2: (A_top_left + A_top_right) @ B_top -> C_top += M2 (wait, this isn't standard Strassen)
        #
        # Actually, C_top = A_top_left @ B_top + A_top_right @ B_bottom
        # This is a sum of two 2x2x2 multiplications.
        # We can do each with 7 multiplications, total 14.
        # Or use a combined Strassen approach.
        #
        # Let me just do it as two separate 2x2x2 Strassen:
        # Part 1: A_top_left @ B_top -> contributes to C_top
        # Part 2: A_top_right @ B_bottom -> contributes to C_top

        # Part 1: A_top_left @ B_top
        a_tl = [0,1,4,5]
        b_t = [0,1,2,3]
        w_ct = [0,1,4,5]
        results.extend(strassen_2x2x2_mapped(a_tl, b_t, w_ct))

        # Part 2: A_top_right @ B_bottom
        a_tr = [2,3,6,7]
        b_b = [4,5,6,7]
        results.extend(strassen_2x2x2_mapped(a_tr, b_b, w_ct))

        # C_bottom = A_bottom_left @ B_top + A_bottom_right @ B_bottom
        # Part 3: A_bottom_left @ B_top
        a_bl = [8,9,12,13]
        w_cb = [2,3,6,7]
        results.extend(strassen_2x2x2_mapped(a_bl, b_t, w_cb))

        # Part 4: A_bottom_right @ B_bottom
        a_br = [10,11,14,15]
        results.extend(strassen_2x2x2_mapped(a_br, b_b, w_cb))

        return results  # 28 multiplications

    # Build the full 4x4x6 decomposition
    U4x4x4 = build_4x4x4()  # 49 multiplications
    U4x4x2 = build_4x4x2()  # 28 multiplications

    total_rank = len(U4x4x4) + len(U4x4x2)  # 77

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # Fill in 4x4x4 part (cols 0-3 of B, cols 0-3 of C)
    for r, (u_list, v_list, w_list) in enumerate(U4x4x4):
        for (idx, coeff) in u_list:
            U[idx, r] += coeff
        for (idx, coeff) in v_list:
            V[idx, r] += coeff
        for (idx, coeff) in w_list:
            W[idx, r] += coeff

    # Fill in 4x4x2 part (cols 4-5 of B, cols 4-5 of C)
    # B indices: 4x6, row-major: row*6+col
    # B_right: cols 4-5, so B indices for row j, col k (k=4,5) -> j*6+k
    # In the 4x4x2 sub-problem, B is 4x2 with indices row*2+col
    # B_right[row, col] in 4x2 -> B_full[row, col+4] in 4x6
    # So 4x2 index (row*2+col) -> 4x6 index (row*6 + col + 4)
    #
    # C indices: 4x6, W index = col*4+row (col-major)
    # C_right: cols 4-5, so W index for col k (k=4,5), row i -> k*4+i
    # In the 4x4x2 sub-problem, C is 4x2 with W index = col*4+row (col=0,1)
    # C_right[col, row] in 4x2 -> C_full[col+4, row] in 4x6
    # So 4x2 W index (col*4+row) -> 4x6 W index ((col+4)*4+row)

    for r, (u_list, v_list, w_list) in enumerate(U4x4x2):
        offset = len(U4x4x4)
        for (idx, coeff) in u_list:
            U[idx, offset + r] += coeff
        for (idx, coeff) in v_list:
            # Map 4x2 B index to 4x6 B index
            row = idx // 2
            col = idx % 2
            full_idx = row * 6 + col + 4
            V[full_idx, offset + r] += coeff
        for (idx, coeff) in w_list:
            # Map 4x2 W index to 4x6 W index
            col = idx // 4
            row = idx % 4
            full_idx = (col + 4) * 4 + row
            W[full_idx, offset + r] += coeff

    # Verify
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)

    if not np.allclose(T_decomp, T_expected, atol=1e-10):
        # Fallback to naive
        rank_naive = n * m * p
        U_naive = np.zeros((n * m, rank_naive), dtype=int)
        V_naive = np.zeros((m * p, rank_naive), dtype=int)
        W_naive = np.zeros((p * n, rank_naive), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j][col] = 1
                    V_naive[j*p+k][col] = 1
                    W_naive[k*n+i][col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
