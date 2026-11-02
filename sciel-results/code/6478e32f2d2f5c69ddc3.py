"""Rewritten program: Strassen block decomposition + rank-term merging for 4x4*4x5."""
import time
from collections import defaultdict

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 20 x rank
    # W: (p*n) x rank = 20 x rank
    # Convention: W[k*n+i, r]

    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition as baseline
    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p  # 80

    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # r=0: M1 = (A11+A22)*(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # r=1: M2 = (A21+A22)*B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1

        # r=2: M3 = A11*(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1

        # r=3: M4 = A22*(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        # r=4: M5 = (A11+A12)*B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1

        return U, V, W

    def strassen_2x3():
        """Return U, V, W for 2x2 * 2x3 multiplication.
        U: 4x?, V: 6x?, W: 6x?
        Convention: U[i*2+j, r], V[j*3+k, r], W[k*2+i, r]
        Uses 7 terms: 4 from Strassen-like + 3 extra for the 3rd column.
        Actually, let's use a direct approach: 2x2*2x3 can be done in 7 terms
        by sharing the 2x2 Strassen structure for cols 0,1 and adding col 2.
        """
        # For 2x2 * 2x3:
        # C[i][k] = sum_j A[i][j] * B[j][k]
        # We can use Strassen for the 2x2 part (cols 0,1) and add col 2 separately
        # But that gives 7+6=13. Let's try a smarter approach.
        # Actually, 2x2*2x3 has tensor rank at most 8 (schoolbook).
        # Let's just use schoolbook: 2*2*3 = 12 terms.
        # Or we can try to be clever. For now, schoolbook.
        rank = 2 * 2 * 3
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W

    def test_strassen():
        """Test Strassen 2x2 decomposition."""
        U, V, W = strassen_2x2()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def test_strassen_2x3():
        """Test 2x2*2x3 schoolbook decomposition."""
        U, V, W = strassen_2x3()
        rank = len(U[0])
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0
                    for r in range(rank):
                        s += U[i*2+j][r] * V[j*3+k][r] * W[k*2+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if test_strassen():
        sU, sV, sW = strassen_2x2()

        def build_strassen_block():
            """Build 4x4*4x5 using Strassen for 2x2*2x2 and schoolbook for 2x2*2x3."""
            U_list = []
            V_list = []
            W_list = []

            # Strassen A index -> full U index for each A block
            # A11: rows 0-1, cols 0-1 -> U[i*4+j]
            a11 = [0, 1, 4, 5]
            # A12: rows 0-1, cols 2-3 -> U[i*4+j]
            a12 = [2, 3, 6, 7]
            # A21: rows 2-3, cols 0-1 -> U[i*4+j]
            a21 = [8, 9, 12, 13]
            # A22: rows 2-3, cols 2-3 -> U[i*4+j]
            a22 = [10, 11, 14, 15]

            # Strassen B index -> full V index for each B block
            # B11: rows 0-1, cols 0-1 -> V[j*5+k]
            b11 = [0, 1, 5, 6]
            # B12: rows 0-1, cols 2-4 -> V[j*5+k] (but Strassen needs 2x2, use cols 0-1 of B12)
            # For Strassen on B12 (2x3), we only use the first 2 columns as a 2x2
            # Actually, Strassen requires square blocks. For 2x2*2x3, we can't directly use Strassen.
            # We'll use schoolbook for 2x2*2x3 blocks.

            # Strassen W index -> full W index for each C block
            # C11: rows 0-1, cols 0-1 -> W[k*4+i]
            c11 = [0, 4, 1, 5]
            # C21: rows 2-3, cols 0-1 -> W[k*4+i]
            c21 = [2, 6, 3, 7]

            def add_strassen(a_map, b_map, c_map):
                for r in range(7):
                    U_new = [0]*16
                    V_new = [0]*20
                    W_new = [0]*20
                    for si in range(4):
                        if sU[si][r] != 0:
                            U_new[a_map[si]] = sU[si][r]
                    for sj in range(4):
                        if sV[sj][r] != 0:
                            V_new[b_map[sj]] = sV[sj][r]
                    for sk in range(4):
                        if sW[sk][r] != 0:
                            W_new[c_map[sk]] = sW[sk][r]
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

            # C11 = A11*B11 + A12*B21
            add_strassen(a11, b11, c11)  # A11*B11
            # A12*B21: A12 is 2x2, B21 is 2x2 (rows 2-3, cols 0-1)
            b21 = [10, 11, 15, 16]
            add_strassen(a12, b21, c11)  # A12*B21

            # C21 = A21*B11 + A22*B21
            add_strassen(a21, b11, c21)  # A21*B11
            add_strassen(a22, b21, c21)  # A22*B21

            # C12 = A11*B12 + A12*B22 (2x3 blocks, use schoolbook)
            # A11: rows 0-1, cols 0-1
            # B12: rows 0-1, cols 2-4
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+j] = 1
                        V_new[j*5+(2+k)] = 1
                        W_new[(2+k)*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A12*B22: A12 rows 0-1 cols 2-3, B22 rows 2-3 cols 2-4
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+(2+j)] = 1
                        V_new[(2+j)*5+(2+k)] = 1
                        W_new[(2+k)*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # C22 = A21*B12 + A22*B22 (2x3 blocks, use schoolbook)
            # A21: rows 2-3, cols 0-1
            # B12: rows 0-1, cols 2-4
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+j] = 1
                        V_new[j*5+(2+k)] = 1
                        W_new[(2+k)*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A22*B22: A22 rows 2-3 cols 2-3, B22 rows 2-3 cols 2-4
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+(2+j)] = 1
                        V_new[(2+j)*5+(2+k)] = 1
                        W_new[(2+k)*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            return {"U": U_list, "V": V_list, "W": W_list}

        strassen_result = build_strassen_block()
        rank = len(strassen_result["U"])

        if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
            if rank < best_rank:
                best = strassen_result
                best_rank = rank

    # Try to reduce rank by merging identical terms
    def merge_terms(U, V, W):
        """Merge terms that have identical (U, V, W) patterns by summing coefficients.
        Returns (new_U, new_V, new_W) or None if no reduction possible.
        Coefficients are absorbed into the W matrix (or U) since we can scale.
        """
        term_dict = {}
        term_order = []
        for r in range(len(U[0])):
            u_key = tuple(U[i][r] for i in range(16))
            v_key = tuple(V[j][r] for j in range(20))
            w_key = tuple(W[k][r] for k in range(20))
            key = (u_key, v_key, w_key)
            if key in term_dict:
                term_dict[key] += 1
            else:
                term_dict[key] = 1
                term_order.append(key)

        # Check if we can reduce
        if len(term_order) >= len(U[0]):
            return None

        new_U = []
        new_V = []
        new_W = []
        for key in term_order:
            coeff = term_dict[key]
            u_key, v_key, w_key = key
            new_U.append(list(u_key))
            new_V.append(list(v_key))
            # Absorb coefficient into W
            new_W.append([w * coeff for w in w_key])

        return new_U, new_V, new_W

    # Try merging on the current best
    if time.time() - start_time < deadline:
        merged = merge_terms(best["U"], best["V"], best["W"])
        if merged is not None:
            mU, mV, mW = merged
            if verify(mU, mV, mW):
                if len(mU) < best_rank:
                    best = {"U": mU, "V": mV, "W": mW}
                    best_rank = len(mU)

    # Try Strassen for 2x2*2x3 blocks to reduce schoolbook terms
    if time.time() - start_time < deadline:
        def build_strassen_2x3_block():
            """Build 4x4*4x5 using Strassen 2x2*2x2 for square parts and
            optimized 2x2*2x3 for rectangular parts."""
            U_list = []
            V_list = []
            W_list = []

            a11 = [0, 1, 4, 5]
            a12 = [2, 3, 6, 7]
            a21 = [8, 9, 12, 13]
            a22 = [10, 11, 14, 15]

            b11 = [0, 1, 5, 6]
            b21 = [10, 11, 15, 16]
            c11 = [0, 4, 1, 5]
            c21 = [2, 6, 3, 7]

            sU, sV, sW = strassen_2x2()

            def add_strassen(a_map, b_map, c_map):
                for r in range(7):
                    U_new = [0]*16
                    V_new = [0]*20
                    W_new = [0]*20
                    for si in range(4):
                        if sU[si][r] != 0:
                            U_new[a_map[si]] = sU[si][r]
                    for sj in range(4):
                        if sV[sj][r] != 0:
                            V_new[b_map[sj]] = sV[sj][r]
                    for sk in range(4):
                        if sW[sk][r] != 0:
                            W_new[c_map[sk]] = sW[sk][r]
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

            # Square 2x2*2x2 blocks using Strassen (7 terms each)
            add_strassen(a11, b11, c11)  # A11*B11
            add_strassen(a12, b21, c11)  # A12*B21
            add_strassen(a21, b11, c21)  # A21*B11
            add_strassen(a22, b21, c21)  # A22*B21

            # Rectangular 2x2*2x3 blocks using schoolbook (12 terms each)
            # A11*B12: A11 is 2x2 (rows 0-1, cols 0-1), B12 is 2x3 (rows 0-1, cols 2-4)
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+j] = 1
                        V_new[j*5+(2+k)] = 1
                        W_new[(2+k)*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A12*B22: A12 is 2x2 (rows 0-1, cols 2-3), B22 is 2x3 (rows 2-3, cols 2-4)
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+(2+j)] = 1
                        V_new[(2+j)*5+(2+k)] = 1
                        W_new[(2+k)*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A21*B12: A21 is 2x2 (rows 2-3, cols 0-1), B12 is 2x3 (rows 0-1, cols 2-4)
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+j] = 1
                        V_new[j*5+(2+k)] = 1
                        W_new[(2+k)*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A22*B22: A22 is 2x2 (rows 2-3, cols 2-3), B22 is 2x3 (rows 2-3, cols 2-4)
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+(2+j)] = 1
                        V_new[(2+j)*5+(2+k)] = 1
                        W_new[(2+k)*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            return {"U": U_list, "V": V_list, "W": W_list}

        s23_result = build_strassen_2x3_block()
        s23_rank = len(s23_result["U"])
        if verify(s23_result["U"], s23_result["V"], s23_result["W"]):
            if s23_rank < best_rank:
                best = s23_result
                best_rank = s23_rank

            # Try merging on this result too
            if time.time() - start_time < deadline:
                merged23 = merge_terms(s23_result["U"], s23_result["V"], s23_result["W"])
                if merged23 is not None:
                    m23U, m23V, m23W = merged23
                    if verify(m23U, m23V, m23W):
                        if len(m23U) < best_rank:
                            best = {"U": m23U, "V": m23V, "W": m23W}
                            best_rank = len(m23U)

    # Try a different block partition: split into 4x2 * 2x5 style
    # Actually, let's try splitting the 4x4 into a 2x2 and 2x2 row split
    # and the 4x5 into 4x2 and 4x3 column splits
    if time.time() - start_time < deadline:
        # Also try: use Strassen on 2x2*2x2 sub-blocks where possible
        # and a 3-term decomposition for 2x2*2x1 (which is just schoolbook = 4 terms,
        # but 2x2*2x1 has rank at most 4, so no improvement)
        pass
        def build_alt_block():
            """Alternative: split A by rows (2x2 blocks), B by cols (4x2, 4x3)."""
            # A = [A_top (2x4); A_bot (2x4)]
            # B = [B_left (4x2) | B_right (4x3)]
            # C = [C_top_left (2x2)  C_top_right (2x3)]
            #     [C_bot_left (2x2)  C_bot_right (2x3)]
            # C_top_left = A_top * B_left
            # C_top_right = A_top * B_right
            # C_bot_left = A_bot * B_left
            # C_bot_right = A_bot * B_right

            U_list = []
            V_list = []
            W_list = []

            # A_top: rows 0-1, cols 0-3 -> U[i*4+j] for i=0,1
            # A_bot: rows 2-3, cols 0-3 -> U[i*4+j] for i=2,3
            # B_left: rows 0-3, cols 0-1 -> V[j*5+k] for k=0,1
            # B_right: rows 0-3, cols 2-4 -> V[j*5+k] for k=2,3,4

            # C_top_left: rows 0-1, cols 0-1 -> W[k*4+i] for i=0,1 k=0,1
            # C_top_right: rows 0-1, cols 2-4 -> W[k*4+i] for i=0,1 k=2,3,4
            # C_bot_left: rows 2-3, cols 0-1 -> W[k*4+i] for i=2,3 k=0,1
            # C_bot_right: rows 2-3, cols 2-4 -> W[k*4+i] for i=2,3 k=2,3,4

            # Schoolbook for each sub-product
            # A_top * B_left: 2x4 * 4x2 = 16 terms
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+j] = 1
                        V_new[j*5+k] = 1
                        W_new[k*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A_top * B_right: 2x4 * 4x3 = 24 terms
            for i in range(2):
                for j in range(4):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+j] = 1
                        V_new[j*5+(2+k)] = 1
                        W_new[(2+k)*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A_bot * B_left: 2x4 * 4x2 = 16 terms
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+j] = 1
                        V_new[j*5+k] = 1
                        W_new[k*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A_bot * B_right: 2x4 * 4x3 = 24 terms
            for i in range(2):
                for j in range(4):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+j] = 1
                        V_new[j*5+(2+k)] = 1
                        W_new[(2+k)*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            return {"U": U_list, "V": V_list, "W": W_list}

        alt_result = build_alt_block()
        alt_rank = len(alt_result["U"])
        if verify(alt_result["U"], alt_result["V"], alt_result["W"]):
            if alt_rank < best_rank:
                best = alt_result
                best_rank = alt_rank

            # Try merging on alt result
            if time.time() - start_time < deadline:
                merged_alt = merge_terms(alt_result["U"], alt_result["V"], alt_result["W"])
                if merged_alt is not None:
                    maU, maV, maW = merged_alt
                    if verify(maU, maV, maW):
                        if len(maU) < best_rank:
                            best = {"U": maU, "V": maV, "W": maW}
                            best_rank = len(maU)

    # Final: try a 3x1 split on B to get more Strassen opportunities
    # 4x4 * 4x5 = 4x4*4x3 + 4x4*4x2
    # 4x4*4x2: split into 2x2 blocks -> 4 Strassen 2x2*2x2 = 4*7 = 28
    # 4x4*4x3: split into 2x2 blocks -> 2 Strassen 2x2*2x2 + 2x2*2x1 parts
    # Actually this is getting complex. Let's try a 1x4 row split.
    if time.time() - start_time < deadline:
        def build_row_split():
            """Split A by rows: 1x4, 1x4, 1x4, 1x4 (each row of A times B).
            Each 1x4 * 4x5 = 20 terms (schoolbook). Total = 80. No improvement.
            Instead, split into 2x4 * 4x5 blocks: 2 blocks of 2x4*4x5.
            2x4*4x5: split B into 4x2 and 4x3.
            2x4*4x2: 2*4*2 = 16 terms
            2x4*4x3: 2*4*3 = 24 terms
            Total per 2x4 block: 40. Two blocks: 80. Same as schoolbook.
            """
            # This doesn't help. Let's skip.
            return None

    return best

# EVOLVE-BLOCK-END
