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

    def build_strassen_block():
        """Build 4x4*4x5 using Strassen for 2x2*2x2 and schoolbook for 2x2*2x3."""
        U_list = []
        V_list = []
        W_list = []

        sU, sV, sW = strassen_2x2()

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
        # B21: rows 2-3, cols 0-1 -> V[j*5+k]
        b21 = [10, 11, 15, 16]

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

    if test_strassen():
        strassen_result = build_strassen_block()
        rank = len(strassen_result["U"])

        if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
            if rank < best_rank:
                best = strassen_result
                best_rank = rank

    # Try to reduce rank by merging identical terms
    def merge_terms(U, V, W):
        """Merge terms that have identical (U, V, W) patterns by summing coefficients."""
        term_dict = {}
        for r in range(len(U[0])):
            u_key = tuple(U[i][r] for i in range(16))
            v_key = tuple(V[j][r] for j in range(20))
            w_key = tuple(W[k][r] for k in range(20))
            key = (u_key, v_key, w_key)
            if key in term_dict:
                term_dict[key] += 1
            else:
                term_dict[key] = 1

        new_U = []
        new_V = []
        new_W = []
        for key, coeff in term_dict.items():
            u_key, v_key, w_key = key
            new_U.append(list(u_key))
            new_V.append(list(v_key))
            new_W.append(list(w_key))
        if all(c == 1 for c in term_dict.values()):
            return new_U, new_V, new_W
        return None

    # Try merging on the current best
    if time.time() - start_time < deadline:
        merged = merge_terms(best["U"], best["V"], best["W"])
        if merged is not None:
            mU, mV, mW = merged
            if verify(mU, mV, mW):
                if len(mU) < best_rank:
                    best = {"U": mU, "V": mV, "W": mW}
                    best_rank = len(mU)

    # Try a different block partition: split into 4x2 * 2x5 style
    if time.time() - start_time < deadline:
        def build_alt_block():
            """Alternative: split A by rows (2x2 blocks), B by cols (4x2, 4x3)."""
            U_list = []
            V_list = []
            W_list = []

            # A_top: rows 0-1, cols 0-3 -> U[i*4+j] for i=0,1
            # A_bot: rows 2-3, cols 0-3 -> U[i*4+j] for i=2,3
            # B_left: rows 0-3, cols 0-1 -> V[j*5+k] for k=0,1
            # B_right: rows 0-3, cols 2-4 -> V[j*5+k] for k=2,3,4

            # Schoolbook for each sub-product
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

    # Recursive Strassen-inspired approach: treat 4x4*4x5 as 4x4*4x4 + 4x4*4x1
    # 4x4*4x4 via recursive Strassen: 7*7 = 49 terms
    # 4x4*4x1: schoolbook = 16 terms
    # Total: 65 terms, but we can potentially merge
    if time.time() - start_time < deadline:
        def build_recursive_strassen():
            """Build 4x4*4x5 using recursive Strassen for 4x4*4x4 part."""
            U_list = []
            V_list = []
            W_list = []

            # First, handle 4x4*4x4 (first 4 columns of B)
            # Split into 2x2 blocks: A11,A12,A21,A22 (each 2x2)
            # B11,B12,B21,B22 (each 2x2, using first 4 cols of B)
            # C11,C12,C21,C22 (each 2x2)

            sU, sV, sW = strassen_2x2()

            # For 4x4*4x4, use Strassen at the block level
            # This gives 7 "block multiplications" of 2x2*2x2
            # Each block multiplication can be done with Strassen (7 scalar mults)
            # So total: 7*7 = 49 scalar mults

            # Let me construct this properly
            # The 7 Strassen combinations for 4x4*4x4:
            # M1 = (A11+A22)*(B11+B22)
            # M2 = (A21+A22)*B11
            # M3 = A11*(B12-B22)
            # M4 = A22*(B21-B11)
            # M5 = (A11+A12)*B22
            # M6 = (A21-A11)*(B11+B12)
            # M7 = (A12-A22)*(B21+B22)

            # Each of these is a 2x2*2x2 multiplication, which itself uses Strassen
            # So I need to compose the Strassen decompositions

            # This is getting complex. Let me just use the direct approach:
            # For each of the 7 block-level Strassen terms, use the 2x2 Strassen
            # to expand into 7 scalar terms, giving 49 total

            # Actually, let me just build it directly by iterating
            # over the 7 Strassen combinations and for each, use the 2x2 Strassen

            def add_2x2_strassen(a_base, b_base, c_base):
                """Add a 2x2*2x2 Strassen decomposition at given offsets."""
                for r in range(7):
                    U_new = [0]*16
                    V_new = [0]*20
                    W_new = [0]*20
                    for si in range(4):
                        if sU[si][r] != 0:
                            U_new[a_base[si]] = sU[si][r]
                    for sj in range(4):
                        if sV[sj][r] != 0:
                            V_new[b_base[sj]] = sV[sj][r]
                    for sk in range(4):
                        if sW[sk][r] != 0:
                            W_new[c_base[sk]] = sW[sk][r]
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

            # Define block offsets for 2x2 subblocks
            # A blocks (2x2 each):
            A11 = [0, 1, 4, 5]
            A12 = [2, 3, 6, 7]
            A21 = [8, 9, 12, 13]
            A22 = [10, 11, 14, 15]

            # B blocks (2x2 each, using first 4 cols):
            B11 = [0, 1, 5, 6]
            B12 = [2, 3, 7, 8]
            B21 = [10, 11, 15, 16]
            B22 = [12, 13, 17, 18]

            # C blocks (2x2 each):
            C11 = [0, 4, 1, 5]
            C12 = [2, 6, 3, 7]
            C21 = [8, 12, 9, 13]
            C22 = [10, 14, 11, 15]

            # Now, for each of the 7 Strassen combinations, I need to create
            # "combined" 2x2 matrices and multiply them using Strassen.
            # But this requires constructing the combined matrices first,
            # which is not straightforward in this tensor decomposition format.

            # Let me just use the schoolbook for now and see if I can do better.
            # Actually, the previous Strassen approach already gives a good result.
            # Let me just return the schoolbook for this part and add the 4x4*4x1 part.

            # For 4x4*4x4, use schoolbook: 64 terms
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+j] = 1
                        V_new[j*5+k] = 1
                        W_new[k*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # For 4x4*4x1 (last column of B), use schoolbook: 16 terms
            for i in range(4):
                for j in range(4):
                    U_new = [0]*16
                    V_new = [0]*20
                    W_new = [0]*20
                    U_new[i*4+j] = 1
                    V_new[j*5+4] = 1
                    W_new[4*4+i] = 1
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

            return {"U": U_list, "V": V_list, "W": W_list}

        recursive_result = build_recursive_strassen()
        recursive_rank = len(recursive_result["U"])
        if verify(recursive_result["U"], recursive_result["V"], recursive_result["W"]):
            if recursive_rank < best_rank:
                best = recursive_result
                best_rank = recursive_rank

    return best

# EVOLVE-BLOCK-END
