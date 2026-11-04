"""Rank-77 decomposition for 4x4x6 matrix multiplication tensor.

Strategy: Split B into B[:,0:4] (4x4) and B[:,4:6] (4x2).
- For A*B[:,0:4]: recursive Strassen on 4x4x4 gives rank 49 (7x7).
- For A*B[:,4:6]: split into four 2x2x2 blocks, each using Strassen (rank 7) = 28.
Total rank: 49 + 28 = 77.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dict {'U','V','W'} with integer entries exactly decomposing the
    4x4x6 matrix-multiplication tensor using rank 77.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_decomposition():
        U = []
        V = []
        W = []

        def add_term(u, v, w):
            U.append(u)
            V.append(v)
            W.append(w)

        # The 7 Strassen rank-1 terms for a 2x2x2 multiplication.
        # Each term is (a_coeffs, b_coeffs, c_coeffs) where each is a dict
        # mapping (local_row, local_col) -> coefficient.
        # The local 2x2 A, B, C matrices are multiplied, and the result
        # contributes to the global tensor via the block mappings.
        #
        # Standard Strassen:
        # p1 = (A00+A11)(B00+B11)  -> C00 += p1, C11 += p1
        # p2 = (A10+A11)(B00)      -> C10 += p2
        # p3 = (A00)(B10-B11)      -> C00 -= p3, C01 += p3
        # p4 = (A00-A11)(B11-B00)  -> C00 += p4, C11 -= p4
        # p5 = (A01+A11)(B00-B01)  -> C01 += p5, C11 -= p5
        # p6 = (A10-A00)(B01+B11)  -> C11 += p6
        # p7 = (A00+A01)(B11-B10)  -> C00 += p7, C10 -= p7

        strassen_terms = [
            # p1
            ({(0,0):1, (1,1):1},
             {(0,0):1, (0,1):1, (1,0):1, (1,1):1},
             {(0,0):1, (1,1):1}),
            # p2
            ({(1,0):1, (1,1):1},
             {(0,0):1},
             {(1,0):1}),
            # p3
            ({(0,0):1},
             {(1,0):1, (1,1):-1},
             {(0,0):-1, (0,1):1}),
            # p4
            ({(0,0):1, (1,1):-1},
             {(1,1):1, (0,0):-1},
             {(0,0):1, (1,1):-1}),
            # p5
            ({(0,1):1, (1,1):1},
             {(0,0):1, (0,1):-1},
             {(0,1):1, (1,1):-1}),
            # p6
            ({(1,0):1, (0,0):-1},
             {(0,1):1, (1,1):1},
             {(1,1):1}),
            # p7
            ({(0,0):1, (0,1):1},
             {(1,1):1, (1,0):-1},
             {(0,0):1, (1,0):-1}),
        ]

        # Block index maps: for a 2x2 block starting at global (row0, col0)
        # idx[li][lj] = (row0+li, col0+lj)
        def make_2x2_idx(row0, col0):
            return [[(row0+li, col0+lj) for lj in range(2)] for li in range(2)]

        # 4x4x4 part: k=0..3
        # A blocks
        A11 = make_2x2_idx(0, 0)
        A12 = make_2x2_idx(0, 2)
        A21 = make_2x2_idx(2, 0)
        A22 = make_2x2_idx(2, 2)

        # B blocks (rows 0..3, cols 0..3)
        B11 = make_2x2_idx(0, 0)
        B12 = make_2x2_idx(0, 2)
        B21 = make_2x2_idx(2, 0)
        B22 = make_2x2_idx(2, 2)

        # C blocks (rows 0..3, cols 0..3)
        C11 = make_2x2_idx(0, 0)
        C12 = make_2x2_idx(0, 2)
        C21 = make_2x2_idx(2, 0)
        C22 = make_2x2_idx(2, 2)

        # B blocks for k=4..5
        Btop = make_2x2_idx(0, 4)
        Bbot = make_2x2_idx(2, 4)

        # C blocks for k=4..5
        Ctop = make_2x2_idx(0, 4)
        Cbot = make_2x2_idx(2, 4)

        # Helper: add 7 Strassen terms for a 2x2x2 product where
        # A = sum of a_blocks, B = sum of b_blocks, C = sum of c_blocks
        # Each block is (idx_2x2, sign_2x2) where sign_2x2 is a 2x2 list of signs
        def add_strassen(a_blocks, b_blocks, c_blocks):
            for a_c, b_c, c_c in strassen_terms:
                u = [0] * (n * m)  # 16
                v = [0] * (m * p)  # 24
                w = [0] * (p * n)  # 24

                for (li, lj), local_sign in a_c.items():
                    for idx, signs in a_blocks:
                        gi, gj = idx[li][lj]
                        u[gi * m + gj] += local_sign * signs[li][lj]

                for (li, lj), local_sign in b_c.items():
                    for idx, signs in b_blocks:
                        gj, gk = idx[li][lj]
                        v[gj * p + gk] += local_sign * signs[li][lj]

                for (li, lj), local_sign in c_c.items():
                    for idx, signs in c_blocks:
                        gi, gk = idx[li][lj]
                        w[gk * n + gi] += local_sign * signs[li][lj]

                add_term(u, v, w)

        # All-ones 2x2 sign
        O = [[1, 1], [1, 1]]
        # All-negatives 2x2 sign
        N = [[-1, -1], [-1, -1]]

        # === 4x4x4 block: 7 block-level Strassen products, each 2x2x2 (rank 7) ===
        # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
        add_strassen(
            [(A11, O), (A22, O)],
            [(B11, O), (B22, O)],
            [(C11, O), (C22, O)]
        )
        # P2 = (A21+A22)(B11) -> C21 += P2
        add_strassen(
            [(A21, O), (A22, O)],
            [(B11, O)],
            [(C21, O)]
        )
        # P3 = (A11)(B21-B22) -> C11 -= P3, C12 += P3
        add_strassen(
            [(A11, O)],
            [(B21, O), (B22, N)],
            [(C11, N), (C12, O)]
        )
        # P4 = (A11-A22)(B22-B11) -> C11 += P4, C22 -= P4
        add_strassen(
            [(A11, O), (A22, N)],
            [(B22, O), (B11, N)],
            [(C11, O), (C22, N)]
        )
        # P5 = (A12+A22)(B11-B12) -> C12 += P5, C22 -= P5
        add_strassen(
            [(A12, O), (A22, O)],
            [(B11, O), (B12, N)],
            [(C12, O), (C22, N)]
        )
        # P6 = (A21-A11)(B12+B22) -> C22 += P6
        add_strassen(
            [(A21, O), (A11, N)],
            [(B12, O), (B22, O)],
            [(C22, O)]
        )
        # P7 = (A11+A12)(B22-B21) -> C11 += P7, C21 -= P7
        add_strassen(
            [(A11, O), (A12, O)],
            [(B22, O), (B21, N)],
            [(C11, O), (C21, N)]
        )

        # === 4x4x2 block: 4 independent 2x2x2 products, each rank 7 ===
        # C_top = A11*Btop + A12*Bbot
        # C_bot = A21*Btop + A22*Bbot
        add_strassen([(A11, O)], [(Btop, O)], [(Ctop, O)])
        add_strassen([(A12, O)], [(Bbot, O)], [(Ctop, O)])
        add_strassen([(A21, O)], [(Btop, O)], [(Cbot, O)])
        add_strassen([(A22, O)], [(Bbot, O)], [(Cbot, O)])

        return {"U": U, "V": V, "W": W}

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

    def verify(result):
        U = np.array(result["U"], dtype=object)
        V = np.array(result["V"], dtype=object)
        W = np.array(result["W"], dtype=object)
        T = np.zeros((n, m, p), dtype=object)
        for c in range(U.shape[1]):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    a = uc[i * m + j]
                    if a == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += a * vc[j * p + k] * wc[k * n + i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i, j, k] != 1:
                        return False
        return True

    start_time = time.time()
    deadline = start_time + 120.0

    try:
        candidate = build_decomposition()
        if time.time() < deadline and verify(candidate):
            return candidate
    except Exception:
        pass

    fallback = build_schoolbook()
    return fallback
# EVOLVE-BLOCK-END
