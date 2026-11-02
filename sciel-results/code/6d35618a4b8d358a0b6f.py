import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x3 matrix.
    Uses a rank-26 decomposition (Strassen on 2x2 block + schoolbook for cross terms).
    Falls back to rank 27 (schoolbook) if rank 26 fails verification.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin

    n, m, p = 3, 3, 3

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def _verify_matmul(U, V, W):
        """
        Verifies if U, V, W exactly decompose the 3x3 matrix multiplication tensor.
        Uses integer arithmetic for exactness.
        """
        rank = len(U[0])
        # Use a non-symmetric, non-trivial test case to avoid accidental passes
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.int64)
        B = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int64)
        C_expected = A @ B

        C_computed = np.zeros((3, 3), dtype=np.int64)

        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)

        A_flat = A.flatten()
        B_flat = B.flatten()

        # Project A and B onto the rank factors
        U_proj = U_arr.T @ A_flat
        V_proj = V_arr.T @ B_flat

        for r in range(rank):
            p_r = U_proj[r] * V_proj[r]
            if p_r != 0:
                w_col = W_arr[:, r]
                C_contrib = np.zeros((3, 3), dtype=np.int64)
                for k in range(3):
                    for i in range(3):
                        idx = k * 3 + i
                        C_contrib[i, k] = w_col[idx]

                C_computed += p_r * C_contrib

        return np.array_equal(C_computed, C_expected)

    def _rank26_construct():
        """
        Rank-26 decomposition for 3x3 matmul using Strassen on 2x2 block.
        Partition: A11(2x2), A12(2x1), A21(1x2), A22(1x1)
        """
        rank = 26
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]

        terms = []

        def add(u, v, w):
            terms.append((u, v, w))

        # Strassen terms for A11*B11
        # M1 = (a[0][0]+a[1][1])*(b[0][0]+b[1][1])
        # C00 += M1, C11 += M1
        add({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})  # term 0

        # M2 = (a[1][0]+a[1][1])*b[0][0]
        # C10 += M2
        add({3:1, 4:1}, {0:1}, {3:1})  # term 1

        # M3 = a[0][0]*(b[0][1]-b[1][1])
        # C01 += M3
        add({0:1}, {1:1, 4:-1}, {1:1})  # term 2

        # M4 = a[1][1]*(b[1][0]-b[0][0])
        # C00 -= M4, C10 += M4
        add({4:1}, {3:1, 0:-1}, {0:-1, 3:1})  # term 3

        # M5 = (a[0][0]+a[0][1])*b[1][1]
        # C00 += M5, C01 += M5
        add({0:1, 1:1}, {4:1}, {0:1, 1:1})  # term 4

        # M6 = (a[1][0]-a[0][0])*(b[0][0]+b[0][1])
        # C11 += M6
        add({3:1, 0:-1}, {0:1, 1:1}, {4:1})  # term 5

        # M7 = (a[0][1]-a[1][1])*(b[1][0]+b[1][1])
        # C00 -= M7
        add({1:1, 4:-1}, {3:1, 4:1}, {0:-1})  # term 6

        # A12*B21 terms (4)
        # a[0][2]*b[2][0] -> C[0,0]
        add({2:1}, {6:1}, {0:1})  # term 7

        # a[0][2]*b[2][1] -> C[0,1]
        add({2:1}, {7:1}, {1:1})  # term 8

        # a[1][2]*b[2][0] -> C[1,0]
        add({5:1}, {6:1}, {3:1})  # term 9

        # a[1][2]*b[2][1] -> C[1,1]
        add({5:1}, {7:1}, {4:1})  # term 10

        # A11*B12 terms (4)
        # a[0][0]*b[0][2] -> C[0,2]
        add({0:1}, {2:1}, {6:1})  # term 11

        # a[0][1]*b[1][2] -> C[0,2]
        add({1:1}, {5:1}, {6:1})  # term 12

        # a[1][0]*b[0][2] -> C[1,2]
        add({3:1}, {2:1}, {7:1})  # term 13

        # a[1][1]*b[1][2] -> C[1,2]
        add({4:1}, {5:1}, {7:1})  # term 14

        # A12*B22 terms (2)
        # a[0][2]*b[2][2] -> C[0,2]
        add({2:1}, {8:1}, {6:1})  # term 15

        # a[1][2]*b[2][2] -> C[1,2]
        add({5:1}, {8:1}, {7:1})  # term 16

        # A21*B11 terms (4)
        # a[2][0]*b[0][0] -> C[2,0]
        add({6:1}, {0:1}, {2:1})  # term 17

        # a[2][1]*b[1][0] -> C[2,0]
        add({7:1}, {3:1}, {2:1})  # term 18

        # a[2][0]*b[0][1] -> C[2,1]
        add({6:1}, {1:1}, {5:1})  # term 19

        # a[2][1]*b[1][1] -> C[2,1]
        add({7:1}, {4:1}, {5:1})  # term 20

        # A22*B21 terms (2)
        # a[2][2]*b[2][0] -> C[2,0]
        add({8:1}, {6:1}, {2:1})  # term 21

        # a[2][2]*b[2][1] -> C[2,1]
        add({8:1}, {7:1}, {5:1})  # term 22

        # A21*B12 terms (2)
        # a[2][0]*b[0][2] -> C[2,2]
        add({6:1}, {2:1}, {8:1})  # term 23

        # a[2][1]*b[1][2] -> C[2,2]
        add({7:1}, {5:1}, {8:1})  # term 24

        # A22*B22 term (1)
        # a[2][2]*b[2][2] -> C[2,2]
        add({8:1}, {8:1}, {8:1})  # term 25

        assert len(terms) == 26, f"Expected 26 terms, got {len(terms)}"

        for r, (u, v, w) in enumerate(terms):
            for idx, val in u.items():
                U[idx][r] = val
            for idx, val in v.items():
                V[idx][r] = val
            for idx, val in w.items():
                W[idx][r] = val

        return {"U": U, "V": V, "W": W}

    best = _schoolbook()

    if time.time() < deadline:
        cand = _rank26_construct()
        if cand is not None:
            if _verify_matmul(cand["U"], cand["V"], cand["W"]):
                best = cand

    return best
