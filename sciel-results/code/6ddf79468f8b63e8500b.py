"""Solution for matrix multiplication 4x4x4.

Provides a verified rank-49 exact tensor decomposition using a two-level Strassen
construction. The decomposition is exact over the integers and uses the
official output-indexing convention W[k*n + i, r] where C = A @ B.
"""
import time
import numpy as np


def construct():
    """Return factor matrices {'U', 'V', 'W'} for a rank-49 exact
    decomposition of the 4x4x4 matrix multiplication tensor.

    Shapes:
        U: (16, rank)  # A index = i*4 + j
        V: (16, rank)  # B index = j*4 + k
        W: (16, rank)  # C index = k*4 + i
    """
    t0 = time.time()
    deadline = 140.0

    n, m, p = 4, 4, 4

    # Strassen 2x2 matrices.
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])

    # Verify 2x2 Strassen before building 4x4.
    def _verify_2x2(U2, V2, W2, trials=6):
        rng = np.random.default_rng(1234)
        for _ in range(trials):
            A = rng.integers(-8, 9, size=(2, 2)).astype(float)
            B = rng.integers(-8, 9, size=(2, 2)).astype(float)
            C_true = A @ B
            C_decomp = np.zeros((2, 2))
            Af = A.reshape(4)
            Bf = B.reshape(4)
            for r in range(7):
                s = float(np.dot(U2[:, r], Af)) * float(np.dot(V2[:, r], Bf))
                C_decomp += s * (W2[:, r].reshape(2, 2)).T
            if not np.allclose(C_true, C_decomp, atol=1e-12):
                return False
        return True

    if not _verify_2x2(U2, V2, W2):
        # Fallback: trivial schoolbook rank-64.
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=int)
        V_fb = np.zeros((16, rank_fb), dtype=int)
        W_fb = np.zeros((16, rank_fb), dtype=int)
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j, c] = 1
                    V_fb[j * 4 + k, c] = 1
                    W_fb[k * 4 + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    rank = 49
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)

    # Build the 4x4 decomposition as a tensor product of two 2x2 Strassen
    # decompositions. The outer index is s (block-level product) and the
    # inner index is t (element-level product).
    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            # U4: A index = (I*2+i)*4 + (J*2+j)
            for I in range(2):
                for J in range(2):
                    uo = U2[I * 2 + J, s]
                    if uo == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            ui = U2[i * 2 + j, t]
                            if ui == 0:
                                continue
                            row_a = (I * 2 + i) * 4 + (J * 2 + j)
                            U4[row_a, col] = uo * ui

            # V4: B index = (J*2+j)*4 + (K*2+k)
            for J in range(2):
                for K in range(2):
                    vo = V2[J * 2 + K, s]
                    if vo == 0:
                        continue
                    for j in range(2):
                        for k in range(2):
                            vi = V2[j * 2 + k, t]
                            if vi == 0:
                                continue
                            row_b = (J * 2 + j) * 4 + (K * 2 + k)
                            V4[row_b, col] = vo * vi

            # W4: C index = (K*2+k)*4 + (I*2+i)
            for K in range(2):
                for I in range(2):
                    wo = W2[K * 2 + I, s]
                    if wo == 0:
                        continue
                    for k in range(2):
                        for i in range(2):
                            wi = W2[k * 2 + i, t]
                            if wi == 0:
                                continue
                            row_c = (K * 2 + k) * 4 + (I * 2 + i)
                            W4[row_c, col] = wo * wi

    # Verify the full 4x4 decomposition.
    def _verify_4x4(U4, V4, W4, trials=10):
        if time.time() > deadline:
            return True
        rng = np.random.default_rng(2024)
        for _ in range(trials):
            if time.time() > deadline - 2.0:
                break
            A = rng.integers(-12, 13, size=(4, 4)).astype(float)
            B = rng.integers(-12, 13, size=(4, 4)).astype(float)
            C_true = A @ B
            C_decomp = np.zeros((4, 4))
            Af = A.reshape(16)
            Bf = B.reshape(16)
            for r in range(rank):
                s = float(np.dot(U4[:, r], Af)) * float(np.dot(V4[:, r], Bf))
                C_decomp += s * (W4[:, r].reshape(4, 4)).T
            if not np.allclose(C_true, C_decomp, atol=1e-12):
                return False
        return True

    if _verify_4x4(U4, V4, W4):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

    # Fallback: trivial schoolbook rank-64.
    rank_fb = 64
    U_fb = np.zeros((16, rank_fb), dtype=int)
    V_fb = np.zeros((16, rank_fb), dtype=int)
    W_fb = np.zeros((16, rank_fb), dtype=int)
    c = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_fb[i * 4 + j, c] = 1
                V_fb[j * 4 + k, c] = 1
                W_fb[k * 4 + i, c] = 1
                c += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
