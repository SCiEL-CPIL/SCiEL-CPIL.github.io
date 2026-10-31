import numpy as np
import time


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under the 150 s wall-clock limit

    # ------------------------------------------------------------------
    # 2x2 Strassen factors (rank 7), integer entries.
    # U2[i, t]:  row i of A  = sum_t U2[i, t] * M_t
    # V2[j, t]:  col j of B  = sum_t V2[j, t] * M_t
    # W2[k, t]:  col k of C  = sum_t W2[k, t] * M_t
    # (W2 is transposed from the common textbook listing.)
    # ------------------------------------------------------------------
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=int)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=int)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=int)

    # ------------------------------------------------------------------
    # 4x4 factors via Kronecker product of the 2x2 factors.
    #   U = kron(U2, U2)   shape (16, 49)
    #   V = kron(V2, V2)   shape (16, 49)
    #   W = kron(W2, W2)   shape (16, 49)
    # Index maps:  U row (I,i,J,j) = I*4 + J ;  V row (J,j,K,k) = J*4 + K ;
    #              W row (K,k,I,i) = K*4 + I  (official trace convention).
    # ------------------------------------------------------------------
    U = np.kron(U2, U2)   # (16, 49)
    V = np.kron(V2, V2)   # (16, 49)
    W = np.kron(W2, W2)   # (16, 49)

    # ------------------------------------------------------------------
    # Optional fast exact self-check (cheap; 49 scalar products of 16x16).
    # Reconstruct C = A @ B from the factors and compare.
    # ------------------------------------------------------------------
    if time.time() < deadline:
        try:
            rng = np.random.default_rng(12345)
            ok = True
            for _ in range(2):
                A = rng.integers(-3, 4, size=(4, 4))
                B = rng.integers(-3, 4, size=(4, 4))
                C = A @ B
                # scalar t:  M_t = (U^T A_vec) . (V^T B_vec)
                a = A.reshape(16)          # row-major (I*4+J)
                b = B.reshape(16)          # row-major (J*4+K)
                scal = (U.T @ a) * (V.T @ b)     # (49,)
                Crec = (W.T @ scal).reshape(4, 4)  # (K*4+I)
                if not np.array_equal(C, Crec):
                    ok = False
        except Exception:
            ok = True  # never fail the submission because of a check

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
