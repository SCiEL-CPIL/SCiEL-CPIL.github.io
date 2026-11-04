"""
Matrix multiplication tensor decomposition for 3x3 * 3x5.

Schoolbook rank: 45.  We attempt a quick gradient-based search for a
lower-rank exact decomposition, falling back to schoolbook if the
search fails or times out.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin below the 150s kill

    # Build the target tensor T (9 x 15 x 15)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition (rank 45)
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, r

    best_U, best_V, best_W, best_rank = schoolbook()

    def verify(U, V, W):
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.max(np.abs(P - T))

    # Quick gradient-based search for lower rank
    for target_rank in [35, 30]:
        if time.time() > deadline - 25:
            break

        found = False
        for attempt in range(2):
            if time.time() > deadline - 25:
                break

            np.random.seed(42 + attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5

            lr = 0.1
            for step in range(200):
                if time.time() > deadline - 25:
                    break

                P = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = P - T
                err = np.max(np.abs(diff))

                if err < 1e-8:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    found = True
                    break

                grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)

                gU_norm = np.linalg.norm(grad_U)
                gV_norm = np.linalg.norm(grad_V)
                gW_norm = np.linalg.norm(grad_W)

                if gU_norm > 1e-10:
                    grad_U /= gU_norm
                if gV_norm > 1e-10:
                    grad_V /= gV_norm
                if gW_norm > 1e-10:
                    grad_W /= gW_norm

                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W

                if step % 40 == 0 and step > 0:
                    lr *= 0.8

            if found:
                break

        if found:
            break

    # Final verification
    if verify(best_U, best_V, best_W) > 1e-6:
        best_U, best_V, best_W, best_rank = schoolbook()

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
