import time
import numpy as np
import random

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS

    # Build rank-32 decomposition: 4 Strassen blocks (rank 7 each) + 1 trivial (rank 4)
    rank = 32

    U = np.zeros((n * m, rank))  # 4 x 32
    V = np.zeros((m * p, rank))  # 18 x 32
    W = np.zeros((p * n, rank))  # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M2
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M4
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        # M5
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        # M6
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1; col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1; col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1; col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Build tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))

    tensor_a = np.array([e[0] for e in tensor_entries])
    tensor_b = np.array([e[1] for e in tensor_entries])
    tensor_c = np.array([e[2] for e in tensor_entries])
    tensor_val = np.ones(len(tensor_entries))
    N = len(tensor_entries)

    target_rank = 31

    def compute_residual(Uc, Vc, Wc):
        prod = Uc[tensor_a] * Vc[tensor_b] * Wc[tensor_c]
        vals = prod.sum(axis=1)
        return np.sum((vals - tensor_val) ** 2)

    # Try dropping each column and check residual
    best_residual = float('inf')
    best_init = None

    for drop_col in range(rank):
        if time.time() > deadline:
            break
        mask = np.ones(rank, dtype=bool)
        mask[drop_col] = False
        Uc = U[:, mask]
        Vc = V[:, mask]
        Wc = W[:, mask]
        res = compute_residual(Uc, Vc, Wc)
        if res < best_residual:
            best_residual = res
            best_init = (Uc.copy(), Vc.copy(), Wc.copy())

    if best_init is not None and best_residual < 1e-10:
        return {"U": best_init[0].tolist(), "V": best_init[1].tolist(), "W": best_init[2].tolist()}

    # Local search with gradient-based optimization
    if best_init is not None:
        Uc = best_init[0].copy()
        Vc = best_init[1].copy()
        Wc = best_init[2].copy()
        cur_res = best_residual

        lr = 0.05
        for iteration in range(2000):
            if time.time() > deadline:
                break

            # Stochastic gradient step
            idx = random.randint(0, N - 1)
            a, b, c = tensor_entries[idx]

            pred = np.sum(Uc[a] * Vc[b] * Wc[c])
            err = pred - 1.0

            # Update with gradient
            grad_U = err * Vc[b] * Wc[c]
            grad_V = err * Uc[a] * Wc[c]
            grad_W = err * Uc[a] * Vc[b]

            Uc[a] -= lr * grad_U
            Vc[b] -= lr * grad_V
            Wc[c] -= lr * grad_W

            # Check residual periodically
            if iteration % 20 == 0:
                new_res = compute_residual(Uc, Vc, Wc)
                if new_res < cur_res:
                    cur_res = new_res
                    if cur_res < 1e-10:
                        return {"U": Uc.tolist(), "V": Vc.tolist(), "W": Wc.tolist()}
                else:
                    # Revert if it got worse
                    Uc = best_init[0].copy()
                    Vc = best_init[1].copy()
                    Wc = best_init[2].copy()
                    cur_res = best_residual
                    lr *= 0.5
                    if lr < 1e-4:
                        break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
