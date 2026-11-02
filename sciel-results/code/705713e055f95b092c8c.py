import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = DIMENSIONS
    U_dim = n * m  # 15
    V_dim = m * p  # 35
    W_dim = p * n  # 21

    # Build target tensor and precompute non-zero patterns
    T = np.zeros((U_dim, V_dim, W_dim))
    T_a = {}
    T_b = {}
    T_c = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                T_a.setdefault(a, []).append((b, c))
                T_b.setdefault(b, []).append((a, c))
                T_c.setdefault(c, []).append((a, b))

    # Schoolbook fallback
    rank_school = n * m * p
    U_s = np.zeros((U_dim, rank_school))
    V_s = np.zeros((V_dim, rank_school))
    W_s = np.zeros((W_dim, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school

    # Try decreasing ranks
    for rank in range(35, rank_school):
        if time.time() > deadline:
            break
        if rank >= best_rank:
            break

        found = False
        for seed in range(40):
            if time.time() > deadline:
                break

            np.random.seed(seed * 7919 + rank * 104729)
            U = np.random.randn(U_dim, rank)
            V = np.random.randn(V_dim, rank)
            W = np.random.randn(W_dim, rank)

            # Alternating least squares
            for it in range(150):
                if time.time() > deadline:
                    break

                # Update U
                for a in range(U_dim):
                    pairs = T_a.get(a, [])
                    if not pairs:
                        continue
                    A = np.array([V[b, :] * W[c, :] for b, c in pairs])
                    b_vec = np.ones(len(pairs))
                    U[a, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

                # Update V
                for b in range(V_dim):
                    pairs = T_b.get(b, [])
                    if not pairs:
                        continue
                    A = np.array([U[a, :] * W[c, :] for a, c in pairs])
                    b_vec = np.ones(len(pairs))
                    V[b, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

                # Update W
                for c in range(W_dim):
                    pairs = T_c.get(c, [])
                    if not pairs:
                        continue
                    A = np.array([U[a, :] * V[b, :] for a, b in pairs])
                    b_vec = np.ones(len(pairs))
                    W[c, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

            # Check exactness
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.max(np.abs(T_recon - T))
            if error < 1e-8:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                found = True
                break

        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
