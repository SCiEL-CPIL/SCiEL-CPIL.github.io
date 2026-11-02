"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140

    # Precompute index lists
    indices = []
    by_a = [[] for _ in range(nm)]
    by_b = [[] for _ in range(mp)]
    by_c = [[] for _ in range(pn)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                idx = len(indices)
                indices.append((a, b, c))
                by_a[a].append(idx)
                by_b[b].append(idx)
                by_c[c].append(idx)

    def compute_error(U, V, W):
        err = 0.0
        for (a, b, c) in indices:
            val = np.dot(U[a], V[b] * W[c])
            err += (val - 1.0)**2
        return err

    def solve_als(rank, seed=42, max_iters=500):
        np.random.seed(seed)
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5

        for it in range(max_iters):
            if time.time() > deadline - 20:
                break

            # Update U
            for a in range(nm):
                M = np.zeros(rank)
                for idx in by_a[a]:
                    _, b, c = indices[idx]
                    M += V[b] * W[c]
                norm = np.dot(M, M)
                if norm > 1e-12:
                    U[a] = M / norm

            # Update V
            for b in range(mp):
                M = np.zeros(rank)
                for idx in by_b[b]:
                    a, _, c = indices[idx]
                    M += U[a] * W[c]
                norm = np.dot(M, M)
                if norm > 1e-12:
                    V[b] = M / norm

            # Update W
            for c in range(pn):
                M = np.zeros(rank)
                for idx in by_c[c]:
                    a, b, _ = indices[idx]
                    M += U[a] * V[b]
                norm = np.dot(M, M)
                if norm > 1e-12:
                    W[c] = M / norm

            err = compute_error(U, V, W)
            if err < 1e-10:
                break

        return U, V, W, compute_error(U, V, W)

    # Try different ranks and seeds
    for rank in [40, 44, 48, 52, 56, 60, 64, 68, 72]:
        if time.time() > deadline - 40:
            break

        for seed in range(5):
            if time.time() > deadline - 20:
                break

            U, V, W, err = solve_als(rank, seed=seed)
            if err < 1e-6:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2

                err_r = compute_error(U_r, V_r, W_r)
                if err_r < 1e-6:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
