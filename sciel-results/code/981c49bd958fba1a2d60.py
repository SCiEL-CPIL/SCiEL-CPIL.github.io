"""
Rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication.
Uses ALS search with improved initialization and convergence.
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 30, 20

    # Build target tensor T[i, j, k]
    # i = i1*m + j1 (row index in A)
    # j = j1*p + k2 (col index in B)
    # k = k2*n + i1 (output index)
    T = np.zeros((nm, mp, pn))
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                T[i1 * m + j1, j1 * p + k2, k2 * n + i1] = 1.0

    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 60
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                U_school[i1 * m + j1, col] = 1.0
                V_school[j1 * p + k2, col] = 1.0
                W_school[k2 * n + i1, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()

    deadline = time.time() + 130  # Safety margin before 150s limit

    # Try ranks from 59 down to min_dim
    min_dim = max(nm, mp, pn)  # 30
    for r in range(rank_school - 1, min_dim - 1, -1):
        if time.time() > deadline:
            break

        found = False
        # Try multiple random seeds and initialization strategies
        for seed in range(30):
            if time.time() > deadline:
                break
            np.random.seed(seed)

            # Different initialization strategies
            if seed < 6:
                # Start from schoolbook truncated
                U_c = U_school[:, :r].copy()
                V_c = V_school[:, :r].copy()
                W_c = W_school[:, :r].copy()
            elif seed < 12:
                # Random Gaussian initialization
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
            elif seed < 18:
                # Sparse random initialization
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                W_c = np.random.randn(pn, r)
                # Zero out some entries to encourage sparsity
                for mat in [U_c, V_c, W_c]:
                    mask = np.random.rand(*mat.shape) < 0.4
                    mat[mask] = 0
            elif seed < 24:
                # Structured initialization: assign terms in a pattern
                U_c = np.zeros((nm, r))
                V_c = np.zeros((mp, r))
                W_c = np.zeros((pn, r))
                for col in range(r):
                    i1 = col % n
                    j1 = (col // n) % m
                    k2 = (col // (n * m)) % p
                    U_c[i1 * m + j1, col] = 1.0
                    V_c[j1 * p + k2, col] = 1.0
                    W_c[k2 * n + i1, col] = 1.0
                # Add small perturbation
                U_c += 0.05 * np.random.randn(nm, r)
                V_c += 0.05 * np.random.randn(mp, r)
                W_c += 0.05 * np.random.randn(pn, r)
            else:
                # Hybrid: half schoolbook, half random
                half = r // 2
                U_c = np.zeros((nm, r))
                V_c = np.zeros((mp, r))
                W_c = np.zeros((pn, r))
                # First half from schoolbook
                U_c[:, :half] = U_school[:, :half]
                V_c[:, :half] = V_school[:, :half]
                W_c[:, :half] = W_school[:, :half]
                # Second half random
                U_c[:, half:] = np.random.randn(nm, r - half)
                V_c[:, half:] = np.random.randn(mp, r - half)
                W_c[:, half:] = np.random.randn(pn, r - half)

            # Alternating Least Squares
            for it in range(25):
                if time.time() > deadline:
                    break

                # Solve for U: T[i, :, :] = U[i, :] @ (V * W)
                M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                U_c = T.reshape(nm, mp * pn) @ np.linalg.pinv(M.T)

                # Solve for V
                N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                V_c = T.reshape(mp, nm * pn) @ np.linalg.pinv(N.T)

                # Solve for W
                P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                W_c = T.reshape(pn, nm * mp) @ np.linalg.pinv(P.T)

            # Check convergence
            T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
            error = np.max(np.abs(T_check - T))

            if error < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U_c * 2) / 2
                V_r = np.round(V_c * 2) / 2
                W_r = np.round(W_c * 2) / 2

                T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.max(np.abs(T_check2 - T)) < 1e-10:
                    best_rank = r
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break

        if found:
            break

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
