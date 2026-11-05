"""Matrix multiplication 4x4 x 4x6 tensor decomposition.

Strategy:
  1. Build a rank-77 decomposition by top-level Strassen (7 terms) where each
     term is a (2x2) x (2x3) product.  For each such product, use inner
     Strassen (rank 7) for the 2x2 x 2x2 part and schoolbook (rank 4) for the
     2x2 x 2x1 part, giving 11 terms per top-level term.  Total: 7 * 11 = 77.
  2. Attempt to reduce rank further via a smarter coordinate descent that uses
     the exact linear solve for the best update of each column given the others
     (this is the key refinement: instead of random perturbations, we compute
     the exact residual that column c must explain, then project onto the
     constraint that the column entries should be small/structured).
  3. Fallback to the rank-77 construction if no improvement is found.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},
    {0: 0, 1: 0, 2: 1, 3: 1},
    {0: 1, 1: 0, 2: 0, 3: 0},
    {0: 0, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 1, 2: 0, 3: 0},
    {0: -1, 1: 0, 2: 1, 3: 0},
    {0: 0, 1: 1, 2: 0, 3: -1},
]
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 0, 2: 0, 3: 0},
    {0: 0, 1: 1, 2: 0, 3: -1},
    {0: -1, 1: 0, 2: 1, 3: 0},
    {0: 0, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 1, 2: 0, 3: 0},
    {0: 0, 1: 0, 2: 1, 3: 1},
]
recon = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 0, -1), (0, 1, 1)],
    [(1, 1, 1)],
    [(0, 0, 1)],
]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118

    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def err_of(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - recon) ** 2)

    # ---- Build rank-77 decomposition ----
    U77, V77, W77 = _build_rank77(n, m, p)

    if not check(U77, V77, W77, atol=1e-9):
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=np.float64)
        V_sb = np.zeros((m * p, rank_sb), dtype=np.float64)
        W_sb = np.zeros((p * n, rank_sb), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}

    best_rank = 77
    best_U, best_V, best_W = U77.copy(), V77.copy(), W77.copy()

    # ---- Rank reduction via coordinate descent with exact column solve ----
    for target_rank in range(76, 55, -1):
        if time.time() > deadline - 25:
            break

        U = best_U[:, :target_rank].copy()
        V = best_V[:, :target_rank].copy()
        W = best_W[:, :target_rank].copy()
        err = err_of(U, V, W)

        for restart in range(4):
            if time.time() > deadline - 18:
                break
            if restart > 0:
                U += np.random.randn(n * m, target_rank) * 0.05
                V += np.random.randn(m * p, target_rank) * 0.05
                W += np.random.randn(p * n, target_rank) * 0.05
                err = err_of(U, V, W)

            # Coordinate descent: for each column, try to improve
            for it in range(30):
                if time.time() > deadline - 12:
                    break
                if err < 1e-10:
                    break
                improved = False
                for c in range(target_rank):
                    if time.time() > deadline - 10:
                        break
                    Uo, Vo, Wo = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
                    # Compute residual without column c
                    recon_others = np.einsum('ar,br,cr->abc', U[:, :c], V[:, :c], W[:, :c]) + \
                                  np.einsum('ar,br,cr->abc', U[:, c+1:], V[:, c+1:], W[:, c+1:])
                    residual = T - recon_others
                    # We need u, v, w such that u_a v_b w_c ≈ residual_{abc}
                    # This is a rank-1 approximation problem.
                    # Use SVD on a flattened version, or try a few structured updates.
                    
                    # Try: fix V and W, solve for U (least squares)
                    # residual_{abc} ≈ U_a * (V_b W_c)_b -> for each a, sum_b U_a V_b W_c = residual_{abc}
                    # This is underdetermined; instead use alternating minimization
                    for _ in range(5):
                        # Update U: for each a, U_a = argmin sum_bc (residual_{abc} - U_a V_b W_c)^2
                        # = sum_bc residual_{abc} V_b W_c / sum_bc V_b^2 W_c^2
                        num = np.einsum('abc,b,c->a', residual, V[:, c], W[:, c])
                        den = np.sum(V[:, c]**2) * np.sum(W[:, c]**2)
                        if den > 1e-12:
                            U[:, c] = num / den
                        # Update V
                        num = np.einsum('abc,a,c->b', residual, U[:, c], W[:, c])
                        den = np.sum(U[:, c]**2) * np.sum(W[:, c]**2)
                        if den > 1e-12:
                            V[:, c] = num / den
                        # Update W
                        num = np.einsum('abc,a,b->c', residual, U[:, c], V[:, c])
                        den = np.sum(U[:, c]**2) * np.sum(V[:, c]**2)
                        if den > 1e-12:
                            W[:, c] = num / den
                    
                    new_err = err_of(U, V, W)
                    if new_err < err:
                        err = new_err
                        improved = True
                    else:
                        U[:, c], V[:, c], W[:, c] = Uo, Vo, Wo
                if not improved:
                    break

        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break

    # Final polish
    if time.time() < deadline - 5 and best_rank < 96:
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        err = err_of(U, V, W)
        for it in range(30):
            if time.time() > deadline - 3:
                break
            if err < 1e-12:
                break
            improved = False
            for c in range(best_rank):
                Uo, Vo, Wo = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
                recon_others = np.einsum('ar,br,cr->abc', U[:, :c], V[:, :c], W[:, :c]) + \
                              np.einsum('ar,br,cr->abc', U[:, c+1:], V[:, c+1:], W[:, c+1:])
                residual = T - recon_others
                for _ in range(10):
                    num = np.einsum('abc,b,c->a', residual, V[:, c], W[:, c])
                    den = np.sum(V[:, c]**2) * np.sum(W[:, c]**2)
                    if den > 1e-12:
                        U[:, c] = num / den
                    num = np.einsum('abc,a,c->b', residual, U[:, c], W[:, c])
                    den = np.sum(U[:, c]**2) * np.sum(W[:, c]**2)
                    if den > 1e-12:
                        V[:, c] = num / den
                    num = np.einsum('abc,a,b->c', residual, U[:, c], V[:, c])
                    den = np.sum(U[:, c]**2) * np.sum(V[:, c]**2)
                    if den > 1e-12:
                        W[:, c] = num / den
                new_err = err_of(U, V, W)
                if new_err < err:
                    err = new_err
                    improved = True
                else:
                    U[:, c], V[:, c], W[:, c] = Uo, Vo, Wo
            if not improved:
                break
        if err < 1e-10:
            best_U, best_V, best_W = U, V, W

    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}


def _build_rank77(n, m, p):
    R = 77
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)

    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                coeff_a = alpha[si].get(a, 0)
                if coeff_a != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        coeff_r = alpha[s].get(r, 0)
                        if coeff_r != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * m + (br * 2 + x_col)
                            U[u_idx, col] += coeff_a * coeff_r
            for b in range(4):
                coeff_b = beta[si].get(b, 0)
                if coeff_b != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        coeff_c = beta[s].get(c, 0)
                        if coeff_c != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * p + (bc * 3 + y_inner_col)
                            V[v_idx, col] += coeff_b * coeff_c
            for a_idx, b_idx, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b_idx * 3 + zb) * n + (a_idx * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1

        for u in range(2):
            for v in range(2):
                for r in range(4):
                    coeff_r = alpha[s].get(r, 0)
                    if coeff_r != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * m + (br * 2 + v)
                        U[u_idx, col] += coeff_r
                for c in range(4):
                    coeff_c = beta[s].get(c, 0)
                    if coeff_c != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * p + (bc * 3 + 2)
                        V[v_idx, col] += coeff_c
                for a_idx, b_idx, gamma in recon[s]:
                    w_idx = (b_idx * 3 + 2) * n + (a_idx * 2 + u)
                    W[w_idx, col] += gamma
                col += 1

    return U, V, W


def _to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    print(f"Rank: {U.shape[1]}")
    n, m, p = 4, 4, 6
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T - recon))
    print(f"Max error: {max_err}")
    print(f"Valid: {np.allclose(recon, T, atol=1e-9)}")
