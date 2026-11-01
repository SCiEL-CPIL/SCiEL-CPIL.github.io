import time
import sys
import numpy as np

DIMENSIONS = [2, 2, 11]
DEADLINE = 150.0
SAFETY = 5.0  # seconds to leave before hard kill

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS

    # --- Guaranteed rank-39 construction (Strassen blocks + schoolbook tail) ---
    def build_rank39():
        rank = 5 * 7 + 1 * 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = p + k0; v_h = p + k1
            w_c00 = k0 * n; w_c10 = k0 * n + 1
            w_c01 = k1 * n; w_c11 = k1 * n + 1
            # m1
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            # m2
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            # m3
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            # m4
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            # m5
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            # m6
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            # m7
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        # single column k2 = 10, schoolbook 4
        k2 = 10
        v_e2 = k2; v_g2 = p + k2
        w_c02 = k2 * n; w_c12 = k2 * n + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        return {"U": U, "V": V, "W": W}

    best = build_rank39()

    # --- Verification helpers ---
    def verify(U, V, W):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        # T shape: (n*m, m*p, p*n)
        T = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        
        # Construct target tensor M of shape (n*m, m*p, p*n)
        # M[a, b, c] = 1 if indices are consistent, else 0
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        M = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    M[a, b, c] = 1.0
        
        return np.allclose(T, M, atol=1e-9, rtol=0.0)

    def round_halfint(mat):
        return np.round(np.asarray(mat, dtype=float) * 2.0) / 2.0

    # --- Bounded local search: try to reduce rank via perturbation + rounding ---
    # Hypothesis: a rank-38 (or lower) exact half-integer decomposition may exist near
    # the rank-39 one if we perturb the factors and snap to the half-integer lattice.
    # We test candidate ranks in descending order, with a strict time budget.

    def try_improve(max_rank, min_rank, budget_s):
        nonlocal best
        U0 = np.asarray(best["U"], dtype=float)
        V0 = np.asarray(best["V"], dtype=float)
        W0 = np.asarray(best["W"], dtype=float)
        rng = np.random.default_rng(42)

        for rank in range(max_rank, min_rank - 1, -1):
            if time.time() - t_start > DEADLINE - SAFETY:
                break
            sub_budget = min(20.0, max(2.0, (DEADLINE - SAFETY) - (time.time() - t_start)))
            t_rank_start = time.time()
            # subsample columns from the rank-39 solution to seed rank-r candidates
            n_cols = U0.shape[1]
            for attempt in range(8):
                if time.time() - t_rank_start > sub_budget:
                    break
                if time.time() - t_start > DEADLINE - SAFETY:
                    break
                # pick 'rank' columns (with replacement to allow reuse)
                idx = rng.integers(0, n_cols, size=rank)
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
                # small perturbation to escape exact lattice, then round
                U = U + rng.normal(0, 0.1, U.shape)
                V = V + rng.normal(0, 0.1, V.shape)
                W = W + rng.normal(0, 0.1, W.shape)
                U = round_halfint(U)
                V = round_halfint(V)
                W = round_halfint(W)
                # quick verify
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return True
        return False

    # Try to improve from rank 38 down to 34 (bounded)
    if time.time() - t_start < DEADLINE - SAFETY - 5:
        try_improve(38, 34, 10.0)

    return best
