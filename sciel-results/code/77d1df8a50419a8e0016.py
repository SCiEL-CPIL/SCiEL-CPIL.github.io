"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

_T_START = time.time()
_DEADLINE = _T_START + 140.0  # safety margin vs 150s kill

def _elapsed():
    return time.time() - _T_START

def _time_left():
    return max(0.0, _DEADLINE - _elapsed())

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5

    def check_decomposition(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    expected = 1.0
                    actual = sum(U_arr[a*m+b, r] * V_arr[b*p+c, r] * W_arr[c*n+a, r]
                                  for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def build_rank17():
        """
        Rank-17 candidate: 8 terms for cols 0-2 (generalized Strassen k=3)
        + 7 terms for cols 3-4 (standard Strassen k=2).
        This is a *starting point*; it is generally NOT exact and must be refined.
        """
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_k3(c, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+2][r] = 1
            W[c*n+0][r] = 1; W[(c+2)*n+1][r] = 1

            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+2)*n+1][r] = -1

            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1

            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[(c+1)*n+0][r] = 1

            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1

            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+2)*n+1][r] = 1

            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1

            r = base + 7
            U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[(c+2)*n+0][r] = 1

        def add_strassen_k2(c, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1; W[(c+1)*n+1][r] = 1

            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+1)*n+1][r] = -1

            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1

            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[c*n+1][r] = 1

            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+1][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1

            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+1)*n+1][r] = 1

            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1

        add_strassen_k3(0, 0)
        add_strassen_k2(3, 8)
        return U, V, W

    def build_rank18():
        """Verified rank-18 fallback: two Strassen-2 blocks + schoolbook col 4."""
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(col0, col1, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1; W[col1*n+1][r] = 1

            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1
            W[col0*n+1][r] = 1; W[col1*n+1][r] = -1

            r = base + 2
            U[0][r] = 1
            V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
            W[col1*n+0][r] = 1; W[col1*n+1][r] = 1

            r = base + 3
            U[3][r] = 1
            V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
            W[col0*n+0][r] = 1; W[col0*n+1][r] = 1

            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+col1][r] = 1
            W[col0*n+0][r] = -1; W[col1*n+0][r] = 1

            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
            W[col1*n+1][r] = 1

            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1

        add_strassen_block(0, 1, 0)
        add_strassen_block(2, 3, 7)

        r = 14
        U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
        r = 15
        U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
        r = 16
        U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
        r = 17
        U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
        return U, V, W

    def _compute_error(U, V, W, rank):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        e = 0.0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    s = float(sum(U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r]
                                  for r in range(rank)))
                    e += (s - 1.0) ** 2
        return e

    def _snap_to_half(U, V, W, rank):
        out = (np.zeros_like(U), np.zeros_like(V), np.zeros_like(W))
        for (src, dst) in ((U, out[0]), (V, out[1]), (W, out[2])):
            for i in range(src.shape[0]):
                for r in range(rank):
                    v = float(src[i, r])
                    if abs(v) < 1e-6:
                        dst[i, r] = 0.0
                    else:
                        dst[i, r] = round(2.0 * v) / 2.0
        return out

    def _snap_check(U, V, W, rank):
        sU, sV, sW = _snap_to_half(U, V, W, rank)
        return check_decomposition(sU, sV, sW, rank), (sU.tolist(), sV.tolist(), sW.tolist())

    def _als_rank(U_init, V_init, W_init, rank, max_iters, tol=1e-11):
        """Alternating least-squares refinement of a rank-r factorization."""
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        best_err = _compute_error(U, V, W, rank)
        for it in range(max_iters):
            if _time_left() < 2.0:
                break
            for a in range(n):
                for b in range(m):
                    A_mat = np.zeros((p, rank))
                    for c in range(p):
                        A_mat[c, :] = V[b*p+c, :] * W[c*n+a, :]
                    b_vec = np.ones(p)
                    try:
                        U[a*m+b, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except Exception:
                        pass
            for b in range(m):
                for c in range(p):
                    A_mat = np.zeros((n, rank))
                    for a in range(n):
                        A_mat[a, :] = U[a*m+b, :] * W[c*n+a, :]
                    b_vec = np.ones(n)
                    try:
                        V[b*p+c, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except Exception:
                        pass
            for c in range(p):
                for a in range(n):
                    A_mat = np.zeros((m, rank))
                    for b in range(m):
                        A_mat[b, :] = U[a*m+b, :] * V[b*p+c, :]
                    b_vec = np.ones(m)
                    try:
                        W[c*n+a, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except Exception:
                        pass
            err = _compute_error(U, V, W, rank)
            if err < best_err:
                best_err = err
            if err < tol:
                break
        if best_err > 1e-7:
            return None
        ok, snapped = _snap_check(U, V, W, rank)
        if ok:
            return snapped
        if check_decomposition(U, V, W, rank):
            return (U.tolist(), V.tolist(), W.tolist())
        return None

    def _random_init(rank, seed):
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n * m, rank))
        V = rng.standard_normal((m * p, rank))
        W = rng.standard_normal((p * n, rank))
        # Light schoolbook anchoring so random starts are closer to a solution
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    r = (a * m + b) * p + c
                    if r < rank:
                        U[a*m+b, r] += 1.0
                        V[b*p+c, r] += 1.0
                        W[c*n+a, r] += 1.0
        return U, V, W

    # 1) Fast path: rank-18 fallback is always valid and cheap.
    U18, V18, W18 = build_rank18()
    if not check_decomposition(U18, V18, W18, 18):
        # Should never happen; return schoolbook rank-20 as last resort.
        U, V, W = (np.zeros((4, 20)), np.zeros((10, 20)), np.zeros((10, 20)))
        k = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    U[a*m+b, k] = 1.0
                    V[b*p+c, k] = 1.0
                    W[c*n+a, k] = 1.0
                    k += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best = {"U": U18, "V": V18, "W": W18}
    best_rank = 18

    # 2) Rank-16 random-restart ALS (new: below parent's floor of 17).
    if _time_left() > 35.0:
        for seed in range(6):
            if _time_left() < 8.0:
                break
            U0, V0, W0 = _random_init(16, 1000 + seed)
            res = _als_rank(U0, V0, W0, 16, max_iters=80)
            if res is not None:
                best = {"U": res[0], "V": res[1], "W": res[2]}
                best_rank = 16
                break

    # 3) Rank-17 multi-start (new: 8 perturbed seeds vs parent's single start).
    if best_rank == 18 and _time_left() > 30.0:
        U17, V17, W17 = build_rank17()
        # Seed 0: the pure structural start
        res = _als_rank(U17, V17, W17, 17, max_iters=120)
        if res is not None:
            best = {"U": res[0], "V": res[1], "W": res[2]}
            best_rank = 17
        # Perturbed starts: blend structural with small random noise
        if best_rank == 18:
            for s in range(1, 9):
                if _time_left() < 8.0:
                    break
                rng = np.random.default_rng(7000 + s)
                U0 = np.array(U17, dtype=float) + 0.15 * rng.standard_normal((n*m, 17))
                V0 = np.array(V17, dtype=float) + 0.15 * rng.standard_normal((m*p, 17))
                W0 = np.array(W17, dtype=float) + 0.15 * rng.standard_normal((p*n, 17))
                res = _als_rank(U0, V0, W0, 17, max_iters=100)
                if res is not None:
                    best = {"U": res[0], "V": res[1], "W": res[2]}
                    best_rank = 17
                    break

    return best
# EVOLVE-BLOCK-END
