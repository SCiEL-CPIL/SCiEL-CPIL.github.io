import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _try_rank17(n, m, p, seed, max_iters=120):
    rng = random.Random(seed)
    R = 17

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # ---- structured seed: Strassen-7 (2x2) + rank-2 (2x1) + small random ----
    def _strassen7():
        s1 = [[0.0] * 2 for _ in range(2)]
        s2 = [[0.0] * 2 for _ in range(2)]
        s3 = [[0.0] * 2 for _ in range(2)]
        s4 = [[0.0] * 2 for _ in range(2)]
        s5 = [[0.0] * 2 for _ in range(2)]
        s6 = [[0.0] * 2 for _ in range(2)]
        s7 = [[0.0] * 2 for _ in range(2)]
        s1[0][0] = 1.0
        s2[0][0] = 1.0; s2[1][0] = 1.0
        s3[0][1] = 1.0
        s4[1][1] = 1.0
        s5[0][0] = 1.0; s5[1][1] = 1.0
        s6[0][1] = 1.0; s6[1][1] = 1.0
        s7[0][0] = 1.0; s7[1][0] = 1.0
        return [s1, s2, s3, s4, s5, s6, s7]

    def _rank2_2x1():
        return [[[1.0], [0.0]], [[0.0], [1.0]]]

    S7 = _strassen7()
    R2 = _rank2_2x1()
    u = []
    v = []
    w = []
    for t in range(7):
        uu = [0.0] * (n * m)
        uu[0 * m + 0] = S7[t][0][0]
        uu[0 * m + 1] = S7[t][0][1]
        uu[1 * m + 0] = S7[t][1][0]
        uu[1 * m + 1] = S7[t][1][1]
        vv = [0.0] * (m * p)
        vv[0 * p + 0] = S7[t][0][0]
        vv[0 * p + 1] = S7[t][0][1]
        vv[1 * p + 0] = S7[t][1][0]
        vv[1 * p + 1] = S7[t][1][1]
        ww = [0.0] * (p * n)
        ww[0 * n + 0] = S7[t][0][0]
        ww[0 * n + 1] = S7[t][0][1]
        ww[1 * n + 0] = S7[t][1][0]
        ww[1 * n + 1] = S7[t][1][1]
        u.append(uu); v.append(vv); w.append(ww)
    for t in range(2):
        uu = [0.0] * (n * m)
        uu[0 * m + 2] = R2[t][0][0]
        uu[1 * m + 2] = R2[t][1][0]
        vv = [0.0] * (m * p)
        vv[2 * p + 0] = R2[t][0][0]
        vv[2 * p + 1] = R2[t][0][1]
        vv[2 * p + 2] = R2[t][0][1]
        ww = [0.0] * (p * n)
        ww[0 * n + 0] = R2[t][0][0]
        ww[1 * n + 0] = R2[t][1][0]
        ww[2 * n + 0] = R2[t][1][0]
        u.append(uu); v.append(vv); w.append(ww)
    for t in range(8):
        u.append([rng.uniform(-1, 1) for _ in range(n * m)])
        v.append([rng.uniform(-1, 1) for _ in range(m * p)])
        w.append([rng.uniform(-1, 1) for _ in range(p * n)])

    def compute_residual(u, v, w):
        res = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += u[r][u_idx(i, j)] * v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    res[(i, j, k)] = s - 1.0
        return res

    def solve_u_given_vw(v, w, free_r):
        A = np.zeros((n * m * p, n * m))
        rhs = np.ones(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        if r == free_r:
                            continue
                        s += u[r][u_idx(i, j)] * v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    rhs[eq] = 1.0 - s
                    for c in range(n * m):
                        A[eq, c] = v[free_r][v_idx(j, k)] * w[free_r][w_idx(k, i)] * (1.0 if c == u_idx(i, j) else 0.0)
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        u_new = [row[:] for row in u]
        u_new[free_r] = sol
        return u_new

    def solve_v_given_uw(u, w, free_r):
        A = np.zeros((n * m * p, m * p))
        rhs = np.ones(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        if r == free_r:
                            continue
                        s += u[r][u_idx(i, j)] * v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    rhs[eq] = 1.0 - s
                    for c in range(m * p):
                        A[eq, c] = u[free_r][u_idx(i, j)] * w[free_r][w_idx(k, i)] * (1.0 if c == v_idx(j, k) else 0.0)
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        v_new = [row[:] for row in v]
        v_new[free_r] = sol
        return v_new

    def solve_w_given_uv(u, v, free_r):
        A = np.zeros((n * m * p, p * n))
        rhs = np.ones(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        if r == free_r:
                            continue
                        s += u[r][u_idx(i, j)] * v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    rhs[eq] = 1.0 - s
                    for c in range(p * n):
                        A[eq, c] = u[free_r][u_idx(i, j)] * v[free_r][v_idx(j, k)] * (1.0 if c == w_idx(k, i) else 0.0)
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        w_new = [row[:] for row in w]
        w_new[free_r] = sol
        return w_new

    def residual_norm(u, v, w):
        res = compute_residual(u, v, w)
        return sum(x * x for x in res.values()) ** 0.5

    best_res = float('inf')
    best_state = None
    order = list(range(R))
    rng.shuffle(order)

    for it in range(max_iters):
        if deadline_exceeded():
            break
        free_r = order[it % R]
        u2 = solve_u_given_vw(v, w, free_r)
        if u2 is None:
            continue
        v2 = solve_v_given_uw(u2, w, free_r)
        if v2 is None:
            continue
        w2 = solve_w_given_uv(u2, v2, free_r)
        if w2 is None:
            continue
        u, v, w = u2, v2, w2
        rn = residual_norm(u, v, w)
        if rn < best_res:
            best_res = rn
            best_state = ([row[:] for row in u], [row[:] for row in v], [row[:] for row in w])
            if rn < 1e-13:
                break

    if best_state is None:
        return None
    u, v, w = best_state

    # ---- final high-precision polish pass ----
    for _ in range(30):
        if deadline_exceeded():
            break
        for free_r in range(R):
            u2 = solve_u_given_vw(v, w, free_r)
            if u2 is None:
                continue
            v2 = solve_v_given_uw(u2, w, free_r)
            if v2 is None:
                continue
            w2 = solve_w_given_uv(u2, v2, free_r)
            if w2 is None:
                continue
            u, v, w = u2, v2, w2
        rn = residual_norm(u, v, w)
        if rn < best_res:
            best_res = rn
            best_state = ([row[:] for row in u], [row[:] for row in v], [row[:] for row in w])
            if rn < 1e-14:
                break

    if best_state is None:
        return None
    u, v, w = best_state

    def round_half(x):
        h = x * 2.0
        r = round(h)
        if abs(h - r) < 1e-6:
            return Fraction(r, 2)
        return Fraction(x)

    U_frac = [[round_half(x) for x in row] for row in u]
    V_frac = [[round_half(x) for x in row] for row in v]
    W_frac = [[round_half(x) for x in row] for row in w]

    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
        return {"U": [[float(x) for x in row] for row in U_frac],
                "V": [[float(x) for x in row] for row in V_frac],
                "W": [[float(x) for x in row] for row in W_frac]}

    return None


def construct():
    n, m, p = DIMENSIONS
    fallback = _schoolbook(n, m, p)

    for seed in range(120):
        if deadline_exceeded():
            break
        result = _try_rank17(n, m, p, seed, max_iters=120)
        if result is not None:
            return result

    return fallback


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
