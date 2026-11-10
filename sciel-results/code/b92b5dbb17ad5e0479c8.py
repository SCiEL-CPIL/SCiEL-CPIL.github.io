# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations


def _cross(px, py, qx, qy, rx, ry):
    return (qx - px) * (ry - qy) - (qy - py) * (rx - qx)


def _count_convex7(pts):
    n = len(pts)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    P = pts[order]
    N = n
    cnt = 0
    for i in range(N):
        pi = P[i]
        j = np.arange(N)
        later = j > i
        if later.sum() < 7:
            continue
        ang = np.arctan2(P[1, :] - pi[1], P[0, :] - pi[0])
        ang = np.where(later, ang, np.pi)
        idx = np.argsort(ang)
        idx = idx[later[idx]]
        m = len(idx)
        if m < 7:
            continue
        # convex chain DP: dp[t][c] = # convex chains of length c ending at position t
        dp = np.zeros((m, 7), dtype=np.int64)
        for t in range(1, m):
            pt = P[idx[t]]
            for s in range(t):
                ps = P[idx[s]]
                c = _cross(pi[0], pi[1], ps[0], ps[1], pt[0], pt[1])
                if c > 1e-9:
                    dp[t, 1] += 1
                    for k in range(2, 7):
                        dp[t, k] += dp[s, k - 1]
        # close the 7-gon: need chain of length 6 ending at t with a convex turn back to pivot
        for t in range(1, m):
            pt = P[idx[t]]
            c = _cross(pt[0], pt[1], pi[0], pi[1], P[idx[0]][0], P[idx[0]][1])
            if c > 1e-9:
                cnt += dp[t, 6]
    return cnt


def _is_valid(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    # min squared distance
    for i in range(33):
        for j in range(i + 1, 33):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            if dx * dx + dy * dy < 1e-12:
                return False
    # no three collinear
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                v = abs(pts[i, 0] * (pts[j, 1] - pts[k, 1]) +
                        pts[j, 0] * (pts[k, 1] - pts[i, 1]) +
                        pts[k, 0] * (pts[i, 1] - pts[j, 1]))
                if v < 1e-9:
                    return False
    return True


def _generate_cup_cap():
    n = 33
    # 17 points on a lower convex arc (cup), 16 on an upper convex arc (cap)
    pts = []
    n_low = 17
    n_up = 16
    for i in range(n_low):
        t = i / (n_low - 1)
        x = t
        # shallow downward arc
        y = 0.05 + 0.9 * (t * t) * 0.5 + 0.0
        y = 0.05 + 0.9 * (1 - (1 - t) ** 2) * 0.0 + 0.0
        y = 0.05 + 0.9 * t * t
        pts.append([x, y])
    for i in range(n_up):
        t = i / (n_up - 1)
        x = t
        y = 0.95 - 0.9 * t * t
        pts.append([x, y])
    pts = np.array(pts, dtype=float)
    return pts


def erdos_szekeres_problem7() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0
    safety = 60.0

    # Strong deterministic seed: cup-and-cap
    best = _generate_cup_cap().copy()
    if not _is_valid(best):
        # fallback: jittered grid
        best = np.random.RandomState(0).rand(33, 2)
        best = np.clip(best, 1e-4, 1 - 1e-4)

    best_score = _count_convex7(best)

    rng = np.random.RandomState(12345)

    def elapsed():
        return time.time() - start

    def try_candidate(cand):
        nonlocal best, best_score
        if not _is_valid(cand):
            return
        sc = _count_convex7(cand)
        if sc < best_score:
            best_score = sc
            best = cand.copy()

    # Local search: coordinate nudges and swaps
    while elapsed() < deadline - safety:
        improved = False
        # coordinate nudge pass
        for _ in range(3):
            if elapsed() > deadline - safety:
                break
            i = rng.randint(33)
            axis = rng.randint(2)
            step = rng.choice([-1, 1]) * rng.uniform(0.01, 0.08)
            cand = best.copy()
            cand[i, axis] += step
            if 1e-5 <= cand[i, axis] <= 1 - 1e-5:
                try_candidate(cand)
        # swap pass
        for _ in range(2):
            if elapsed() > deadline - safety:
                break
            i, j = rng.choice(33, size=2, replace=False)
            cand = best.copy()
            cand[[i, j], :] = cand[[j, i], :]
            try_candidate(cand)

    return best


# EVOLVE-BLOCK-END
