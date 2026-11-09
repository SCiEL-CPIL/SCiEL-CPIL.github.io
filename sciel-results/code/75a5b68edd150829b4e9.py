# EVOLVE-BLOCK-START
import time
import numpy as np


def _cross(a, b, c):
    """Signed cross product of vectors (b-a) and (c-b): (bx-ax)(cy-by) - (by-ay)(cx-bx)."""
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def _is_valid(pts):
    """Check validity constraints: shape, range, distinctness, non-collinearity."""
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    # pairwise squared distance
    d2 = (pts[:, None, 0] - pts[None, :, 0]) ** 2 + (pts[:, None, 1] - pts[None, :, 1]) ** 2
    if np.any(np.triu(d2, 1) < 1e-12):
        return False
    # collinearity
    for i in range(33):
        for j in range(i + 1, 33):
            for l in range(j + 1, 33):
                v = pts[i, 0] * (pts[j, 1] - pts[l, 1]) + pts[j, 0] * (pts[l, 1] - pts[i, 1]) + pts[l, 0] * (pts[i, 1] - pts[j, 1])
                if abs(v) < 1e-9:
                    return False
    return True


def _count_convex7(pts):
    """Count convex 7-gons via the dynamic-program-over-convex-chains method."""
    n = len(pts)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]
    C = 0
    for i in range(n):
        later = list(range(i + 1, n))
        if len(later) < 6:
            continue
        later.sort(key=lambda j: np.arctan2(p[j, 1] - p[i, 1], p[j, 0] - p[i, 0]))
        m = len(later)
        # dp[j][t] = number of convex chains of length t ending at later[j] (in angular order)
        # A chain is convex if every consecutive triple (including pivot) makes a positive turn.
        dp = [[0] * (8) for _ in range(m)]
        for j in range(m):
            # chains of length 2: pivot -> later[j], valid
            dp[j][2] = 1
        for t in range(3, 7):
            for j in range(t - 1, m):
                s = 0
                for k in range(j - 1, -1, -1):
                    if dp[k][t - 1] == 0:
                        continue
                    # turn from later[k-1] to later[k] to later[j] must be positive
                    # (the previous edge in the chain is later[k-1] -> later[k])
                    if k == 0:
                        # previous vertex is pivot i
                        a = p[i]
                    else:
                        a = p[later[k - 1]]
                    b = p[later[k]]
                    c = p[later[j]]
                    if _cross(a, b, c) > 1e-9:
                        s += dp[k][t - 1]
                dp[j][t] = s
        # close to a 7-gon: chain of length 6 (pivot + 5) ending at later[j];
        # need turn later[j-1] -> later[j] -> pivot positive, and pivot -> later[0] -> later[1] positive
        for j in range(5, m):
            if dp[j][6] == 0:
                continue
            # closing turn: later[j-1] -> later[j] -> pivot
            a = p[later[j - 1]]
            b = p[later[j]]
            c = p[i]
            if _cross(a, b, c) <= 1e-9:
                continue
            # opening turn: pivot -> later[0] -> later[1]
            a = p[i]
            b = p[later[0]]
            c = p[later[1]]
            if _cross(a, b, c) <= 1e-9:
                continue
            C += dp[j][6]
    return C


def _make_cup_cap(cup_xs, cup_ys, cap_xs, cap_ys):
    """Build the 33-point array: cup points then cap points."""
    n = len(cup_xs) + len(cap_xs)
    pts = np.empty((n, 2), dtype=float)
    pts[:len(cup_xs), 0] = cup_xs
    pts[:len(cup_xs), 1] = cup_ys
    pts[len(cup_xs):, 0] = cap_xs
    pts[len(cup_xs):, 1] = cap_ys
    return pts


def _cup_cap_value(cup_xs, cup_ys, cap_xs, cap_ys):
    pts = _make_cup_cap(cup_xs, cup_ys, cap_xs, cap_ys)
    if not _is_valid(pts):
        return -1.0
    C = _count_convex7(pts)
    return 1.0 / (1.0 + C)


def erdos_szekeres_problem7() -> np.ndarray:
    deadline = time.time() + 580.0  # safety margin before 630s kill

    # ---- Base cup-cap construction: cup of 17, cap of 16 ----
    # Cup: increasing x, convex (curving upward). Cap: increasing x, concave (curving downward).
    # To guarantee general position and avoid collinearity, add tiny deterministic wiggles.
    base_cup_xs = np.linspace(0.02, 0.98, 17)
    base_cup_ys = np.linspace(0.05, 0.95, 17) ** 1.6   # convex (increasing slope)
    base_cap_xs = np.linspace(0.02, 0.98, 16)
    base_cap_ys = 1.0 - np.linspace(0.05, 0.95, 16) ** 1.6  # concave (cap)

    # Deterministic tiny wiggles to break any accidental collinearity.
    rng = np.random.RandomState(42)
    cup_xs = base_cup_xs + rng.uniform(-1e-4, 1e-4, 17)
    cup_ys = base_cup_ys + rng.uniform(-1e-4, 1e-4, 17)
    cap_xs = base_cap_xs + rng.uniform(-1e-4, 1e-4, 16)
    cap_ys = base_cap_ys + rng.uniform(-1e-4, 1e-4, 16)

    best_pts = _make_cup_cap(cup_xs, cup_ys, cap_xs, cap_ys)
    best_val = _cup_cap_value(cup_xs, cup_ys, cap_xs, cap_ys)
    if best_val < 0:
        # fallback: pure random (should not happen)
        rng2 = np.random.RandomState(0)
        best_pts = rng2.rand(33, 2)
        best_val = 1.0

    # ---- Refinement: perturb points to reduce convex 7-gons, keep validity ----
    # Each perturbation is small; we accept if it lowers C (raises value) and stays valid.
    rng = np.random.RandomState(7)
    C_best = None
    if best_val > 0:
        C_best = int(round(1.0 / best_val - 1.0))

    step = 1e-3
    attempts = 0
    while time.time() < deadline - 30.0:
        attempts += 1
        # pick a random point and perturb
        idx = rng.randint(0, 33)
        delta = rng.normal(0.0, step, 2)
        new_pts = best_pts.copy()
        new_pts[idx] = best_pts[idx] + delta
        # clamp into [0,1]
        new_pts = np.clip(new_pts, 0.0, 1.0)
        if not _is_valid(new_pts):
            continue
        C = _count_convex7(new_pts)
        if C < (C_best if C_best is not None else float('inf')):
            best_pts = new_pts
            C_best = C
            best_val = 1.0 / (1.0 + C)
        # occasionally shrink step
        if attempts % 200 == 0:
            step *= 0.85
            if step < 1e-5:
                break

    # Final safety: ensure validity
    if not _is_valid(best_pts):
        rng2 = np.random.RandomState(0)
        best_pts = rng2.rand(33, 2)

    return best_pts


# EVOLVE-BLOCK-END
