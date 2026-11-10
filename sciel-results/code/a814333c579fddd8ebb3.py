import numpy as np
import time
from math import gcd

_DEADLINE = time.time() + 620.0
_EPS = 1e-6


def _normalize_line(A, B, C):
    """Canonical form: divide by gcd, first nonzero coefficient positive."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    if g > 1:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _valid(points, lines):
    try:
        pts = np.asarray(points)
        lns = np.asarray(lines)
    except Exception:
        return False
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(lns)):
        return False
    if not np.all(np.abs(pts - np.round(pts)) <= _EPS):
        return False
    if not np.all(np.abs(lns - np.round(lns)) <= _EPS):
        return False
    pts = np.round(pts).astype(np.int64)
    lns = np.round(lns).astype(np.int64)
    if len(set(map(tuple, pts.tolist()))) != 54:
        return False
    seen = set()
    for A, B, C in lns.tolist():
        if A == 0 and B == 0:
            return False
        key = _normalize_line(int(A), int(B), int(C))
        if key in seen:
            return False
        seen.add(key)
    return True


def _incidence_count(points, lines):
    P = np.asarray(points, dtype=np.int64)
    L = np.asarray(lines, dtype=np.int64)
    vals = P[:, 0, None] * L[None, :, 0] + P[:, 1, None] * L[None, :, 1] + L[None, :, 2]
    return int(np.count_nonzero(vals == 0))


def _build_initial():
    lines = []
    for x in range(9):
        lines.append(_normalize_line(1, 0, -x))
    for y in range(15):
        lines.append(_normalize_line(0, 1, -y))
    for c in range(1, 11):
        lines.append(_normalize_line(1, 1, -c))
    L = np.array(lines, dtype=np.int64)

    pts = set()
    for i in range(27):
        for j in range(i + 1, 27):
            A1, B1, C1 = L[i]
            A2, B2, C2 = L[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            x_num = B1 * C2 - B2 * C1
            y_num = C1 * A2 - C2 * A1
            if x_num % det != 0 or y_num % det != 0:
                continue
            x = x_num // det
            y = y_num // det
            if -50 <= x <= 50 and -50 <= y <= 50:
                pts.add((int(x), int(y)))
    if len(pts) < 54:
        raise RuntimeError("not enough intersections")

    inc = np.zeros(len(pts), dtype=np.int64)
    for (x, y) in pts:
        for k in range(27):
            A, B, C = L[k]
            if A * x + B * y + C == 0:
                inc[k] += 1
    order = np.argsort(-inc)
    top = order[:54]
    points = np.array([pts[i] for i in top], dtype=np.int64)
    return points, L


def _candidate_lines(L, P):
    cands = {}
    for k in range(27):
        A, B, C = L[k]
        cands[_normalize_line(int(A), int(B), int(C))] = L[k]
    for i in range(54):
        for j in range(i + 1, 54):
            x1, y1 = P[i]
            x2, y2 = P[j]
            B = x1 - x2
            A = y2 - y1
            C = x2 * y1 - x1 * y2
            key = _normalize_line(int(A), int(B), int(C))
            if key not in cands:
                cands[key] = np.array([A, B, C], dtype=np.int64)
    return list(cands.values())


def _swap_point(P, L, idx, cand, inc):
    oldx, oldy = P[idx]
    old_total = 0
    for k in range(27):
        A, B, C = L[k]
        if A * oldx + B * oldy + C == 0:
            old_total += 1
            inc[k] -= 1
    new_total = 0
    for k in range(27):
        A, B, C = L[k]
        if A * cand[0] + B * cand[1] + C == 0:
            new_total += 1
            inc[k] += 1
    return new_total - old_total


def _swap_line(P, L, idx, newline, inc):
    oldA, oldB, oldC = L[idx]
    newA, newB, newC = newline
    delta = 0
    for i in range(54):
        x, y = P[i]
        old_hit = (oldA * x + oldB * y + oldC) == 0
        new_hit = (newA * x + newB * y + newC) == 0
        if old_hit:
            inc[idx] -= 1
            delta -= 1
        if new_hit:
            inc[idx] += 1
            delta += 1
    return delta


def _local_search(points, lines, budget=180.0):
    P = np.array(points, dtype=np.int64)
    L = np.array(lines, dtype=np.int64)
    inc = np.zeros(27, dtype=np.int64)
    for i in range(54):
        x, y = P[i]
        for k in range(27):
            A, B, C = L[k]
            if A * x + B * y + C == 0:
                inc[k] += 1
    base = int(inc.sum())
    best_P = P.copy()
    best_L = L.copy()
    best_val = base
    cand_lines = _candidate_lines(L, P)
    cand_points = set(map(tuple, P.tolist()))
    no_improve = 0
    t0 = time.time()
    while time.time() - t0 < budget and no_improve < 150:
        improved = False
        # --- line swaps ---
        for k in range(27):
            if time.time() > _DEADLINE:
                break
            if time.time() - t0 > budget:
                break
            cur = inc[k]
            for newline in cand_lines:
                if np.array_equal(newline, L[k]):
                    continue
                delta = _swap_line(P, L, k, newline, inc)
                if cur + delta > best_val:
                    L[k] = newline
                    best_val = cur + delta
                    best_P = P.copy()
                    best_L = L.copy()
                    improved = True
                    no_improve = 0
                    break
                else:
                    inc[k] = cur
        # --- point swaps ---
        for i in range(54):
            if time.time() > _DEADLINE:
                break
            if time.time() - t0 > budget:
                break
            cur = 0
            for k in range(27):
                A, B, C = L[k]
                if A * P[i, 0] + B * P[i, 1] + C == 0:
                    cur += 1
            for cand in cand_points:
                if cand == tuple(P[i]):
                    continue
                delta = _swap_point(P, L, i, cand, inc)
                if cur + delta > best_val:
                    P[i] = cand
                    best_val = cur + delta
                    best_P = P.copy()
                    best_L = L.copy()
                    improved = True
                    no_improve = 0
                    break
                else:
                    inc[i] = cur  # (inc is per-line, restoring handled by _swap_point)
        if not improved:
            no_improve += 1
    return best_P, best_L, best_val


def szemeredi_trotter_incidences():
    global _DEADLINE
    t0 = time.time()
    points, lines = _build_initial()
    if not _valid(points, lines):
        raise RuntimeError("initial invalid")
    best_P, best_L, best_val = _local_search(points, lines, budget=180.0)
    # safety: return best valid result found
    if time.time() > _DEADLINE - 2.0:
        return np.round(best_P).astype(np.int64), np.round(best_L).astype(np.int64)
    # one more refinement pass with remaining time
    rem = max(0.0, _DEADLINE - time.time() - 5.0)
    if rem > 20.0:
        p2, l2, v2 = _local_search(best_P, best_L, budget=rem)
        if v2 >= best_val:
            best_P, best_L, best_val = p2, l2, v2
    if not _valid(best_P, best_L):
        return np.round(points).astype(np.int64), np.round(lines).astype(np.int64)
    return np.round(best_P).astype(np.int64), np.round(best_L).astype(np.int64)
