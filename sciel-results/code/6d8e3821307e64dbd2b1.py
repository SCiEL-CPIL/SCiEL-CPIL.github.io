# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd


def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_key(A, B, C):
    return _normalize_line(A, B, C)


def _build_line_structure(points):
    """Return dict: normalized line key -> set of point indices on that line (>=2 pts)."""
    pts = points.tolist()
    n = len(pts)
    line_to_indices = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            s = line_to_indices.get(key)
            if s is None:
                line_to_indices[key] = {i, j}
            else:
                s.add(i)
                s.add(j)
    return {k: v for k, v in line_to_indices.items() if len(v) >= 2}


def _top27_from_structure(line_dict):
    """Pick the 27 lines maximizing total incidences (greedy by count)."""
    # sort by count desc, then key for determinism
    items = sorted(line_dict.items(), key=lambda t: (-len(t[1]), t[0]))
    chosen = [k for k, _ in items[:27]]
    total = sum(len(line_dict[k]) for k in chosen)
    return chosen, total


def _evaluate(points, lines):
    total = 0
    for A, B, C in lines:
        cnt = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                cnt += 1
        total += cnt
    return total


def _generate_grid(w, h, ox=0, oy=0):
    xs = np.arange(w, dtype=np.int64) + ox
    ys = np.arange(h, dtype=np.int64) + oy
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _remove_points_from_grid(points, remove_set):
    """Remove points at given indices from the (w*h) grid, return 54-point array."""
    n = len(points)
    keep = [i for i in range(n) if i not in remove_set]
    if len(keep) != 54:
        return None
    return points[keep].astype(np.int64)


def _search_grid(w, h, deadline, best):
    """Try all 2-point removals from a w*h grid (w*h=56), return best (points, lines, score)."""
    if w * h != 56:
        # only handle 8x7 / 7x8 for now
        return best
    base_points = _generate_grid(w, h)
    n = w * h  # 56
    # Precompute the full 56-point line structure for speed of evaluation
    full_line_dict = _build_line_structure(base_points)
    # We need to evaluate removals. For each removal set, build the 54-point structure.
    # To be fast, iterate over all C(56,2)=1540 removal sets.
    indices = list(range(n))
    best_local = best
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            if time.time() > deadline - 60:
                return best_local
            remove_set = {r1, r2}
            pts = _remove_points_from_grid(base_points, remove_set)
            if pts is None:
                continue
            line_dict = _build_line_structure(pts)
            if len(line_dict) < 27:
                continue
            chosen, total = _top27_from_structure(line_dict)
            if total > best_local[2]:
                lines = np.array(chosen, dtype=np.int64)
                best_local = (pts.copy(), lines, total)
    return best_local


def _try_local_improvement(points, lines, score, deadline):
    """Local search: move individual points to nearby integer positions to gain incidences."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = score

    for _round in range(3):
        if time.time() > deadline - 30:
            break
        improved = False
        point_set = set(map(tuple, best_points.tolist()))
        for i in range(len(best_points)):
            if time.time() > deadline - 30:
                break
            x0, y0 = best_points[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if (nx, ny) in point_set:
                        continue
                    test_points = best_points.copy()
                    test_points[i] = [nx, ny]
                    # recompute best 27 lines
                    line_dict = _build_line_structure(test_points)
                    if len(line_dict) < 27:
                        continue
                    chosen, total = _top27_from_structure(line_dict)
                    if total > best_score:
                        best_points = test_points
                        best_lines = np.array(chosen, dtype=np.int64)
                        best_score = total
                        point_set = set(map(tuple, best_points.tolist()))
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    return best_points, best_lines, best_score


def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 600.0

    best_points = None
    best_lines = None
    best_score = -1

    # Search 8x7 grid with all 2-point removals
    best_points, best_lines, best_score = _search_grid(8, 7, deadline, (best_points, best_lines, best_score))
    best_points, best_lines, best_score = _search_grid(7, 8, deadline, (best_points, best_lines, best_score))

    # Fallback: if nothing found, use a plain grid
    if best_points is None:
        pts = _generate_grid(9, 6)
        line_dict = _build_line_structure(pts)
        chosen, total = _top27_from_structure(line_dict)
        best_points = pts
        best_lines = np.array(chosen, dtype=np.int64)
        best_score = total

    # Local improvement
    if time.time() < deadline - 30:
        best_points, best_lines, best_score = _try_local_improvement(
            best_points, best_lines, best_score, deadline
        )

    # Finalize
    best_points = np.round(best_points).astype(np.int64)
    best_lines = np.round(best_lines).astype(np.int64)

    # Validation
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    assert len(set(map(tuple, best_points.tolist()))) == 54
    for A, B, C in best_lines:
        assert not (A == 0 and B == 0)
    seen = set()
    for A, B, C in best_lines:
        k = _normalize_line(int(A), int(B), int(C))
        assert k not in seen
        seen.add(k)
    assert np.all(np.isfinite(best_points))
    assert np.all(np.isfinite(best_lines))

    return best_points, best_lines


# EVOLVE-BLOCK-END
