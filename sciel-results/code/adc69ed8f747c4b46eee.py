import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """points (N,2), lines (M,3) -> total number of incidences."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    res = A * x + B * y + C
    return int(np.sum(res == 0))

def _build_candidates(points, W, H):
    """Build a broad set of candidate lines (normalized) that can hit grid points."""
    pts = points
    cands = set()
    # Verticals
    for c in range(W):
        cands.add((1, 0, -c))
    # Horizontals
    for c in range(H):
        cands.add((0, 1, -c))
    # All rational slopes: line through two grid points
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = int(x2 - x1)
            dy = int(y2 - y1)
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # direction (dx,dy) -> normal (dy, -dx), C = dx*y1 - dy*x1
            A, B = dy, -dx
            C = dx * y1 - dy * x1
            n = _normalize_line(A, B, C)
            if n is not None:
                cands.add(n)
    return np.array(list(cands), dtype=np.int64)

def _top_lines_for_points(points, lines):
    """Return (top-27 line indices, incidence count) for the given points."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    res = A * x + B * y + C
    line_cov = np.sum(res == 0, axis=1)
    if len(line_cov) >= 27:
        top = np.argsort(-line_cov)[:27]
    else:
        top = np.arange(len(line_cov))
    return top, int(np.sum(line_cov[top]))

def _top_points_for_lines(grid_points, lines, k=54):
    """Return top-k grid points by coverage of the given lines."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = grid_points[:, 0:1].T
    y = grid_points[:, 1:2].T
    res = A * x + B * y + C
    point_cov = np.sum(res == 0, axis=0)
    if len(point_cov) >= k:
        idx = np.argsort(-point_cov)[:k]
    else:
        idx = np.arange(len(point_cov))
    return grid_points[idx]

def _local_refine(points, lines, deadline):
    """Move individual points to nearby lattice positions to raise incidence."""
    A_l = lines[:, 0].astype(np.int64)
    B_l = lines[:, 1].astype(np.int64)
    C_l = lines[:, 2].astype(np.int64)

    def pt_inc(pt):
        return int(np.sum(A_l * pt[0] + B_l * pt[1] + C_l == 0))

    cur = np.array(points, dtype=np.int64).copy()
    total = _count_incidence(cur, lines)
    for k in range(len(cur)):
        if time.time() > deadline:
            break
        bx, by = int(cur[k, 0]), int(cur[k, 1])
        best_local = total - pt_inc((bx, by))
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nxp, nyp = bx + dx, by + dy
                if any(int(cur[j, 0]) == nxp and int(cur[j, 1]) == nyp for j in range(len(cur)) if j != k):
                    continue
                nl = pt_inc((nxp, nyp))
                if nl > best_local:
                    best_local = nl
                    bx, by = nxp, nyp
        if (bx, by) != (int(cur[k, 0]), int(cur[k, 1])):
            total += pt_inc((bx, by)) - pt_inc((int(cur[k, 0]), int(cur[k, 1])))
            cur[k] = [bx, by]
    return cur, total

def _swap_refine(points, lines, grid_points, deadline):
    """Try swapping one selected point for one unselected grid point."""
    cur = np.array(points, dtype=np.int64).copy()
    total = _count_incidence(cur, lines)
    # occupancy
    occ = set()
    for p in cur:
        occ.add((int(p[0]), int(p[1])))
    unselected = [gp for gp in grid_points if (int(gp[0]), int(gp[1])) not in occ]

    A_l = lines[:, 0].astype(np.int64)
    B_l = lines[:, 1].astype(np.int64)
    C_l = lines[:, 2].astype(np.int64)
    def pt_inc(pt):
        return int(np.sum(A_l * pt[0] + B_l * pt[1] + C_l == 0))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for k in range(len(cur)):
            if time.time() > deadline:
                break
            base = total - pt_inc((int(cur[k, 0]), int(cur[k, 1])))
            for gp in unselected:
                gain = pt_inc((int(gp[0]), int(gp[1])))
                if base + gain > total:
                    occ.discard((int(cur[k, 0]), int(cur[k, 1])))
                    occ.add((int(gp[0]), int(gp[1])))
                    cur[k] = [int(gp[0]), int(gp[1])]
                    total = base + gain
                    improved = True
                    break
    return cur, total

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin under 630s wall-clock

    best_score = -1
    best_points = None
    best_lines = None

    def consider(points, lines):
        nonlocal best_score, best_points, best_lines
        # ensure exactly 54 points and 27 lines
        if len(points) != 54 or len(lines) != 27:
            return
        score = _count_incidence(points, lines)
        if score > best_score:
            best_score = score
            best_points = np.array(points, dtype=np.int64).copy()
            best_lines = np.array(lines, dtype=np.int64).copy()

    for W, H in [(7, 8), (8, 7)]:
        if time.time() > deadline:
            break
        xs = np.arange(W, dtype=np.int64)
        ys = np.arange(H, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        grid_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        n_grid = grid_points.shape[0]  # 56

        # Broad candidate line pool (hypergraph edges)
        lines_arr = _build_candidates(grid_points, W, H)
        if len(lines_arr) == 0:
            continue

        # ---- Phase A: exhaustive removal of 2 grid points, top-27 lines ----
        for i, j in combinations(range(n_grid), 2):
            if time.time() > deadline:
                break
            mask = np.ones(n_grid, dtype=bool)
            mask[i] = False
            mask[j] = False
            pts = grid_points[mask]
            top, score = _top_lines_for_points(pts, lines_arr)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                best_lines = lines_arr[top].copy()

        # ---- Phase B: iterative co-optimization (hypergraph vertex/edge selection) ----
        # init: top-27 lines on full grid, top-54 points on those lines
        top, _ = _top_lines_for_points(grid_points, lines_arr)
        cur_lines = lines_arr[top]
        cur_points = _top_points_for_lines(grid_points, cur_lines, 54)
        cur_score = _count_incidence(cur_points, cur_lines)
        if cur_score > best_score:
            best_score = cur_score
            best_points = cur_points.copy()
            best_lines = cur_lines.copy()

        for _ in range(60):
            if time.time() > deadline:
                break
            # update lines from points
            top, _ = _top_lines_for_points(cur_points, lines_arr)
            new_lines = lines_arr[top]
            # update points from lines
            new_points = _top_points_for_lines(grid_points, new_lines, 54)
            new_score = _count_incidence(new_points, new_lines)
            if new_score > cur_score:
                cur_score = new_score
                cur_points, cur_lines = new_points, new_lines
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = cur_points.copy()
                    best_lines = cur_lines.copy()
            else:
                # stuck -> try perturbation: randomize a few lines, re-optimize points
                if len(lines_arr) > 27:
                    perm = np.random.permutation(len(lines_arr))
                    rand_lines = lines_arr[perm[:27]]
                    rp = _top_points_for_lines(grid_points, rand_lines, 54)
                    rs = _count_incidence(rp, rand_lines)
                    # also re-optimize lines for these points
                    top2, _ = _top_lines_for_points(rp, lines_arr)
                    rl = lines_arr[top2]
                    rs2 = _count_incidence(rp, rl)
                    if max(rs, rs2) > cur_score:
                        if rs2 >= rs:
                            cur_points, cur_lines, cur_score = rp, rl, rs2
                        else:
                            cur_points, cur_lines, cur_score = rp, rand_lines, rs
                        if cur_score > best_score:
                            best_score = cur_score
                            best_points = cur_points.copy()
                            best_lines = cur_lines.copy()
                    else:
                        break
                else:
                    break

        # ---- Phase C: local + swap refinement on the best configuration so far ----
        if best_points is not None:
            refined, _ = _local_refine(best_points, best_lines, deadline)
            consider(refined, best_lines)
            swapped, _ = _swap_refine(best_points, best_lines, grid_points, deadline)
            consider(swapped, best_lines)
            # re-optimize lines after refinement, then refine again
            if time.time() < deadline:
                top, _ = _top_lines_for_points(refined, lines_arr)
                nl = lines_arr[top]
                consider(refined, nl)
                refined2, _ = _local_refine(refined, nl, deadline)
                consider(refined2, nl)

    # Fallback safety: if nothing valid, build a minimal valid answer
    if best_points is None or len(best_points) != 54:
        W, H = 7, 8
        xs = np.arange(W, dtype=np.int64)
        ys = np.arange(H, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        gp = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_points = gp[:54].copy()
        lines_arr = _build_candidates(gp, W, H)
        top, _ = _top_lines_for_points(best_points, lines_arr)
        best_lines = lines_arr[top].copy()

    # Final validation / cleanup: enforce integer int64, no duplicates, unique lines
    best_points = np.asarray(best_points, dtype=np.int64)
    best_lines = np.asarray(best_lines, dtype=np.int64)

    # dedup points
    if len(best_points) > 54:
        _, idx = np.unique(best_points, axis=0, return_index=True)
        best_points = best_points[np.sort(idx)[:54]]
    # dedup lines
    if len(best_lines) > 27:
        norm = []
        for A, B, C in best_lines:
            n = _normalize_line(int(A), int(B), int(C))
            if n is not None:
                norm.append(n)
        seen = set()
        uniq = []
        for n in norm:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        best_lines = np.array(uniq[:27], dtype=np.int64)

    # Ensure shapes exactly (54,2) and (27,3)
    if len(best_points) < 54:
        # pad with distinct far points (should not happen)
        pad = np.array([[1000 + i, 0] for i in range(54 - len(best_points))], dtype=np.int64)
        best_points = np.vstack([best_points, pad])
    if len(best_lines) < 27:
        pad = np.array([[1, 0, -2000 - i] for i in range(27 - len(best_lines))], dtype=np.int64)
        best_lines = np.vstack([best_lines, pad])

    return best_points, best_lines
