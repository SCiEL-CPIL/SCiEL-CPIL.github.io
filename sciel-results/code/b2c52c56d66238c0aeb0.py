"""Combine grid baseline with vectorized hill-climbing refinement using incremental updates."""
import time
import numpy as np
import random

# Hard wall-clock budget for the whole construct() call (safety margin under 90s).
_DEADLINE = 85.0
_MAX_AREA = 4000  # Safe upper bound for |det| in [-30,30]^2


def _doubled_areas_full(pts):
    """Return (areas, tri_idx): areas = |det| for each triple, tri_idx = (i,j,k) arrays.

    Vectorized over all C(n,3) triples. n is small (30) so this is cheap.
    """
    n = len(pts)
    p = np.asarray(pts, dtype=np.int64)
    ix, jx, kx = np.indices((n, n, n))
    # keep strictly i<j<k
    mask = (ix < jx) & (jx < kx)
    i = ix[mask].astype(np.int64)
    j = jx[mask].astype(np.int64)
    k = kx[mask].astype(np.int64)
    xi, yi = p[i, 0], p[i, 1]
    xj, yj = p[j, 0], p[j, 1]
    xk, yk = p[k, 0], p[k, 1]
    det = (xi * (yj - yk)) + (xj * (yk - yi)) + (xk * (yi - yj))
    areas = np.abs(det)
    return areas, (i, j, k)


def _best_multiplicity(areas):
    """Largest multiplicity of a single positive doubled area."""
    pos = areas[areas > 0]
    if pos.size == 0:
        return 0
    vals, counts = np.unique(pos, return_counts=True)
    return int(counts.max())


def _evaluate(pts):
    areas, _ = _doubled_areas_full(pts)
    return _best_multiplicity(areas)


def _init_counts(pts):
    """Initialize area counts histogram and return (counts, max_count)."""
    areas, _ = _doubled_areas_full(pts)
    counts = np.zeros(_MAX_AREA + 1, dtype=np.int64)
    # Filter out 0 areas for the "positive area" metric
    pos_areas = areas[areas > 0]
    if pos_areas.size > 0:
        np.add.at(counts, pos_areas, 1)
    max_count = int(counts.max()) if counts.size > 0 else 0
    return counts, max_count


def _hill_climb(pts, deadline):
    """Local search with incremental area count updates."""
    pts = [tuple(p) for p in pts]
    n = len(pts)
    cur_set = set(pts)
    
    counts, cur_best_val = _init_counts(pts)
    
    # Precompute all pairs (i, j) with i < j for fast update when moving point k
    # When moving point idx, we need to update areas for all triples containing idx
    # Triple (a, b, c) where one is idx.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each point
        for idx in range(n):
            if time.time() >= deadline:
                break
            ox, oy = pts[idx]
            
            # Candidate moves: nearby cells + some random distant
            cands = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in cur_set:
                            cands.append((nx, ny))
            
            # Add a few random distant moves
            for _ in range(3):
                c = (random.randint(-30, 30), random.randint(-30, 30))
                if c not in cur_set:
                    cands.append(c)
            
            for c in cands:
                if time.time() >= deadline:
                    break
                nx, ny = c
                new_pts = list(pts)
                new_pts[idx] = (nx, ny)
                
                # Compute new areas for triples involving idx
                # Old areas for triples (i, j, idx) where i,j != idx
                # We need to subtract old contributions and add new ones
                
                # Efficient update:
                # For each pair (i, j) with i < j and i != idx and j != idx:
                #   old_area = |det(pts[i], pts[j], pts[idx])|
                #   new_area = |det(pts[i], pts[j], (nx, ny))|
                #   if old_area > 0: counts[old_area] -= 1
                #   if new_area > 0: counts[new_area] += 1
                
                p = np.asarray(pts, dtype=np.int64)
                # Get all other points
                others = [p[i] for i in range(n) if i != idx]
                others_idx = [i for i in range(n) if i != idx]
                
                # For each pair of others, compute old and new areas
                old_areas = []
                new_areas = []
                
                # Vectorized computation
                m = len(others)
                if m >= 2:
                    oi = np.array(others_idx)
                    # All pairs i < j in others
                    ii, jj = np.triu_indices(m, 1)
                    pi = p[others[ii]]
                    pj = p[others[jj]]
                    
                    # Old area with pts[idx]
                    xi, yi = pi[:, 0], pi[:, 1]
                    xj, yj = pj[:, 0], pj[:, 1]
                    xo, yo = ox, oy
                    old_det = xi * (yj - yo) + xj * (yo - yi) + xo * (yi - yj)
                    old_abs = np.abs(old_det)
                    
                    # New area with (nx, ny)
                    xn, yn = nx, ny
                    new_det = xi * (yj - yn) + xj * (yn - yi) + xn * (yi - yj)
                    new_abs = np.abs(new_det)
                    
                    # Update counts
                    old_pos = old_abs[old_abs > 0]
                    new_pos = new_abs[new_abs > 0]
                    
                    if old_pos.size > 0:
                        np.add.at(counts, old_pos, -1)
                    if new_pos.size > 0:
                        np.add.at(counts, new_pos, 1)
                    
                    new_max = int(counts.max())
                    
                    # Revert if not better
                    if new_max <= cur_best_val:
                        # Revert
                        if old_pos.size > 0:
                            np.add.at(counts, old_pos, 1)
                        if new_pos.size > 0:
                            np.add.at(counts, new_pos, -1)
                        continue
                    
                    # Accept move
                    pts[idx] = (nx, ny)
                    cur_set.discard((ox, oy))
                    cur_set.add((nx, ny))
                    cur_best_val = new_max
                    improved = True
                    break  # Restart point loop after improvement
    
    return pts, cur_best_val


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    # Baseline: 6x5 grid (the seed) centered to use range well.
    base = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    best_pts = [tuple(p) for p in base]
    best_val = _evaluate(best_pts)

    # Hill-climb from the baseline.
    pts, val = _hill_climb(best_pts, deadline)
    if val > best_val:
        best_pts, best_val = pts, val

    # If time remains, try a couple of alternative seeds and hill-climb each.
    if time.time() < deadline - 5:
        # Seed 2: two long parallel lines
        seed2 = [(x, 0) for x in range(-7, 8)] + [(x, 1) for x in range(-7, 8)]
        p2, v2 = _hill_climb(seed2, deadline)
        if v2 > best_val:
            best_pts, best_val = p2, v2

    # If time remains, try a 5x6 grid
    if time.time() < deadline - 5:
        seed3 = [(x, y) for x in range(-2, 4) for y in range(-2, 4)]
        p3, v3 = _hill_climb(seed3, deadline)
        if v3 > best_val:
            best_pts, best_val = p3, v3

    # Final guard: ensure exactly 30 distinct points in range.
    seen = set()
    out = []
    for p in best_pts:
        x, y = p
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    # If somehow short (shouldn't happen), pad with fresh cells.
    if len(out) < 30:
        for x in range(-30, 31):
            for y in range(-30, 31):
                if len(out) >= 30:
                    break
                if (x, y) not in seen:
                    seen.add((x, y))
                    out.append((x, y))
    return out[:30]
