"""Combine grid baseline with vectorized hill-climbing refinement and incremental area updates."""
import time
import numpy as np
import random
from collections import Counter

# Hard wall-clock budget for the whole construct() call (safety margin under 90s).
_DEADLINE = 78.0
_MAX_AREA = 4000  # Safe upper bound for |det| in [-30,30]^2


def _doubled_areas(pts):
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
    areas, _ = _doubled_areas(pts)
    return _best_multiplicity(areas)


def _init_counts(pts):
    """Initialize area counts histogram and return (counts, max_count)."""
    areas, _ = _doubled_areas(pts)
    counts = np.zeros(_MAX_AREA + 1, dtype=np.int64)
    pos_areas = areas[areas > 0]
    if pos_areas.size > 0:
        np.add.at(counts, pos_areas, 1)
    max_count = int(counts.max()) if counts.size > 0 else 0
    return counts, max_count


def _hill_climb(pts, deadline, radius=3, random_trials=6):
    """Local search: move a point to a nearby empty integer cell if it improves
    the max positive-area multiplicity (tie-break: total area concentration)."""
    pts = [tuple(p) for p in pts]
    n = len(pts)
    cur = _evaluate(pts)
    cur_set = set(pts)

    # Precompute all candidate cells in range.
    cells = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]

    improved = True
    stagnation = 0
    while improved and time.time() < deadline:
        improved = False
        # Try each point; for each, try a bounded set of nearby moves.
        for idx in range(n):
            if time.time() >= deadline:
                break
            ox, oy = pts[idx]
            # candidate new positions: neighbors within given radius
            cands = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in cur_set:
                            cands.append((nx, ny))
            # Also try a few random distant cells to escape local minima.
            for _ in range(random_trials):
                c = random.choice(cells)
                if c not in cur_set:
                    cands.append(c)
            for c in cands:
                if time.time() >= deadline:
                    break
                new_pts = list(pts)
                new_pts[idx] = c
                new_set = cur_set - {pts[idx]} | {c}
                # quick filter: must keep 30 distinct
                if len(new_set) != 30:
                    continue
                new_val = _evaluate(new_pts)
                if new_val > cur:
                    # accept
                    cur_set = new_set
                    pts = new_pts
                    cur = new_val
                    improved = True
                    stagnation = 0
                    break
        if not improved:
            stagnation += 1
            if stagnation >= 3:
                break
    return pts, cur


def _hexagonal_seed(n=30, cx=0, cy=0, scale=3):
    """Generate a hexagonal lattice seed of n points centered at (cx,cy)."""
    pts = []
    # Hexagonal packing: rows offset by half spacing
    row_y = 0
    row_offset = 0
    while len(pts) < n:
        for x in range(-15, 16):
            px = cx + x * scale + (row_offset * scale // 2)
            py = cy + row_y * scale
            if -30 <= px <= 30 and -30 <= py <= 30 and (px, py) not in pts:
                pts.append((px, py))
            if len(pts) >= n:
                break
        row_y += 1
        row_offset = 1 - row_offset
    return pts[:n]


def _grid_seed(rows, cols, cx=0, cy=0, sx=3, sy=3):
    """Generate a rectangular grid seed centered at (cx,cy)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - cols // 2) * sx
            y = cy + (r - rows // 2) * sy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts[:30]


def _two_line_seed(n_per_line=15, spacing=2):
    """Two parallel lines with evenly spaced points."""
    pts = []
    start_x = -n_per_line * spacing // 2
    for i in range(n_per_line):
        pts.append((start_x + i * spacing, 0))
        pts.append((start_x + i * spacing, 1))
    return pts[:30]


def _sheared_grid_seed(rows, cols, cx=0, cy=0, sx=3, sy=3, shear=1):
    """Generate a sheared grid seed: each row is shifted by shear * row_index."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - cols // 2) * sx + r * shear
            y = cy + (r - rows // 2) * sy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts[:30]


def _clustered_random_seed(n=30, cx=0, cy=0, spread=5, seed=42):
    """Generate a clustered random seed: points concentrated near center."""
    rng = random.Random(seed)
    pts = set()
    while len(pts) < n:
        x = rng.randint(cx - spread, cx + spread)
        y = rng.randint(cy - spread, cy + spread)
        if -30 <= x <= 30 and -30 <= y <= 30:
            pts.add((x, y))
    return list(pts)[:n]


def _simulated_annealing(pts, deadline, T_start=100.0, T_end=0.1, cooling=0.999):
    """Simulated Annealing to find a configuration with high max multiplicity.
    
    Objective: Minimize sum of squared deviations of positive areas from the current best area.
    This encourages clustering of areas around a single value.
    """
    pts = [tuple(p) for p in pts]
    n = len(pts)
    cur_set = set(pts)
    
    # Initial evaluation
    areas, _ = _doubled_areas(pts)
    pos_areas = areas[areas > 0]
    if pos_areas.size == 0:
        best_val = 0
        best_area = 0
    else:
        vals, counts = np.unique(pos_areas, return_counts=True)
        best_idx = np.argmax(counts)
        best_area = int(vals[best_idx])
        best_val = int(counts[best_idx])
        
    current_obj = np.sum((pos_areas - best_area) ** 2) if pos_areas.size > 0 else 0
    
    T = T_start
    rng = np.random.RandomState(42)
    
    while time.time() < deadline and T > T_end:
        # Select a point to move
        idx = rng.randint(0, n)
        ox, oy = pts[idx]
        
        # Generate a candidate move: random step in a box around current point
        # Box size 5 is good for local exploration
        size = 5
        nx = ox + rng.randint(-size, size)
        ny = oy + rng.randint(-size, size)
        
        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            T *= cooling
            continue
            
        new_pt = (nx, ny)
        if new_pt in cur_set:
            T *= cooling
            continue
            
        # Perform move
        old_pt = pts[idx]
        pts[idx] = new_pt
        cur_set.remove(old_pt)
        cur_set.add(new_pt)
        
        # Evaluate new state
        new_areas, _ = _doubled_areas(pts)
        new_pos = new_areas[new_areas > 0]
        
        if new_pos.size == 0:
            new_obj = 1e9 # Penalize empty area sets
            new_best_val = 0
            new_best_area = 0
        else:
            # We want to minimize deviation from the *current* best area to keep the landscape consistent
            # But if the new best area is different, we should re-evaluate.
            # Actually, a simpler objective is just the max count, but that's discrete.
            # Let's use: Objective = -max_count * 1000 + sum((areas - best_area)^2)
            # This prioritizes count, then tightness.
            
            # To make it fast, let's just check if max count improved.
            # If it improved, accept.
            # If not, calculate objective for acceptance.
            
            new_vals, new_counts = np.unique(new_pos, return_counts=True)
            new_max_count = int(new_counts.max())
            new_best_area_idx = np.argmax(new_counts)
            new_best_area = int(new_vals[new_best_area_idx])
            
            # New objective: primarily count, secondarily tightness
            # We want to minimize "cost". Cost = -count * 1000 + tightness
            # But tightness depends on which area is "the" area.
            # Let's stick to a simpler robust objective:
            # Cost = sum((areas - best_area)^2) - count * 50
            # This rewards high count and low variance.
            
            # However, if the best area changes, the variance term jumps.
            # Let's just use the max count as the primary driver and variance as secondary for tie-breaking/annealing.
            
            new_obj = np.sum((new_pos - new_best_area) ** 2) - new_max_count * 100
            
        # Acceptance probability
        if new_obj < current_obj:
            accept = True
        else:
            delta = new_obj - current_obj
            # T is temperature
            if T > 0:
                prob = np.exp(-delta / T)
                accept = (rng.rand() < prob)
            else:
                accept = False
                
        if accept:
            current_obj = new_obj
            best_val = new_max_count if new_pos.size > 0 else 0
            best_area = new_best_area if new_pos.size > 0 else 0
        else:
            # Revert
            pts[idx] = old_pt
            cur_set.remove(new_pt)
            cur_set.add(old_pt)
            
        T *= cooling
        
    return pts, best_val


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    random.seed(42)

    # Multiple diverse seeds
    seeds = [
        ("grid_6x5", [(x, y) for x in range(-2, 4) for y in range(-2, 3)]),
        ("grid_5x6", _grid_seed(5, 6, cx=0, cy=0, sx=3, sy=3)),
        ("hex_scale3", _hexagonal_seed(30, 0, 0, scale=3)),
        ("hex_scale4", _hexagonal_seed(30, 0, 0, scale=4)),
        ("two_lines", _two_line_seed(15, spacing=2)),
        ("grid_10x3", _grid_seed(10, 3, cx=0, cy=0, sx=5, sy=3)),
        ("sheared_6x5", _sheared_grid_seed(6, 5, cx=0, cy=0, sx=3, sy=3, shear=1)),
        ("sheared_5x6", _sheared_grid_seed(5, 6, cx=0, cy=0, sx=3, sy=3, shear=-1)),
        ("clustered", _clustered_random_seed(30, 0, 0, spread=5, seed=42)),
        ("clustered2", _clustered_random_seed(30, 0, 0, spread=4, seed=123)),
    ]

    best_pts = None
    best_val = 0

    # Phase 1: Quick hill-climb from each seed with moderate radius
    per_seed_budget = (deadline - start) / len(seeds)
    for name, seed in seeds:
        if time.time() >= deadline - 2:
            break
        seed_pts = [tuple(p) for p in seed[:30]]
        # Ensure exactly 30 distinct points in range
        seen = set()
        valid_seed = []
        for p in seed_pts:
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in seen:
                seen.add(p)
                valid_seed.append(p)
        # Pad if needed
        if len(valid_seed) < 30:
            for x in range(-30, 31):
                for y in range(-30, 31):
                    if len(valid_seed) >= 30:
                        break
                    if (x, y) not in seen:
                        seen.add((x, y))
                        valid_seed.append((x, y))
        valid_seed = valid_seed[:30]

        seed_deadline = min(deadline - 2, time.time() + per_seed_budget)
        pts, val = _hill_climb(valid_seed, seed_deadline, radius=2, random_trials=4)
        if val > best_val:
            best_val = val
            best_pts = pts

    # Phase 2: Simulated Annealing from the best solution
    if best_pts is not None and time.time() < deadline - 10:
        sa_deadline = deadline - 2
        pts_sa, val_sa = _simulated_annealing(best_pts, sa_deadline)
        if val_sa > best_val:
            best_val = val_sa
            best_pts = pts_sa
        # Re-hill-climb after SA to polish
        if time.time() < deadline - 2:
            pts2, val2 = _hill_climb(best_pts, deadline - 1, radius=3, random_trials=5)
            if val2 > best_val:
                best_val = val2
                best_pts = pts2

    # Phase 3: If time remains, try perturbation restarts from the best
    if best_pts is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() >= deadline - 2:
                break
            # Perturb: randomly move 3-5 points
            perturbed = list(best_pts)
            cur_set = set(perturbed)
            num_moves = random.randint(3, 5)
            for _m in range(num_moves):
                idx = random.randint(0, len(perturbed) - 1)
                ox, oy = perturbed[idx]
                for _attempt in range(20):
                    nx = random.randint(max(-30, ox - 8), min(30, ox + 8))
                    ny = random.randint(max(-30, oy - 8), min(30, oy + 8))
                    if (nx, ny) not in cur_set:
                        cur_set.discard(perturbed[idx])
                        cur_set.add((nx, ny))
                        perturbed[idx] = (nx, ny)
                        break
            p3, v3 = _hill_climb(perturbed, min(deadline - 1, time.time() + 8), radius=3, random_trials=6)
            if v3 > best_val:
                best_val = v3
                best_pts = p3

    # Final guard: ensure exactly 30 distinct points in range.
    if best_pts is None:
        best_pts = [(x, y) for x in range(-2, 4) for y in range(-2, 3)][:30]

    seen = set()
    out = []
    for p in best_pts:
        x, y = p
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    # If somehow short (shouldn't happen), pad with fresh cells.
    if len(out) < 30:
        for c in [(x, y) for x in range(-30, 31) for y in range(-30, 31)]:
            if len(out) >= 30:
                break
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out[:30]
