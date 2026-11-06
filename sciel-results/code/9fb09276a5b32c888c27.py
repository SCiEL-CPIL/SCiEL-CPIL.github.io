"""Optimized local search with target-area-aware moves."""
import time
import numpy as np
import random
from collections import Counter

_DEADLINE = 75.0
_MAX_AREA = 4000  # Safe upper bound for |det| in [-30,30]^2


def _init_areas(pts):
    """
    Initialize the areas array for all triples.
    Returns:
        areas: np.array of shape (n, n, n) where areas[i,j,k] is |det| for i<j<k, else 0.
        triples: list of (i,j,k) tuples for i<j<k.
    """
    n = len(pts)
    p = np.asarray(pts, dtype=np.int64)
    areas = np.zeros((n, n, n), dtype=np.int64)
    triples = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = p[i]
                x2, y2 = p[j]
                x3, y3 = p[k]
                det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                a = abs(det)
                areas[i, j, k] = a
                areas[j, k, i] = a
                areas[k, i, j] = a
                triples.append((i, j, k))
                
    return areas, triples


def _compute_score(pts):
    """Compute the maximum multiplicity of any positive doubled area."""
    n = len(pts)
    areas, triples = _init_areas(pts)
    return _get_max_multiplicity(areas, triples)


def _get_max_multiplicity(areas, triples):
    """Calculate max multiplicity from areas array."""
    if not triples:
        return 0
    # Extract only valid areas
    vals = []
    for i, j, k in triples:
        a = areas[i, j, k]
        if a > 0:
            vals.append(a)
    if not vals:
        return 0
    # Use numpy for speed
    arr = np.array(vals)
    vals_unique, counts = np.unique(arr, return_counts=True)
    return int(counts.max())


def _find_target_area(areas, triples):
    """Find the area with the highest multiplicity (the target area to maximize)."""
    if not triples:
        return 0, 0
    vals = []
    for i, j, k in triples:
        a = areas[i, j, k]
        if a > 0:
            vals.append(a)
    if not vals:
        return 0, 0
    arr = np.array(vals)
    vals_unique, counts = np.unique(arr, return_counts=True)
    best_idx = counts.argmax()
    return int(vals_unique[best_idx]), int(counts[best_idx])


def _local_refine_fast(pts, deadline):
    """
    Optimized local search with O(1) score evaluation via incremental area counts.
    """
    pts = [tuple(p) for p in pts]
    n = len(pts)
    
    # Precompute triples
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
                
    # Initialize areas and counts
    areas, _ = _init_areas(pts)
    area_counts = {}
    for i, j, k in triples:
        a = areas[i, j, k]
        if a > 0:
            area_counts[a] = area_counts.get(a, 0) + 1
            
    if not area_counts:
        return pts, 0
        
    cur_best = max(area_counts.values())
    
    # Build index: for each point, which triples involve it
    point_triples = [[] for _ in range(n)]
    for i, j, k in triples:
        point_triples[i].append((i, j, k))
        point_triples[j].append((i, j, k))
        point_triples[k].append((i, j, k))
    
    improved = True
    no_improve_count = 0
    radius = 4
    pts_set = set(pts)
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        
        # Adjust radius if stuck
        if no_improve_count > 5:
            radius = 8
        elif no_improve_count > 10:
            radius = 15
        elif no_improve_count > 15:
            radius = 30 # Global search step
            
        for idx in range(n):
            if time.time() >= deadline:
                break
            
            ox, oy = pts[idx]
            best_move = None
            best_val = cur_best
            best_new_counts = None
            
            # Generate candidates
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        # Check collision
                        if (nx, ny) in pts_set:
                            continue
                        candidates.append((nx, ny))
                        
            if not candidates:
                continue
                
            # If too many candidates, sample
            if len(candidates) > 200:
                candidates = random.sample(candidates, 200)
                
            for cand in candidates:
                nx, ny = cand
                new_counts = area_counts.copy()
                
                # Update counts for triples involving idx
                for i, j, k in point_triples[idx]:
                    old_a = areas[i, j, k]
                    if old_a > 0:
                        new_counts[old_a] -= 1
                        if new_counts[old_a] == 0:
                            del new_counts[old_a]
                    
                    # Calculate new area
                    others = [x for x in (i, j, k) if x != idx]
                    p2 = pts[others[0]]
                    p3 = pts[others[1]]
                    
                    x1, y1 = nx, ny
                    x2, y2 = p2
                    x3, y3 = p3
                    
                    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    new_a = abs(det)
                    
                    if new_a > 0:
                        new_counts[new_a] = new_counts.get(new_a, 0) + 1
                        
                    # Store for rollback if we accept? No, we just need to check score.
                    # But we need to update `areas` if we accept.
                    # For now, just track the change in areas for this specific triple if we accept.
                    # Actually, we can just recalculate areas for idx if we accept.
                    
                if new_counts:
                    val = max(new_counts.values())
                    if val > best_val:
                        best_val = val
                        best_move = cand
                        best_new_counts = new_counts # Store reference? No, we need to apply it.
                        # We don't store best_new_counts directly, we just note that this move is good.
                        # We will re-apply the move and update areas/counts properly if accepted.
            
            if best_move:
                # Apply the best move
                pts[idx] = best_move
                pts_set.remove((ox, oy))
                pts_set.add(best_move)
                
                # Update areas and counts
                for i, j, k in point_triples[idx]:
                    old_a = areas[i, j, k]
                    if old_a > 0:
                        area_counts[old_a] -= 1
                        if area_counts[old_a] == 0:
                            del area_counts[old_a]
                            
                    nx, ny = best_move
                    others = [x for x in (i, j, k) if x != idx]
                    p2 = pts[others[0]]
                    p3 = pts[others[1]]
                    
                    x1, y1 = nx, ny
                    x2, y2 = p2
                    x3, y3 = p3
                    
                    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    new_a = abs(det)
                    
                    areas[i, j, k] = new_a
                    areas[j, k, i] = new_a
                    areas[k, i, j] = new_a
                    
                    if new_a > 0:
                        area_counts[new_a] = area_counts.get(new_a, 0) + 1
                        
                cur_best = best_val
                improved = True
                no_improve_count = 0
                break  # Restart the loop over points
    
    return pts, cur_best


def _grid_seed():
    """6x5 grid baseline."""
    return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]


def _two_line_seed():
    """Two parallel lines."""
    return [(x, 0) for x in range(-7, 8)] + [(x, 1) for x in range(-7, 8)]


def _hex_seed():
    """Hexagonal-like arrangement."""
    pts = []
    pts.append((0, 0))
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
        pts.append((dx*2, dy*2))
    for dx, dy in [(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2),(1,2),(-1,2),(1,-2),(-1,-2)]:
        pts.append((dx*2, dy*2))
    for x in range(-30, 31):
        for y in range(-30, 31):
            if len(pts) >= 30:
                break
            if (x, y) not in pts:
                pts.append((x, y))
        if len(pts) >= 30:
            break
    return pts[:30]


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    
    best_pts = None
    best_val = 0
    
    seeds = [
        _grid_seed(),
        _two_line_seed(),
        _hex_seed(),
    ]
    
    # Add some random seeds
    rng = random.Random(42)
    for _ in range(3):
        random_seed = []
        used = set()
        while len(random_seed) < 30:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in used:
                used.add((x, y))
                random_seed.append((x, y))
        seeds.append(random_seed)
        
    for seed_pts in seeds:
        if time.time() >= deadline - 5:
            break
        if len(seed_pts) != 30:
            continue
            
        # Allocate time based on remaining budget
        # Give more time to earlier seeds if they are promising?
        # For now, distribute evenly.
        remaining_time = deadline - time.time()
        time_per_seed = remaining_time / max(1, len(seeds))
        seed_deadline = time.time() + time_per_seed
        
        pts, val = _local_refine_fast(seed_pts, seed_deadline)
        if val > best_val:
            best_val = val
            best_pts = pts
    
    # Final validation and padding
    if best_pts is None or best_val == 0:
        best_pts = _grid_seed()
        # Calculate val for fallback
        areas, triples = _init_areas(best_pts)
        best_val = _get_max_multiplicity(areas, triples)
        
    seen = set()
    out = []
    for p in best_pts:
        x, y = p
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    if len(out) < 30:
        for x in range(-30, 31):
            for y in range(-30, 31):
                if len(out) >= 30:
                    break
                if (x, y) not in seen:
                    seen.add((x, y))
                    out.append((x, y))
            if len(out) >= 30:
                break
    
    return out[:30]
