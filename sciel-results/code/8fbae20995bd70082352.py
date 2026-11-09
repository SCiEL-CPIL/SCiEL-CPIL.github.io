"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

# EVOLVE-BLOCK-START

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    # Vectorized area computation
    # For each triple (i,j,k), doubled area = |det([p_j - p_i, p_k - p_i])|
    # We'll compute all pairwise differences and then determinants
    # Using numpy for speed
    
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # Compute all pairs of differences
    # diff[i,j] = pts[j] - pts[i]
    diff = pts[None, :, :] - pts[:, None, :]  # shape (n, n, 2)
    
    # For triple (i, j, k): det = diff[i,j,0]*diff[i,k,1] - diff[i,j,1]*diff[i,k,0]
    # We need to compute this for all i<j<k
    # det[i,j,k] = (pts[j]-pts[i]) x (pts[k]-pts[i])
    # = (pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0])
    
    # More efficient: compute det matrix for all pairs
    # det[i,j] = pts[i,0]*pts[j,1] - pts[i,1]*pts[j,0]
    det_matrix = pts[:, 0] * pts[1:, 1] - pts[:, 1] * pts[1:, 0]  # This isn't quite right
    
    # Let me use a different approach
    # For each pair (i,j), compute cross product with all k
    # Actually, let's just compute all triple determinants directly but efficiently
    
    # det[i,j,k] = (p_j - p_i) x (p_k - p_i)
    # = p_j x p_k - p_j x p_i - p_i x p_k
    # where a x b = a[0]*b[1] - a[1]*b[0]
    
    # Let cx[i,j] = pts[i,0]*pts[j,1] - pts[i,1]*pts[j,0]
    cx = np.zeros((n, n), dtype=np.int64)
    cx = pts[:, 0:1] * pts[:, 1:2].T - pts[:, 1:2] * pts[:, 0:1].T
    
    # det[i,j,k] = cx[i,j] - cx[i,k] - cx[j,i]  ... wait let me re-derive
    # (p_j - p_i) x (p_k - p_i) = p_j x p_k - p_j x p_i - p_i x p_k + p_i x p_i
    # = cx[j,k] - cx[j,i] - cx[i,k] + 0
    # = cx[j,k] + cx[i,j] - cx[i,k]
    
    # Actually: (a-b) x (c-b) = a x c - a x b - b x c + b x b = a x c - a x b - b x c
    # = cx[a,c] - cx[a,b] - cx[b,c]
    # So with a=j, b=i, c=k: cx[j,k] - cx[j,i] - cx[i,k]
    
    # We want |det| for all i<j<k
    # Let's build the full det array
    det_full = np.zeros((n, n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if i == k or j == k:
                    continue
                det_full[i, j, k] = cx[j, k] - cx[j, i] - cx[i, k]
    
    # Now get all triples i<j<k
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = abs(det_full[i, j, k])
                if a > 0:
                    areas.append(a)
    
    if not areas:
        return 0
    
    cnt = Counter(areas)
    return max(cnt.values())


def _score_fast(points):
    """Faster scoring using numpy vectorization."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # cx[i,j] = pts[i,0]*pts[j,1] - pts[i,1]*pts[j,0]
    cx = pts[:, 0:1] * pts[:, 1:2].T - pts[:, 1:2] * pts[:, 0:1].T
    
    # det[i,j,k] = cx[j,k] - cx[j,i] - cx[i,k] for i<j<k
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = abs(int(cx[j, k] - cx[j, i] - cx[i, k]))
                if a > 0:
                    areas.append(a)
    
    if not areas:
        return 0
    
    cnt = Counter(areas)
    return max(cnt.values())


def _score_vec(points):
    """Vectorized scoring."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    cx = pts[:, 0:1] * pts[:, 1:2].T - pts[:, 1:2] * pts[:, 0:1].T
    
    # Build list of areas for all i<j<k
    areas = []
    for i in range(n-2):
        for j in range(i+1, n-1):
            # det[i,j,k] = cx[j,k] - cx[j,i] - cx[i,k]
            # For fixed i,j, vary k from j+1 to n-1
            det_vals = cx[j, j+1:] - cx[j, i] - cx[i, j+1:]
            abs_vals = np.abs(det_vals)
            # Filter out zeros
            nonzero = abs_vals[abs_vals > 0]
            if len(nonzero) > 0:
                areas.append(nonzero)
    
    if not areas:
        return 0
    
    all_areas = np.concatenate(areas)
    if len(all_areas) == 0:
        return 0
    
    # Find most common value
    vals, counts = np.unique(all_areas, return_counts=True)
    return int(np.max(counts))


def _neighborhood_move(points, rng):
    """Try moving one point to a nearby position."""
    pts = list(points)
    i = rng.randrange(len(pts))
    x, y = pts[i]
    dx = rng.randint(-5, 5)
    dy = rng.randint(-5, 5)
    nx, ny = x + dx, y + dy
    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
        return None
    if (nx, ny) in pts:
        return None
    new_pts = pts[:]
    new_pts[i] = (nx, ny)
    return new_pts


def _swap_move(points, rng):
    """Try swapping two points with nearby positions."""
    pts = list(points)
    i, j = rng.sample(range(len(pts)), 2)
    x1, y1 = pts[i]
    x2, y2 = pts[j]
    dx1 = rng.randint(-5, 5)
    dy1 = rng.randint(-5, 5)
    dx2 = rng.randint(-5, 5)
    dy2 = rng.randint(-5, 5)
    nx1, ny1 = x1 + dx1, y1 + dy1
    nx2, ny2 = x2 + dx2, y2 + dy2
    if nx1 < -30 or nx1 > 30 or ny1 < -30 or ny1 > 30:
        return None
    if nx2 < -30 or nx2 > 30 or ny2 < -30 or ny2 > 30:
        return None
    # Check no duplicates
    new_pts = pts[:]
    new_pts[i] = (nx1, ny1)
    new_pts[j] = (nx2, ny2)
    if len(set(new_pts)) != len(new_pts):
        return None
    return new_pts


def _random_point(rng):
    return (rng.randint(-30, 30), rng.randint(-30, 30))


def _random_config(rng):
    """Generate a random valid configuration of 30 distinct points."""
    pts = set()
    while len(pts) < 30:
        pts.add(_random_point(rng))
    return list(pts)


def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    rng = random.Random(42)
    
    # Start with the seed configuration
    best_pts = [(x, y) for x in range(6) for y in range(5)]
    best_score = _score_vec(best_pts)
    
    # Also try some structured initial configs
    # Grid-like patterns
    initial_configs = []
    
    # Original seed
    initial_configs.append([(x, y) for x in range(6) for y in range(5)])
    
    # Dense 5x6 grid
    initial_configs.append([(x, y) for x in range(5) for y in range(6)])
    
    # Square-ish
    initial_configs.append([(x, y) for x in range(5) for y in range(6)][:30])
    
    # Random
    for _ in range(5):
        initial_configs.append(_random_config(rng))
    
    for cfg in initial_configs:
        if len(set(cfg)) != 30:
            continue
        if any(x < -30 or x > 30 or y < -30 or y > 30 for x, y in cfg):
            continue
        s = _score_vec(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg[:]
    
    # Local search with neighborhood moves and swaps
    current_pts = best_pts[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Choose move type
        if rng.random() < 0.5:
            candidate = _neighborhood_move(current_pts, rng)
        else:
            candidate = _swap_move(current_pts, rng)
        
        if candidate is None:
            no_improve_count += 1
            continue
        
        new_score = _score_vec(candidate)
        
        if new_score > current_score:
            current_pts = candidate
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = candidate[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Restart from best or random
            if rng.random() < 0.5:
                current_pts = best_pts[:]
                current_score = best_score
            else:
                current_pts = _random_config(rng)
                current_score = _score_vec(current_pts)
            no_improve_count = 0
    
    return best_pts

# EVOLVE-BLOCK-END
