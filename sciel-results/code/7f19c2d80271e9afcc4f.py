"""Structural change: combinatorial selection from symmetric lattice base."""
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def _doubled_areas(points):
    """Compute absolute doubled areas for all triples."""
    n = len(points)
    if n < 3:
        return []
    p = np.array(points, dtype=np.int64)
    # Vectorized: for each triple (i,j,k), area2 = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # Use broadcasting: p[i] cross p[j] cross p[k] style
    # Efficient: for each pair (i,j), compute cross with all k
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            # cross product of (p[j]-p[i]) with (p[k]-p[i]) for all k
            d = p[j] - p[i]
            rel = p - p[i]
            cross = rel[:, 0] * d[1] - rel[:, 1] * d[0]
            # exclude i and j themselves (they give 0)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            vals = np.abs(cross[mask])
            vals = vals[vals > 0]
            if len(vals) > 0:
                areas.extend(vals.tolist())
    return areas

def _score(points):
    """Return the max multiplicity of a single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    c = Counter(areas)
    return c.most_common(1)[0][1]

def _generate_hex_pool(bound=30):
    """Generate a hexagonal/triangular lattice pool within [-bound, bound]^2."""
    pts = set()
    # Triangular lattice: basis vectors (1,0) and (0.5, sqrt(3)/2)
    # Use integer approximations: (2,0) and (1, sqrt(3)) -> scale to integers
    # Simpler: use a dense grid with hexagonal offset
    # Row-based hexagonal packing
    for y in range(-bound, bound + 1):
        offset = (y % 2)  # 0 or 1
        for x in range(-bound, bound + 1):
            # Shift every other row by 0.5 in x -> use 2x coordinates
            px = 2 * x + offset
            py = 2 * y
            if -bound <= px <= bound and -bound <= py <= bound:
                pts.add((px, py))
    return list(pts)

def _generate_rect_pool(bound=30):
    """Dense rectangular grid."""
    return [(x, y) for x in range(-bound, bound + 1) for y in range(-bound, bound + 1)]

def _generate_mixed_pool(bound=30):
    """Mix of structured patterns."""
    pts = set()
    # Rectangular grid
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            pts.add((x, y))
    return list(pts)

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Generate candidate pool
    pool = _generate_hex_pool(30)
    if len(pool) < 30:
        pool = _generate_rect_pool(30)
    
    pool = list(set(pool))  # deduplicate
    random.shuffle(pool)
    
    # Initial selection: pick 30 points from the pool
    # Prefer points that form a dense cluster for high multiplicity
    # Start with a small rectangular block from the pool
    best_points = None
    best_score = 0
    
    # Try several initial configurations
    initial_configs = []
    
    # Config 1: first 30 from shuffled pool
    if len(pool) >= 30:
        initial_configs.append(pool[:30])
    
    # Config 2: try to find a 6x5 rectangular block
    for start_x in range(-20, 21, 5):
        for start_y in range(-20, 21, 5):
            block = []
            for dx in range(6):
                for dy in range(5):
                    pt = (start_x + dx, start_y + dy)
                    if pt in set(pool):
                        block.append(pt)
            if len(block) == 30:
                initial_configs.append(block)
                break
        if len(initial_configs) > 1:
            break
    
    # Config 3: hexagonal cluster
    hex_pts = []
    for y in range(-3, 4):
        offset = (y % 2)
        for x in range(-3, 4):
            pt = (2 * x + offset, 2 * y)
            if -30 <= pt[0] <= 30 and -30 <= pt[1] <= 30:
                hex_pts.append(pt)
    if len(hex_pts) >= 30:
        initial_configs.append(hex_pts[:30])
    
    if not initial_configs:
        # Fallback: random 30 from pool
        initial_configs.append(random.sample(pool, min(30, len(pool))))
    
    # Evaluate initial configs
    for cfg in initial_configs:
        if len(cfg) != 30:
            continue
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = list(cfg)
    
    if best_points is None:
        # Last resort
        best_points = pool[:30] if len(pool) >= 30 else pool + [(0,0)] * (30 - len(pool))
        best_score = _score(best_points)
    
    # Local search: swap-based improvement
    # Represent current set as a set for O(1) lookup
    current = list(best_points)
    current_set = set(current)
    cur_score = best_score
    
    # Precompute pool as list for random selection
    pool_list = pool
    pool_set = set(pool_list)
    
    iterations = 0
    while time.time() - t0 < DEADLINE:
        # Pick a random point in current to swap out
        idx = random.randrange(30)
        old_pt = current[idx]
        
        # Pick a random candidate from pool (not in current)
        # Try a few candidates
        improved = False
        for _ in range(20):
            new_pt = random.choice(pool_list)
            if new_pt in current_set:
                continue
            
            # Tentatively swap
            current[idx] = new_pt
            current_set.discard(old_pt)
            current_set.add(new_pt)
            
            s = _score(current)
            
            if s > cur_score:
                cur_score = s
                best_points = list(current)
                best_score = s
                improved = True
                break
            else:
                # Revert
                current[idx] = old_pt
                current_set.discard(new_pt)
                current_set.add(old_pt)
        
        if not improved:
            # Try a different strategy: random restart of the swap target
            pass
        
        iterations += 1
        if iterations % 100 == 0:
            if time.time() - t0 > DEADLINE - 5:
                break
    
    # Final validation
    result = best_points
    # Ensure exactly 30 distinct integer points in [-30,30]^2
    result = list(set(result))
    if len(result) < 30:
        # Fill with random pool points
        for pt in pool:
            if len(result) >= 30:
                break
            if pt not in set(result):
                result.append(pt)
    result = result[:30]
    
    # Validate bounds
    result = [(x, y) for x, y in result if -30 <= x <= 30 and -30 <= y <= 30]
    if len(result) < 30:
        # Fallback to simple grid
        result = [(x, y) for x in range(6) for y in range(5)]
    
    return result
