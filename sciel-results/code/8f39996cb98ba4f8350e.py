import time
import random
import numpy as np
from collections import defaultdict
from math import gcd

_START = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    a = y1 - y2
    b = x2 - x1
    c = -(a * x1 + b * y1)
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a //= g; b //= g; c //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a; b = -b; c = -c
    return (a, b, c)

def _score_fast(pts):
    """
    Score a configuration of 16 points.
    Returns (num_lines_with_exactly_4, has_line_with_5_or_more)
    Uses numpy for vectorized line counting where possible, 
    but fallback to pure python if numpy overhead is too high for small N.
    Actually, for N=16, pure python with optimized loops is often competitive,
    but let's try a hybrid or just a very tight python loop.
    """
    n = len(pts)
    # Check distinctness and bounds
    if len(set(map(tuple, pts))) != 16:
        return -1, True
    for x, y in pts:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1, True
            
    line_pts = defaultdict(set)
    # Pre-compute pairs to avoid redundant work
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            k = _line_key((xi, yi), (xj, yj))
            line_pts[k].add(i)
            line_pts[k].add(j)
            
    cnt4 = 0
    has5 = False
    for k, s in line_pts.items():
        m = len(s)
        if m == 4:
            cnt4 += 1
        elif m >= 5:
            has5 = True
            # Early exit if we just care about validity, but we need score.
            # However, if has5 is true, the configuration is invalid for the task.
            # The task says "no five collinear". So if has5, score is effectively 0 or invalid.
            # We return -1 for invalid in _score wrapper.
            pass
    return cnt4, has5

def _score(pts):
    c, has5 = _score_fast(pts)
    if has5:
        return -1
    return c

def _generate_grid(m, n, sx, sy, ox, oy, rng):
    """
    Generate an m x n grid of points.
    Total points = m*n. We need exactly 16.
    So we look for m*n >= 16, pick 16 points from the grid structure.
    Or simpler: just generate m*n points if m*n==16.
    Since 16 = 4*4 or 2*8 or 1*16.
    Let's stick to 4x4 grids but with different spacings and offsets.
    """
    if m * n != 16:
        return None
    pts = []
    for i in range(m):
        for j in range(n):
            x = ox + i * sx
            y = oy + j * sy
            if abs(x) > 1000 or abs(y) > 1000:
                return None
            pts.append([x, y])
    return pts

def _random_affine_grid(rng):
    """
    Generate a random 4x4 grid with random spacing and offset.
    This preserves the "grid-like" structure which often yields high collinearity.
    """
    # Random spacings between 1 and 20
    sx = rng.randint(1, 20)
    sy = rng.randint(1, 20)
    # Random offset
    ox = rng.randint(-500, 500)
    oy = rng.randint(-500, 500)
    
    pts = []
    for i in range(4):
        for j in range(4):
            x = ox + i * sx
            y = oy + j * sy
            if abs(x) > 1000 or abs(y) > 1000:
                # Try to adjust offset or return None
                return None
            pts.append([x, y])
    
    # Ensure distinct (grid ensures this if sx, sy != 0)
    if len(set(map(tuple, pts))) != 16:
        return None
    return pts

def _perturb_grid(pts, rng, strength=5):
    """
    Take a grid-like structure and perturb individual points slightly.
    This allows escaping the rigid grid while maintaining local structure.
    """
    new_pts = [list(p) for p in pts]
    # Pick a few points to move
    k = rng.randint(1, 3)
    indices = rng.sample(range(16), k)
    for idx in indices:
        # Move to a nearby integer point
        dx = rng.randint(-strength, strength)
        dy = rng.randint(-strength, strength)
        nx = new_pts[idx][0] + dx
        ny = new_pts[idx][1] + dy
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        new_pts[idx] = [nx, ny]
        
    if len(set(map(tuple, new_pts))) != 16:
        return None
    return new_pts

def _simulated_annealing(start_pts, rng, max_iters, deadline):
    cur = [list(p) for p in start_pts]
    cur_s = _score(cur)
    if cur_s < 0:
        # Invalid start, try to fix or restart
        cur = _random_affine_grid(rng)
        cur_s = _score(cur)
        if cur_s < 0:
            return None, -1
            
    best = [list(p) for p in cur]
    best_s = cur_s
    T = 5.0  # Initial temperature
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        r = rng.random()
        if r < 0.6:
            # Perturb existing points
            nb = _perturb_grid(cur, rng, strength=rng.randint(1, 10))
        elif r < 0.8:
            # Move one point to a random location within bounds
            nb = [list(p) for p in cur]
            idx = rng.randrange(16)
            nb[idx] = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
            if len(set(map(tuple, nb))) != 16:
                continue
        else:
            # Restart with a new random grid (helps escape deep local minima)
            nb = _random_affine_grid(rng)
            if nb is None:
                continue
        
        if nb is None:
            continue
            
        nb_s = _score(nb)
        if nb_s < 0:
            # Invalid (5 collinear or out of bounds or duplicate)
            # Penalize heavily
            delta = -10 - cur_s
        else:
            delta = nb_s - cur_s
            
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-6)):
            cur = nb
            cur_s = nb_s
            if cur_s > best_s:
                best = [list(p) for p in cur]
                best_s = cur_s
                
        T *= 0.999  # Cooling schedule
        if T < 0.01:
            T = 0.01 # Floor temperature
            
    return best, best_s

def construct():
    deadline = _START + _DEADLINE
    rng = random.Random(42)
    np.random.seed(42)
    
    best_pts = None
    best_score = -1
    
    # Phase 1: Explore random affine grids
    # This is much more structured than random points.
    # A 4x4 grid has 10 lines with 4 points (4 rows, 4 cols, 2 diags).
    # We want to find configurations with > 10.
    
    # Try a few specific known good structures if we can identify them,
    # but random affine grids are a good baseline.
    
    num_starts = 50
    iters_per_start = 20000
    
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        # Start with a random affine grid
        start = _random_affine_grid(rng)
        if start is None:
            # Fallback to simple grid
            start = [[i, j] for i in range(4) for j in range(4)]
            
        # Run SA
        b, s = _simulated_annealing(start, rng, iters_per_start, deadline - 5)
        
        if s > best_score:
            best_score = s
            best_pts = b
            
        # Occasionally restart SA from the current best to refine
        if i % 10 == 0 and best_pts is not None:
            b2, s2 = _simulated_annealing(best_pts, rng, iters_per_start // 2, deadline - 5)
            if s2 > best_score:
                best_score = s2
                best_pts = b2

    # Fallback if no valid points found (shouldn't happen)
    if best_pts is None:
        best_pts = [[i, j] for i in range(4) for j in range(4)]
        
    # Final validation
    if _score(best_pts) < 0:
        # Return a known valid grid
        return [[i, j] for i in range(4) for j in range(4)]
        
    return best_pts

# Helper for testing
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(pts)
