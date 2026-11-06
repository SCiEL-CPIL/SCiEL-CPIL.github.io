"""Rewritten program: search for 30 points maximizing multiplicity of a doubled area."""
import time
import random
import numpy as np
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples using NumPy."""
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # Compute all pairwise differences? No, direct triple loop vectorized is better for n=30.
    # i < j < k
    # We can generate indices and compute determinants in a vectorized block.
    # However, for n=30, even vectorized Python overhead might be high if done naively in loops.
    # Let's use a precomputed index list or a fast C-loop equivalent via numpy.
    
    # Method: Generate all triples (i,j,k) with i<j<k
    # There are 4060 triples.
    # We can compute areas for all triples at once using broadcasting if we reshape, 
    # but 30x30x30 is 27000 points, which is fine for memory but we need unique triples.
    
    # Faster approach:
    # Create arrays for x and y coordinates
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # We need to calculate abs(x1(y2-y3) + x2(y3-y1) + x3(y1-y2)) for all i<j<k.
    # Let's create index arrays
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    # Extract indices
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    # Compute areas
    x1, y1 = xs[i_flat], ys[i_flat]
    x2, y2 = xs[j_flat], ys[j_flat]
    x3, y3 = xs[k_flat], ys[k_flat]
    
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    areas = areas[areas > 0]
    
    if len(areas) == 0:
        return 0
        
    # Find max multiplicity
    # np.unique with return_counts is efficient
    vals, counts = np.unique(areas, return_counts=True)
    return int(np.max(counts))

def _local_search(points, max_iters=200, deadline=None):
    """Improve a set of points via local search: move one point to a better position."""
    points = list(points)
    n = len(points)
    best_score = _score(points)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try moving each point to a nearby random position
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            orig = points[idx]
            # Try several candidate positions
            for _ in range(5): # Reduced from 8
                nx = orig[0] + random.randint(-3, 3)
                ny = orig[1] + random.randint(-3, 3)
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    new_pt = (nx, ny)
                    if new_pt not in points:
                        points[idx] = new_pt
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
            if improved:
                break
        if not improved:
            # Try larger jumps
            for _ in range(2): # Reduced from 3
                if deadline and time.time() > deadline:
                    break
                idx = random.randint(0, n-1)
                orig = points[idx]
                for _ in range(8): # Reduced from 12
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_pt = (nx, ny)
                    if new_pt not in points:
                        points[idx] = new_pt
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
                if improved:
                    break
            if not improved:
                break
    return points, best_score

def construct():
    # Strict deadline: 45 seconds to be safe within 60s limit
    deadline = time.time() + 45
    
    # Start with a grid-like configuration which often has good properties
    # 6x5 grid centered at 0
    best_points = [(x - 2, y - 2) for x in range(6) for y in range(5)]
    best_score = _score(best_points)
    
    random.seed(42)
    
    # Reduce restarts to fit in time
    for restart in range(30): 
        if time.time() > deadline:
            break
        
        # Generate a starting configuration
        if restart == 0:
            pts = best_points[:]
        else:
            # Mix of structured and random starts
            if restart % 3 == 0:
                # Grid with perturbation
                pts = []
                for x in range(6):
                    for y in range(5):
                        px = (x - 2) + random.randint(-1, 1)
                        py = (y - 2) + random.randint(-1, 1)
                        px = max(-30, min(30, px))
                        py = max(-30, min(30, py))
                        pts.append((px, py))
                seen = set()
                clean = []
                for p in pts:
                    if p not in seen:
                        seen.add(p)
                        clean.append(p)
                while len(clean) < 30:
                    p = (random.randint(-30, 30), random.randint(-30, 30))
                    if p not in seen:
                        seen.add(p)
                        clean.append(p)
                pts = clean
            elif restart % 3 == 1:
                # Random points
                pts = []
                seen = set()
                while len(pts) < 30:
                    p = (random.randint(-30, 30), random.randint(-30, 30))
                    if p not in seen:
                        seen.add(p)
                        pts.append(p)
            else:
                # Structured lattice
                pts = []
                seen = set()
                xs = random.sample(range(-30, 31), 6)
                ys = random.sample(range(-30, 31), 5)
                for x in xs:
                    for y in ys:
                        pts.append((x, y))
                        seen.add((x, y))
            
            # Local search with reduced iterations
            pts, s = _local_search(pts, max_iters=100, deadline=deadline)
            if s > best_score:
                best_score = s
                best_points = pts
        
        # Periodically do a deeper local search on the best
        if restart % 5 == 0 and best_points:
            pts, s = _local_search(best_points[:], max_iters=150, deadline=deadline)
            if s > best_score:
                best_score = s
                best_points = pts
    
    # Final polish if time remains
    if time.time() < deadline - 2:
        pts, s = _local_search(best_points[:], max_iters=200, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = pts
    
    return best_points
