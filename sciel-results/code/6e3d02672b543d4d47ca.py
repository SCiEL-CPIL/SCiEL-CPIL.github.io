"""Rewritten program: search for 30 points maximizing multiplicity of a doubled area."""
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples using NumPy."""
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Precompute cross-product matrix: cross[i,j] = xi*yj - xj*yi
    # Then area(i,j,k) = cross[i,j] + cross[j,k] + cross[k,i]
    X = xs[:, None]
    Y = ys[:, None]
    cross = X * Y.T - X.T * Y  # cross[i,j] = xi*yj - xj*yi
    
    # Generate all triples i<j<k
    triples = list(combinations(range(n), 3))
    i_idx = np.array([t[0] for t in triples], dtype=np.int32)
    j_idx = np.array([t[1] for t in triples], dtype=np.int32)
    k_idx = np.array([t[2] for t in triples], dtype=np.int32)
    
    areas = np.abs(cross[i_idx, j_idx] + cross[j_idx, k_idx] + cross[k_idx, i_idx])
    areas = areas[areas > 0]
    
    if len(areas) == 0:
        return 0
        
    vals, counts = np.unique(areas, return_counts=True)
    return int(np.max(counts))

def _local_search(points, max_iters=200, deadline=None):
    """Improve a set of points via local search: move one point to a better position."""
    points = list(points)
    n = len(points)
    best_score = _score(points)
    point_set = set(points)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try moving each point to a nearby random position
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            orig = points[idx]
            # Try several candidate positions with larger neighborhood
            for _ in range(12):
                nx = orig[0] + random.randint(-5, 5)
                ny = orig[1] + random.randint(-5, 5)
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    new_pt = (nx, ny)
                    if new_pt not in point_set:
                        points[idx] = new_pt
                        point_set.discard(orig)
                        point_set.add(new_pt)
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
                            point_set.discard(new_pt)
                            point_set.add(orig)
            if improved:
                break
        if not improved:
            # Try larger jumps
            for _ in range(3):
                if deadline and time.time() > deadline:
                    break
                idx = random.randint(0, n-1)
                orig = points[idx]
                for _ in range(10):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_pt = (nx, ny)
                    if new_pt not in point_set:
                        points[idx] = new_pt
                        point_set.discard(orig)
                        point_set.add(new_pt)
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
                            point_set.discard(new_pt)
                            point_set.add(orig)
                if improved:
                    break
            if not improved:
                break
    return points, best_score

def construct():
    # Strict deadline: 50 seconds to be safe within 90s limit
    deadline = time.time() + 50
    
    # Start with a grid-like configuration which often has good properties
    # 6x5 grid centered at 0
    best_points = [(x - 2, y - 2) for x in range(6) for y in range(5)]
    best_score = _score(best_points)
    
    random.seed(42)
    
    for restart in range(50): 
        if time.time() > deadline:
            break
        
        # Generate a starting configuration
        if restart == 0:
            pts = best_points[:]
        else:
            # Mix of structured and random starts
            if restart % 4 == 0:
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
            elif restart % 4 == 1:
                # Random points
                pts = []
                seen = set()
                while len(pts) < 30:
                    p = (random.randint(-30, 30), random.randint(-30, 30))
                    if p not in seen:
                        seen.add(p)
                        pts.append(p)
            elif restart % 4 == 2:
                # Structured lattice
                pts = []
                seen = set()
                xs = random.sample(range(-30, 31), 6)
                ys = random.sample(range(-30, 31), 5)
                for x in xs:
                    for y in ys:
                        pts.append((x, y))
                        seen.add((x, y))
            else:
                # Lattice-based start with basis vectors
                # Use a 2D lattice generated by two basis vectors
                ax = random.randint(2, 8)
                ay = random.randint(-2, 2)
                bx = random.randint(-2, 2)
                by = random.randint(2, 8)
                pts = []
                seen = set()
                for i in range(6):
                    for j in range(5):
                        px = i * ax + j * bx
                        py = i * ay + j * by
                        # Center and clamp
                        px = px - 2 * ax - 2 * bx
                        py = py - 2 * ay - 2 * by
                        px = max(-30, min(30, px))
                        py = max(-30, min(30, py))
                        if (px, py) not in seen:
                            seen.add((px, py))
                            pts.append((px, py))
                while len(pts) < 30:
                    p = (random.randint(-30, 30), random.randint(-30, 30))
                    if p not in seen:
                        seen.add(p)
                        pts.append(p)
            
            # Local search with more iterations
            pts, s = _local_search(pts, max_iters=150, deadline=deadline)
            if s > best_score:
                best_score = s
                best_points = pts
        
        # Periodically do a deeper local search on the best
        if restart % 4 == 0 and best_points:
            pts, s = _local_search(best_points[:], max_iters=200, deadline=deadline)
            if s > best_score:
                best_score = s
                best_points = pts
    
    # Final polish if time remains
    if time.time() < deadline - 3:
        pts, s = _local_search(best_points[:], max_iters=300, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = pts
    
    return best_points
