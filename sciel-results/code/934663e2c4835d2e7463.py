"""Improved program: local search with incremental area counting to maximize max multiplicity of doubled triangle area."""
import random
import time
import numpy as np
from collections import Counter
from itertools import combinations

def _precompute_triples(n=30):
    """Precompute all triple indices and for each point, which triples it belongs to."""
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    # For each point, store list of (triple_index, other_two_points)
    point_triples = [[] for _ in range(n)]
    for t_idx, (i, j, k) in enumerate(triples):
        point_triples[i].append((t_idx, j, k))
        point_triples[j].append((t_idx, i, k))
        point_triples[k].append((t_idx, i, j))
    return triples, point_triples

_TRIPLES, _POINT_TRIPLES = _precompute_triples(30)

def _compute_areas_and_counts(points):
    """
    Compute all doubled triangle areas and return (max_count, counts_dict).
    Uses vectorized numpy for speed.
    """
    n = len(points)
    if n < 3:
        return 0, {}
    pts = np.array(points, dtype=np.int64)
    x = pts[:, 0]
    y = pts[:, 1]
    
    i_idx = np.array([t[0] for t in _TRIPLES])
    j_idx = np.array([t[1] for t in _TRIPLES])
    k_idx = np.array([t[2] for t in _TRIPLES])
    
    x1 = x[i_idx]
    y1 = y[i_idx]
    x2 = x[j_idx]
    y2 = y[j_idx]
    x3 = x[k_idx]
    y3 = y[k_idx]
    
    dets = np.abs(x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2))
    
    valid_mask = dets > 0
    valid_dets = dets[valid_mask]
    
    if len(valid_dets) == 0:
        return 0, {}
        
    counts = Counter(valid_dets.tolist())
    max_count = max(counts.values())
    
    return max_count, counts

def _update_counts_incremental(points, idx, old_pos, new_pos, counts):
    """
    Update the counts dict when point idx moves from old_pos to new_pos.
    Returns new max_count.
    """
    n = len(points)
    
    # Get coordinates of all points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    # Temporarily set new position for calculation
    xs[idx] = new_x
    ys[idx] = new_y
    
    # Get the triples involving idx
    for t_idx, p1, p2 in _POINT_TRIPLES[idx]:
        # Determine which of p1, p2 are the other two points
        # The triple is (idx, p1, p2) in some order, but we just need coordinates
        x1, y1 = xs[p1], ys[p1]
        x2, y2 = xs[p2], ys[p2]
        xi, yi = xs[idx], ys[idx]
        
        # New area
        area_new = abs(xi * (y1 - y2) - x1 * (yi - y2) + x2 * (yi - y1))
        
        # Old area (need to compute with old position)
        # We need to temporarily restore old position for calculation
        # Actually, let's compute old area separately
        xi_old, yi_old = old_x, old_y
        area_old = abs(xi_old * (y1 - y2) - x1 * (yi_old - y2) + x2 * (yi_old - y1))
        
        # Update counts
        if area_old > 0:
            counts[area_old] -= 1
            if counts[area_old] == 0:
                del counts[area_old]
        
        if area_new > 0:
            if area_new in counts:
                counts[area_new] += 1
            else:
                counts[area_new] = 1
    
    if not counts:
        return 0
        
    return max(counts.values())

def construct():
    # Set deadline to 80s to be safe against the 90s kill
    deadline = time.time() + 80
    
    # Start with a reasonable initial configuration
    # Use a grid-like structure that might produce repeated areas
    def _initial_points():
        # 6x5 grid centered at origin
        pts = [(x - 2, y - 2) for x in range(6) for y in range(5)]
        return pts
    
    points = _initial_points()
    best_score, best_counts = _compute_areas_and_counts(points)
    best_points = points[:]
    
    # Local search: randomly perturb one point, keep if improves
    rng = random.Random(42)
    
    def _random_point():
        return (rng.randint(-30, 30), rng.randint(-30, 30))
    
    def _valid(points):
        """Check all points are distinct and in range."""
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
            if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
                return False
        return True
    
    # Simulated annealing style local search
    current_score = best_score
    current_counts = best_counts.copy()
    current_points = best_points[:]
    temp = 5.0
    cooling = 0.9995
    
    iteration = 0
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = rng.randint(0, 29)
        old_pt = current_points[idx]
        
        # Try a few random new positions
        found_improvement = False
        for _ in range(3):
            new_pt = _random_point()
            
            # Check validity
            if new_pt == old_pt:
                continue
            if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                continue
            # Check distinctness
            if new_pt in current_points:
                continue
            
            # Calculate new score incrementally
            new_score = _update_counts_incremental(
                current_points, idx, old_pt, new_pt, current_counts
            )
            
            if new_score > current_score:
                # Accept improvement
                current_points[idx] = new_pt
                current_score = new_score
                found_improvement = True
                if new_score > best_score:
                    best_score = new_score
                    best_points = current_points[:]
                    best_counts = current_counts.copy()
                break
            elif new_score == current_score:
                # Accept with some probability (annealing)
                if rng.random() < 0.3:
                    current_points[idx] = new_pt
                    break
            else:
                # Accept worse moves with probability based on temperature
                if rng.random() < temp * 0.1:
                    current_points[idx] = new_pt
                    current_score = new_score
                    break
        
        temp *= cooling
        if temp < 0.01:
            # Reset temperature and restart from best
            temp = 5.0
            current_points = best_points[:]
            current_score = best_score
            current_counts = best_counts.copy()
        
        iteration += 1
        # Check time periodically
        if iteration % 100 == 0 and time.time() > deadline - 5:
            break

    return best_points
