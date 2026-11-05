"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def _precompute_triples(n):
    """Precompute all triples and mapping from point index to triple indices."""
    triples = list(combinations(range(n), 3))
    triple_map = {i: [] for i in range(n)}
    for t_idx, (i, j, k) in enumerate(triples):
        triple_map[i].append(t_idx)
        triple_map[j].append(t_idx)
        triple_map[k].append(t_idx)
    return triples, triple_map

def _calc_area(p1, p2, p3):
    """Calculate absolute doubled area of triangle formed by p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2))

def _score_config(points):
    """Return the maximum multiplicity of any positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    triples, _ = _precompute_triples(n)
    areas = []
    for i, j, k in triples:
        a = _calc_area(points[i], points[j], points[k])
        if a > 0:
            areas.append(a)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _local_search(points, max_iter=10000, time_limit=45):
    """Local search using incremental updates to maximize max multiplicity."""
    start_time = time.time()
    n = len(points)
    pts = list(points)
    triples, triple_map = _precompute_triples(n)
    
    # Initial areas and counts
    areas = [0] * len(triples)
    for t_idx, (i, j, k) in enumerate(triples):
        areas[t_idx] = _calc_area(pts[i], pts[j], pts[k])
    
    counts = Counter(a for a in areas if a > 0)
    current_best = max(counts.values()) if counts else 0
    
    best_points = pts.copy()
    best_score = current_best
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = pts[idx]
        
        # Generate candidate moves
        # 1. Random
        nx = random.randint(-30, 30)
        ny = random.randint(-30, 30)
        
        # Check distinctness
        is_distinct = True
        for i, (px, py) in enumerate(pts):
            if i != idx and (px, py) == (nx, ny):
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Store old areas for affected triples
        t_indices = triple_map[idx]
        old_areas = [areas[t_idx] for t_idx in t_indices]
        
        # Apply move tentatively
        pts[idx] = (nx, ny)
        
        # Calculate new areas and update counts
        temp_counts = counts.copy()
        new_areas = [0] * len(t_indices)
        
        for m, t_idx in enumerate(t_indices):
            i, j, k = triples[t_idx]
            old_area = old_areas[m]
            
            if old_area > 0:
                temp_counts[old_area] -= 1
                if temp_counts[old_area] == 0:
                    del temp_counts[old_area]
            
            new_area = _calc_area(pts[i], pts[j], pts[k])
            new_areas[m] = new_area
            
            if new_area > 0:
                temp_counts[new_area] += 1
        
        score = max(temp_counts.values()) if temp_counts else 0
        
        if score > current_best:
            # Accept move
            for m, t_idx in enumerate(t_indices):
                areas[t_idx] = new_areas[m]
            counts = temp_counts
            current_best = score
            best_points = pts.copy()
            best_score = current_best
        else:
            # Revert move
            pts[idx] = (old_x, old_y)
            
    return best_points, best_score

def _initial_grid_points():
    """Generate initial grid of 30 points."""
    points = []
    for x in range(6):
        for y in range(5):
            points.append((x, y))
    return points

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Start with grid
    init_points = _initial_grid_points()
    
    # Run local search
    improved_points, improved_score = _local_search(init_points, max_iter=50000, time_limit=40)
    
    if time.time() > deadline:
        return improved_points
    
    # Try a second phase with a different initialization or perturbation if time allows
    if time.time() < deadline - 10:
        # Perturb the best found points slightly
        perturbed = list(improved_points)
        for _ in range(3):
            idx = random.randint(0, 29)
            perturbed[idx] = (random.randint(-30, 30), random.randint(-30, 30))
        
        # Check distinctness of perturbed
        valid = True
        seen = set()
        for p in perturbed:
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if valid:
            improved2, score2 = _local_search(perturbed, max_iter=10000, time_limit=int(deadline - time.time()) - 5)
            if score2 > improved_score:
                return improved2
    
    return improved_points
# EVOLVE-BLOCK-END
