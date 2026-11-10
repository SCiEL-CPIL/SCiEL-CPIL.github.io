"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def _doubled_area(p1, p2, p3):
    """Compute absolute doubled area of triangle with vertices p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

def _compute_histogram(points):
    """Compute histogram of doubled areas for all triples."""
    n = len(points)
    area_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    area_counts[a] = area_counts.get(a, 0) + 1
    return area_counts

def _evaluate(points):
    """Find the maximum multiplicity of any single positive doubled area."""
    area_counts = _compute_histogram(points)
    if not area_counts:
        return 0
    return max(area_counts.values())

def _local_search(points, iterations=5, pool_size=50):
    """
    Perform local search to improve the area multiplicity.
    """
    # Generate a pool of candidate points to swap in
    # Use a dense grid of candidates to ensure good geometry
    # 7x7 grid centered at 0 gives 49 points
    local_candidates = [(x, y) for x in range(-3, 4) for y in range(-3, 4)]
    
    current_points = list(points)
    current_hist = _compute_histogram(current_points)
    current_best_area = max(current_hist, key=current_hist.get)
    current_best_count = current_hist[current_best_area]
    
    for _ in range(iterations):
        improved = False
        
        for i in range(len(current_points)):
            p_i = current_points[i]
            other_points = [p for idx, p in enumerate(current_points) if idx != i]
            
            best_swap = None
            best_delta = 0
            
            for p_j in local_candidates:
                if p_j == p_i or p_j in current_points:
                    continue
                
                # Calculate change in count for current_best_area
                delta = 0
                for a in range(len(other_points)):
                    for b in range(a+1, len(other_points)):
                        pa = other_points[a]
                        pb = other_points[b]
                        old_a = _doubled_area(p_i, pa, pb)
                        new_a = _doubled_area(p_j, pa, pb)
                        
                        if old_a == current_best_area:
                            delta -= 1
                        if new_a == current_best_area:
                            delta += 1
                
                if delta > best_delta:
                    best_delta = delta
                    best_swap = p_j
            
            if best_swap is not None and best_delta > 0:
                # Perform swap
                p_i_old = current_points[i]
                current_points[i] = best_swap
                
                # Update histogram
                other_points = [p for idx, p in enumerate(current_points) if idx != i]
                for a in range(len(other_points)):
                    for b in range(a+1, len(other_points)):
                        pa = other_points[a]
                        pb = other_points[b]
                        old_a = _doubled_area(p_i_old, pa, pb)
                        new_a = _doubled_area(best_swap, pa, pb)
                        
                        if old_a > 0:
                            current_hist[old_a] -= 1
                            if current_hist[old_a] == 0:
                                del current_hist[old_a]
                        if new_a > 0:
                            current_hist[new_a] = current_hist.get(new_a, 0) + 1
                
                # Update best area/count
                new_best_count = max(current_hist.values())
                new_best_area = max(current_hist, key=current_hist.get)
                
                if new_best_count > current_best_count:
                    current_best_count = new_best_count
                    current_best_area = new_best_area
                    improved = True
                else:
                    # Revert
                    current_points[i] = p_i_old
                    for a in range(len(other_points)):
                        for b in range(a+1, len(other_points)):
                            pa = other_points[a]
                            pb = other_points[b]
                            old_a = _doubled_area(p_i_old, pa, pb)
                            new_a = _doubled_area(best_swap, pa, pb)
                            
                            if old_a > 0:
                                current_hist[old_a] = current_hist.get(old_a, 0) + 1
                            if new_a > 0:
                                current_hist[new_a] -= 1
                                if current_hist[new_a] == 0:
                                    del current_hist[new_a]
        
        if not improved:
            break

    return current_points

def construct():
    """
    Construct 30 distinct integer points in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area.
    """
    best_score = 0
    best_points = None
    
    # Strategy 1: Generate several initial grid candidates
    initial_candidates = []
    
    # 6x5 grid
    pts = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    if len(pts) == 30:
        initial_candidates.append(pts)
        
    # 5x6 grid
    pts = [(x, y) for x in range(-2, 3) for y in range(-3, 3)]
    if len(pts) == 30:
        initial_candidates.append(pts)
        
    # 10x3 grid
    pts = [(x, y) for x in range(-5, 5) for y in range(-1, 2)]
    if len(pts) == 30:
        initial_candidates.append(pts)
        
    # 3x10 grid
    pts = [(x, y) for x in range(-1, 2) for y in range(-5, 5)]
    if len(pts) == 30:
        initial_candidates.append(pts)
        
    # 15x2 grid
    pts = [(x, y) for x in range(-7, 8) for y in range(0, 2)]
    if len(pts) == 30:
        initial_candidates.append(pts)
        
    # 2x15 grid
    pts = [(x, y) for x in range(0, 2) for y in range(-7, 8)]
    if len(pts) == 30:
        initial_candidates.append(pts)
        
    for cand in initial_candidates:
        if len(set(cand)) != 30:
            continue
        # Validate bounds
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in cand):
            continue
            
        # Run local search
        refined = _local_search(cand, iterations=5)
        score = _evaluate(refined)
        
        if score > best_score:
            best_score = score
            best_points = refined
    
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        
    return best_points
# EVOLVE-BLOCK-END
