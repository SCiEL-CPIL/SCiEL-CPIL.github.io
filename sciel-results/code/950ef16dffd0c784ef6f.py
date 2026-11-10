"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

def _doubled_area(p1, p2, p3):
    """Compute absolute doubled area of triangle with vertices p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

def _count_area_A(points, A):
    """Count number of triples with doubled area exactly A."""
    count = 0
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _doubled_area(points[i], points[j], points[k]) == A:
                    count += 1
    return count

def _greedy_construct(pool, target_A, num_points=30):
    """
    Greedily select num_points from pool to maximize count of triangles with area target_A.
    """
    S = []
    pool_list = list(pool)
    
    # Start with a few random or central points to bootstrap
    # Using center points is a good heuristic
    center = (0, 0)
    if center in pool_list:
        S.append(center)
        pool_list.remove(center)
    
    while len(S) < num_points and pool_list:
        best_p = None
        best_gain = -1
        
        for p in pool_list:
            gain = 0
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    if _doubled_area(p, S[i], S[j]) == target_A:
                        gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is not None:
            S.append(best_p)
            pool_list.remove(best_p)
        else:
            S.append(pool_list[0])
            pool_list.pop(0)
            
    return S

def _local_search_area(points, target_A, pool, iterations=3):
    """
    Local search to improve count of triangles with area target_A.
    """
    current_points = list(points)
    current_count = _count_area_A(current_points, target_A)
    pool_set = set(pool)
    
    for _ in range(iterations):
        improved = False
        for i in range(len(current_points)):
            p_i = current_points[i]
            best_swap = None
            best_new_count = current_count
            
            others = [p for idx, p in enumerate(current_points) if idx != i]
            
            for p_j in pool_set:
                if p_j in current_points:
                    continue
                
                delta = 0
                for u_idx in range(len(others)):
                    for v_idx in range(u_idx+1, len(others)):
                        if _doubled_area(p_i, others[u_idx], others[v_idx]) == target_A:
                            delta -= 1
                        if _doubled_area(p_j, others[u_idx], others[v_idx]) == target_A:
                            delta += 1
                
                if current_count + delta > best_new_count:
                    best_new_count = current_count + delta
                    best_swap = p_j
            
            if best_swap is not None:
                current_points[i] = best_swap
                current_count = best_new_count
                improved = True
        
        if not improved:
            break
            
    return current_points, current_count

def construct():
    """
    Construct 30 distinct integer points in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area.
    """
    start_time = time.time()
    time_limit = 55
    
    # Define candidate pool: 11x11 grid from -5 to 5
    pool = [(x, y) for x in range(-5, 6) for y in range(-5, 6)]
    pool_set = set(pool)
    
    best_points = None
    best_score = 0
    
    target_areas = [1, 2, 4]
    
    for A in target_areas:
        if time.time() - start_time > time_limit:
            break
            
        S = _greedy_construct(pool, A, 30)
        refined_S, count_A = _local_search_area(S, A, pool_set, iterations=3)
        
        # Evaluate true max multiplicity
        hist = {}
        n = len(refined_S)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = _doubled_area(refined_S[i], refined_S[j], refined_S[k])
                    if a > 0:
                        hist[a] = hist.get(a, 0) + 1
        
        if hist:
            max_mult = max(hist.values())
            if max_mult > best_score:
                best_score = max_mult
                best_points = refined_S
                
    if best_points is None:
        best_points = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
        
    return best_points
# EVOLVE-BLOCK-END
