"""New program using hypergraph-based area targeting approach."""
import numpy as np
from collections import Counter
import random
from itertools import combinations

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    
    Approach:
    1. Generate diverse candidate sets from structured lattice patterns.
    2. For each candidate, identify the target area A with highest multiplicity.
    3. Apply targeted local search: swap points to increase count of triples with area A.
    """
    
    def compute_all_areas_vectorized(points):
        """Compute all C(n,3) doubled areas using vectorized operations."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        
        # Generate all triple indices
        idx_i, idx_j, idx_k = [], [], []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    idx_i.append(i)
                    idx_j.append(j)
                    idx_k.append(k)
        
        idx_i = np.array(idx_i)
        idx_j = np.array(idx_j)
        idx_k = np.array(idx_k)
        
        xi = pts[idx_i, 0]
        yi = pts[idx_i, 1]
        xj = pts[idx_j, 0]
        yj = pts[idx_j, 1]
        xk = pts[idx_k, 0]
        yk = pts[idx_k, 1]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        return areas
    
    def get_best_area(areas):
        """Get the most frequent positive area and its count."""
        positive_areas = areas[areas > 0]
        if len(positive_areas) == 0:
            return 0, 0
        counter = Counter(positive_areas.tolist())
        best_area, best_count = max(counter.items(), key=lambda x: x[1])
        return best_count, best_area
    
    # Generate diverse candidate sets from structured patterns
    candidates = []
    
    # Pattern 1: Rectangular grids
    for w in range(4, 10):
        for h in range(4, 10):
            if w * h >= 30:
                pts = [(x, y) for x in range(w) for y in range(h)]
                if len(pts) >= 30:
                    candidates.append(pts[:30])
    
    # Pattern 2: Centered grids
    for size in range(5, 10):
        center = 15
        pts = []
        for x in range(-size//2, size//2 + 1):
            for y in range(-size//2, size//2 + 1):
                pts.append((center + x, center + y))
        if len(pts) >= 30:
            candidates.append(pts[:30])
    
    # Pattern 3: Scaled grids
    for scale in [1, 2, 3]:
        for base_w in [5, 6]:
            for base_h in [5, 6]:
                pts = [(x*scale, y*scale) for x in range(base_w) for y in range(base_h)]
                if len(pts) >= 30:
                    pts = pts[:30]
                    if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
                        candidates.append(pts)
    
    # Pattern 4: Hexagonal-like lattice
    for scale in [1, 2, 3]:
        pts = []
        for row in range(7):
            for col in range(5):
                x = col * 2 * scale + (row % 2) * scale
                y = row * scale
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if len(pts) >= 30:
            candidates.append(pts[:30])
    
    # Pattern 5: Diagonal patterns
    for step in [1, 2]:
        for offset_x in range(0, 30, 5):
            for offset_y in range(0, 30, 5):
                pts = []
                for i in range(30):
                    x = offset_x + i % 6
                    y = offset_y + i // 6
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.append((x, y))
                    else:
                        break
                if len(pts) == 30 and len(set(pts)) == 30:
                    candidates.append(pts)
    
    # Evaluate all candidates
    random.seed(42)
    best_score = 0
    best_points = None
    best_area = 0
    
    for cand in candidates:
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        areas = compute_all_areas_vectorized(cand)
        score, area = get_best_area(areas)
        if score > best_score:
            best_score = score
            best_points = cand[:]
            best_area = area
    
    if best_points is None:
        # Fallback: simple grid
        best_points = [(x, y) for x in range(6) for y in range(5)]
        areas = compute_all_areas_vectorized(best_points)
        best_score, best_area = get_best_area(areas)
    
    # Targeted local search
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    random.shuffle(all_valid_points)
    
    for iteration in range(6):
        improved = False
        current_set = set(best_points)
        
        for i in range(30):
            old_point = best_points[i]
            
            # Generate candidate replacements
            candidates_to_try = []
            
            # Points in a neighborhood
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ not in current_set and np_ != old_point:
                            candidates_to_try.append(np_)
            
            # Also try some random points
            for _ in range(15):
                np_ = random.choice(all_valid_points)
                if np_ not in current_set and np_ != old_point:
                    candidates_to_try.append(np_)
            
            random.shuffle(candidates_to_try)
            
            for new_point in candidates_to_try[:12]:  # Limit tries per position
                new_points = best_points[:]
                new_points[i] = new_point
                
                areas = compute_all_areas_vectorized(new_points)
                score, area = get_best_area(areas)
                
                if score > best_score:
                    best_score = score
                    best_points = new_points
                    best_area = area
                    current_set = set(best_points)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_points
