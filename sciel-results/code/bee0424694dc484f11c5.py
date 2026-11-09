import random
import math
import time
from itertools import combinations

_START_TIME = None
_DEADLINE = 145.0 # Safety margin

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _find_max_3_free_subset(points):
    """Returns the size and the actual subset (list of indices) of the largest 3-free subset."""
    n = len(points)
    if n <= 2:
        return n, list(range(n))
    
    # Precompute collinearity masks
    collinearity = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _is_collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinearity[i][j] = mask
            collinearity[j][i] = mask
            
    best_size = [0]
    best_subset = [list(range(n))]
    
    def backtrack(i, subset_mask, subset_list):
        if len(subset_list) > best_size[0]:
            best_size[0] = len(subset_list)
            best_subset[0] = list(subset_list)
        if i == n:
            return
        if len(subset_list) + (n - i) <= best_size[0]:
            return
        
        # Exclude
        backtrack(i + 1, subset_mask, subset_list)
        
        # Include
        can_add = True
        for a in subset_list:
            if collinearity[a][i] & subset_mask:
                can_add = False
                break
        
        if can_add:
            subset_list.append(i)
            backtrack(i + 1, subset_mask | (1 << i), subset_list)
            subset_list.pop()
            
    backtrack(0, 0, [])
    return best_size[0], best_subset[0]

def _count_collinear_triples(points):
    """Count the number of collinear triples in the configuration."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _generate_structured_initialization(seed_offset=0):
    """Generate a structured initial configuration with controlled collinearity."""
    random.seed(42 + seed_offset)
    
    # Use a base pattern that avoids 4-collinear but has some 3-collinear
    # Strategy: Use a slightly perturbed grid with specific perturbations
    # to create controlled collinear triples
    
    points = []
    
    # Use 5x4 grid with structured perturbations
    # Perturb to break 4-collinear but keep some 3-collinear
    for row in range(4):
        for col in range(5):
            base_x = col * 20
            base_y = row * 25
            
            # Structured perturbation: use different patterns for different rows
            if row % 2 == 0:
                px = base_x + random.randint(-2, 2)
                py = base_y + random.randint(-3, 3)
            else:
                px = base_x + random.randint(-3, 3)
                py = base_y + random.randint(-2, 2)
            
            # Keep within bounds
            px = max(-100, min(100, px))
            py = max(-100, min(100, py))
            points.append((px, py))
    
    # Ensure distinct
    for i in range(20):
        for j in range(i + 1, 20):
            if points[i] == points[j]:
                points[j] = (points[j][0] + 1, points[j][1])
    
    # Fix any 4-collinear issues
    attempts = 0
    while _has_four_collinear(points) and attempts < 500:
        # Find a 4-collinear set and move one point
        found = False
        for i in range(20):
            for j in range(i + 1, 20):
                for k in range(j + 1, 20):
                    for l in range(k + 1, 20):
                        if _is_collinear(points[i], points[j], points[k]) and \
                           _is_collinear(points[i], points[j], points[l]):
                            # Move point l
                            old = points[l]
                            points[l] = (old[0] + random.randint(-5, 5), old[1] + random.randint(-5, 5))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
        attempts += 1
    
    return points

def _targeted_local_search(points, current_score, time_budget):
    """
    Targeted local search that specifically tries to break the largest 3-free subset.
    Returns improved points and score.
    """
    best_points = list(points)
    best_score = current_score
    no_improve = 0
    max_no_improve = 30
    
    while _time_left() > time_budget and no_improve < max_no_improve:
        # Find the current maximum 3-free subset
        subset_size, subset_indices = _find_max_3_free_subset(points)
        
        if subset_size < best_score:
            best_score = subset_size
            best_points = list(points)
            no_improve = 0
        else:
            no_improve += 1
        
        if subset_size <= 8:
            break  # Can't do much better than 8
        
        # Try to break the 3-free subset by creating new collinear triples
        # within it
        if len(subset_indices) >= 3:
            # Pick pairs from the 3-free subset and try to add a point that
            # creates a collinear triple with them
            pairs = list(combinations(subset_indices, 2))
            random.shuffle(pairs)
            
            for p1_idx, p2_idx in pairs[:10]:  # Try up to 10 pairs
                if _time_left() <= time_budget:
                    break
                    
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                
                # Find integer points on the line through p1 and p2
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                if dx == 0 and dy == 0:
                    continue
                    
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                    
                sx = dx // g
                sy = dy // g
                
                # Try various positions along the line
                for step in range(-15, 16):
                    if step == 0:
                        continue
                    nx = p1[0] + step * sx
                    ny = p1[1] + step * sy
                    
                    if not (-100 <= nx <= 100 and -100 <= ny <= 100):
                        continue
                    
                    new_point = (nx, ny)
                    
                    # Check if this point is already in use
                    if new_point in points:
                        continue
                    
                    # Try replacing a point that is NOT in the 3-free subset
                    # to minimize damage
                    candidates = [i for i in range(20) if i not in subset_indices]
                    if not candidates:
                        candidates = list(range(20))
                    
                    random.shuffle(candidates)
                    
                    for idx in candidates[:3]:
                        old_point = points[idx]
                        points[idx] = new_point
                        
                        if _has_four_collinear(points):
                            points[idx] = old_point
                            continue
                        
                        new_score, _ = _find_max_3_free_subset(points)
                        
                        if new_score < best_score:
                            best_score = new_score
                            best_points = list(points)
                            no_improve = 0
                        else:
                            points[idx] = old_point
                        
                        break  # Only try one replacement per new_point
                    
                    if no_improve >= max_no_improve:
                        break
        
        # Also try random perturbations
        if _time_left() > time_budget + 2.0:
            idx = random.randint(0, 19)
            old_point = points[idx]
            new_point = (old_point[0] + random.randint(-5, 5), old_point[1] + random.randint(-5, 5))
            
            if new_point not in points:
                points[idx] = new_point
                
                if _has_four_collinear(points):
                    points[idx] = old_point
                else:
                    new_score, _ = _find_max_3_free_subset(points)
                    if new_score < best_score:
                        best_score = new_score
                        best_points = list(points)
                        no_improve = 0
                    else:
                        points[idx] = old_point
    
    return best_points, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_score = 20
    best_points = None
    
    # Try multiple structured initializations
    for restart in range(8):
        if _time_left() < 25.0:
            break
        
        # Generate structured initial configuration
        points = _generate_structured_initialization(seed_offset=restart)
        
        if _has_four_collinear(points):
            continue
        
        # Evaluate initial score
        current_score, _ = _find_max_3_free_subset(points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = list(points)
        
        # Apply targeted local search
        if _time_left() > 15.0:
            points, current_score = _targeted_local_search(points, current_score, 10.0)
            
            if current_score < best_score:
                best_score = current_score
                best_points = list(points)
        
        # If we've found a good score, do more refinement
        if best_score <= 9 and _time_left() > 20.0:
            # Do additional rounds of targeted search
            for extra_round in range(3):
                if _time_left() < 15.0:
                    break
                points, current_score = _targeted_local_search(best_points, best_score, 10.0)
                if current_score < best_score:
                    best_score = current_score
                    best_points = list(points)
    
    if best_points is None:
        # Fallback: use a simple valid configuration
        # Parabola y = x^2 has no 3 collinear, so score = 20
        # But we need something with lower score
        # Use a configuration with more collinear triples
        best_points = []
        for i in range(20):
            if i < 10:
                best_points.append((i, i * 2))
            else:
                best_points.append((i - 10, (i - 10) * 2 + 1))
        
        # Fix any issues
        for i in range(20):
            for j in range(i + 1, 20):
                if best_points[i] == best_points[j]:
                    best_points[j] = (best_points[j][0] + 1, best_points[j][1])
        
        # Ensure no 4-collinear
        attempts = 0
        while _has_four_collinear(best_points) and attempts < 1000:
            idx = random.randint(0, 19)
            old = best_points[idx]
            best_points[idx] = (old[0] + random.randint(-5, 5), old[1] + random.randint(-5, 5))
            attempts += 1
    
    return [list(p) for p in best_points]
