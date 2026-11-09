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
    """Returns the size of the largest 3-free subset."""
    n = len(points)
    if n <= 2:
        return n
    
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
    
    def backtrack(i, subset_mask, subset_list):
        if len(subset_list) > best_size[0]:
            best_size[0] = len(subset_list)
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
    return best_size[0]

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

def _generate_initial_points():
    """Generate a random valid initial configuration."""
    random.seed(42)
    points = []
    attempts = 0
    while len(points) < 20 and attempts < 1000:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in points:
            points.append((x, y))
            attempts += 1
    
    # Fix 4-collinear if any
    while _has_four_collinear(points):
        # Find a 4-collinear set and move one point
        found = False
        for i in range(20):
            for j in range(i + 1, 20):
                for k in range(j + 1, 20):
                    for l in range(k + 1, 20):
                        if _is_collinear(points[i], points[j], points[k]) and \
                           _is_collinear(points[i], points[j], points[l]):
                            old = points[l]
                            while True:
                                nx = old[0] + random.randint(-10, 10)
                                ny = old[1] + random.randint(-10, 10)
                                if -100 <= nx <= 100 and -100 <= ny <= 100 and (nx, ny) not in points:
                                    points[l] = (nx, ny)
                                    break
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
            
    return points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = _generate_initial_points()
    if not best_points:
         best_points = [(i, i*i) for i in range(20)]
    
    # Initial evaluation
    current_triples = _count_collinear_triples(best_points)
    current_score = _find_max_3_free_subset(best_points)
    best_score = current_score
    best_energy = -current_triples # Lower is better (more triples)
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    current_points = list(best_points)
    current_energy = best_energy
    
    while _time_left() > 10.0 and T > T_min:
        # Pick a move
        if random.random() < 0.4:
            # Random move
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            new_p = (random.randint(-100, 100), random.randint(-100, 100))
            if new_p in current_points:
                continue
            candidate = list(current_points)
            candidate[idx] = new_p
        else:
            # Line-snapping move: Pick 2 points, try to move a 3rd onto their line
            i, j = random.sample(range(20), 2)
            p1, p2 = current_points[i], current_points[j]
            
            # Find integer points on line p1-p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            sx = dx // g
            sy = dy // g
            
            # Find a candidate point index to move
            candidates = [k for k in range(20) if k != i and k != j]
            if not candidates:
                continue
            idx = random.choice(candidates)
            
            # Try to place points[idx] on the line
            found_valid_move = False
            for step in range(-20, 21):
                nx = p1[0] + step * sx
                ny = p1[1] + step * sy
                if not (-100 <= nx <= 100 and -100 <= ny <= 100):
                    continue
                new_p = (nx, ny)
                if new_p in current_points:
                    continue
                # Check if new_p is i or j
                if new_p == p1 or new_p == p2:
                    continue
                
                candidate = list(current_points)
                candidate[idx] = new_p
                
                # Check 4-collinear
                is_valid = True
                # Only check lines involving idx for speed
                for a in range(20):
                    if a == idx: continue
                    for b in range(a+1, 20):
                        if b == idx: continue
                        if _is_collinear(candidate[idx], candidate[a], candidate[b]):
                             # Check if there is a 4th
                             for c in range(20):
                                 if c != idx and c != a and c != b:
                                     if _is_collinear(candidate[idx], candidate[a], candidate[c]):
                                         is_valid = False
                                         break
                                 if not is_valid: break
                    if not is_valid: break
                if not is_valid:
                    continue
                
                # If valid, accept as candidate
                found_valid_move = True
                break
            
            if not found_valid_move:
                continue
        
        # Evaluate candidate
        new_triples = _count_collinear_triples(candidate)
        new_energy = -new_triples
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            current_points = candidate
            current_energy = new_energy
            
            # Update best if energy is better
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = list(current_points)
                
                # Periodically check true score
                # This is expensive, so we do it when we find a new best energy
                # or every 1000 steps
                # For now, let's check if the number of triples is significantly higher
                # The true score is correlated with triple count.
                # We will check true score at the end, and maybe once in the middle.
                # To save time, we'll check true score if we have time left.
                
        T *= cooling_rate
        
    # Final check and refinement
    # Compute true score for best_points
    final_score = _find_max_3_free_subset(best_points)
    
    # If we have time, do a quick local search on the best point
    if _time_left() > 20.0:
        # Simple hill climbing on true score
        improved = True
        while improved and _time_left() > 15.0:
            improved = False
            for idx in range(20):
                old_p = best_points[idx]
                # Try moving to nearby points
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0: continue
                        new_p = (old_p[0] + dx, old_p[1] + dy)
                        if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                            continue
                        if new_p in best_points:
                            continue
                        candidate = list(best_points)
                        candidate[idx] = new_p
                        if _has_four_collinear(candidate):
                            continue
                        new_score = _find_max_3_free_subset(candidate)
                        if new_score < final_score:
                            best_points = candidate
                            final_score = new_score
                            improved = True
                            break
                    if improved: break
                if improved: break

    return [list(p) for p in best_points]
