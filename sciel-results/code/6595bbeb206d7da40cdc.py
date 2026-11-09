import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second margin for safety
    
    def _elapsed():
        return time.time() - start_time
    
    def _remaining():
        return deadline - time.time()
    
    def _collinear(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0
    
    def _has_four_collinear(points):
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            slope_counts = {}
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                # Normalize direction
                if dx == 0:
                    key = (1, 0)
                elif dy == 0:
                    key = (0, 1)
                else:
                    g = abs(dx)
                    if abs(dy) > g:
                        g = abs(dy)
                    from math import gcd
                    g = gcd(abs(dx), abs(dy))
                    dx_n = dx // g
                    dy_n = dy // g
                    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                        dx_n = -dx_n
                        dy_n = -dy_n
                    key = (dx_n, dy_n)
                slope_counts[key] = slope_counts.get(key, 0) + 1
                if slope_counts[key] >= 3:
                    return True
        return False
    
    def _greedy_max_no_three_collinear(points):
        """Fast greedy heuristic: repeatedly add point that creates fewest new collinear triples."""
        n = len(points)
        # Precompute for each pair (i,j), how many k are collinear with i,j
        # Actually, for greedy, we need: given current subset S, for candidate p,
        # count how many pairs (a,b) in S have p collinear with a,b
        # We'll just compute on the fly for speed
        
        best_subset = []
        remaining = list(range(n))
        
        while remaining:
            best_idx = -1
            best_conflicts = float('inf')
            for idx in remaining:
                p = points[idx]
                conflicts = 0
                # For each pair in current subset, check if p is collinear
                for i in range(len(best_subset)):
                    for j in range(i+1, len(best_subset)):
                        a = points[best_subset[i]]
                        b = points[best_subset[j]]
                        if _collinear(a, b, p):
                            conflicts += 1
                            break  # One conflict is enough to count this as bad
                # Prefer points with fewer conflicts
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_idx = idx
                if conflicts == 0:
                    break  # Perfect choice found
            
            if best_idx == -1:
                break
            
            best_subset.append(remaining[best_idx])
            remaining.pop(remaining.index(best_idx))
        
        return len(best_subset)
    
    def _exact_max_no_three_collinear(points):
        """Exact branch and bound - use sparingly."""
        n = len(points)
        collinear_pairs = {}
        for i in range(n):
            for j in range(i+1, n):
                collinear_pairs[(i,j)] = []
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if _collinear(points[i], points[j], points[k]):
                        collinear_pairs[(i,j)].append(k)
        
        best = 0
        current = []
        
        def branch_and_bound(next_idx):
            nonlocal best
            if len(current) + (n - next_idx) <= best:
                return
            
            if next_idx == n:
                if len(current) > best:
                    best = len(current)
                return
            
            # Skip
            branch_and_bound(next_idx + 1)
            
            # Try adding
            valid = True
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    a = current[i]
                    b = current[j]
                    key = (a, b) if a < b else (b, a)
                    if next_idx in collinear_pairs.get(key, []):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current.append(next_idx)
                branch_and_bound(next_idx + 1)
                current.pop()
        
        branch_and_bound(0)
        return best
    
    # Generate initial valid configuration
    def _construct_perturbed_parabola():
        """Start with points on a parabola and perturb to break 4-collinearities."""
        pts = []
        for x in range(-10, 10):
            y = x * x
            # Small perturbation
            px = x + random.randint(-2, 2)
            py = y + random.randint(-2, 2)
            if abs(px) <= 100 and abs(py) <= 100:
                pts.append([px, py])
        
        # Remove duplicates and ensure 20 points
        seen = set()
        unique_pts = []
        for p in pts:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique_pts.append(p)
        
        # If we don't have 20, add more random points
        while len(unique_pts) < 20:
            p = [random.randint(-100, 100), random.randint(-100, 100)]
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique_pts.append(p)
        
        return unique_pts[:20]
    
    # Find initial valid configuration
    best_points = None
    for attempt in range(500):
        if _remaining() < 5:
            break
        cand = _construct_perturbed_parabola()
        if not _has_four_collinear(cand):
            best_points = cand
            break
    
    if best_points is None:
        # Fallback: try random
        for attempt in range(1000):
            if _remaining() < 5:
                break
            cand = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
            if len(set(map(tuple, cand))) == 20 and not _has_four_collinear(cand):
                best_points = cand
                break
    
    if best_points is None:
        # Last resort: convex position (no 3 collinear, so no 4 either)
        best_points = [[i, i*i] for i in range(20)]
    
    # Local search
    best_score = _greedy_max_no_three_collinear(best_points)
    
    # Periodically verify with exact B&B on promising candidates
    exact_check_interval = 50
    check_count = 0
    
    while _remaining() > 10:
        idx = random.randint(0, 19)
        old_p = best_points[idx]
        
        # Try multiple perturbations
        for _ in range(30):
            if _remaining() < 10:
                break
            
            # Random perturbation
            new_p = [old_p[0] + random.randint(-8, 8), old_p[1] + random.randint(-8, 8)]
            if abs(new_p[0]) > 100 or abs(new_p[1]) > 100:
                continue
            if new_p in best_points:
                continue
            
            cand = best_points.copy()
            cand[idx] = new_p
            
            if _has_four_collinear(cand):
                continue
            
            # Fast greedy evaluation
            score = _greedy_max_no_three_collinear(cand)
            
            if score < best_score:
                best_score = score
                best_points = cand
                check_count = 0
                break
            elif score == best_score and random.random() < 0.15:
                # Sideways move to escape local minima
                best_points = cand
                check_count += 1
                break
        
        # Periodically do exact check on current best
        check_count += 1
        if check_count >= exact_check_interval and _remaining() > 20:
            exact_score = _exact_max_no_three_collinear(best_points)
            if exact_score < best_score:
                best_score = exact_score
            check_count = 0
    
    # Final exact evaluation on the best candidate found
    if _remaining() > 5:
        final_score = _exact_max_no_three_collinear(best_points)
        best_score = final_score
    
    return best_points
