"""Rewritten program: two-parabola start + local search with exact branch-and-bound scoring."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def has_four_collinear(points):
        n = len(points)
        # Check all 4-subsets
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_collinear(points[i], points[j], points[k]) and \
                           are_collinear(points[i], points[j], points[k], points[l]) if False else \
                           (are_collinear(points[i], points[j], points[k]) and are_collinear(points[i], points[j], points[l])):
                            return True
        return False

    def has_four_collinear_fast(points):
        """Check if any 4 points are collinear using line grouping."""
        n = len(points)
        line_count = {}
        for i in range(n):
            for j in range(i+1, n):
                # Compute line key (normalized)
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize: divide by gcd
                import math
                g = math.gcd(abs(dx), abs(dy)) if dx != 0 or dy != 0 else 1
                if g > 0:
                    dx_n = dx // g
                    dy_n = dy // g
                else:
                    dx_n, dy_n = dx, dy
                # Line key: (dy_n, -dx_n, dy_n*x1 - dx_n*y1) normalized
                # Actually use: a*x + b*y + c = 0 where a=dy, b=-dx, c=dx*y1-dy*x1
                a = dy_n
                b = -dx_n
                c = dx_n * y1 - dy_n * x1
                # Normalize sign
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a, b, c = -a, -b, -c
                key = (a, b, c)
                line_count[key] = line_count.get(key, 0) + 1
        # If any line has count >= C(4,2)=6 pairs... wait, that's not right.
        # Better approach: for each line key, count how many points lie on it.
        # Let me redo this properly.
        
        # Map from line key to set of point indices
        line_points = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                import math
                g = math.gcd(abs(dx), abs(dy)) if dx != 0 or dy != 0 else 1
                if g > 0:
                    dx_n = dx // g
                    dy_n = dy // g
                else:
                    dx_n, dy_n = dx, dy
                a = dy_n
                b = -dx_n
                c = dx_n * y1 - dy_n * x1
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a, b, c = -a, -b, -c
                key = (a, b, c)
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        for key, pts in line_points.items():
            if len(pts) >= 4:
                return True
        return False

    def count_three_collinear(points):
        """Count number of collinear triples."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def max_independent_set_branch_and_bound(points):
        """Exact branch and bound to find largest subset with no three collinear."""
        n = len(points)
        
        # Precompute collinear triples
        collinear_triples = set()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        collinear_triples.add((i, j, k))
        
        # For each point, precompute which other points form a triple with it
        # adjacency: for each point i, set of points j such that there exists k with {i,j,k} collinear
        # Actually we need: for a partial solution S, adding point i is valid iff no triple in S ∪ {i} is collinear.
        # A triple in S ∪ {i} is collinear iff:
        #   - it's entirely in S (already checked), or
        #   - it contains i and two points from S that with i form a collinear triple.
        
        # For each i, precompute: for each pair (j,k) in collinear_triples containing i, 
        # the pair {j,k} is "blocked" for i.
        blocked_pairs_for = [set() for _ in range(n)]
        for (i, j, k) in collinear_triples:
            blocked_pairs_for[i].add((j, k))
            blocked_pairs_for[j].add((i, k))
            blocked_pairs_for[k].add((i, j))
        
        best_size = 0
        best_set = set()
        
        # Use recursive branch and bound
        # State: current_set (frozen set of indices), next_candidate_start
        # We process candidates in order 0..n-1
        def bnb(current_set, next_idx, remaining_upper_bound):
            nonlocal best_size, best_set
            
            if time.time() > deadline - 2.0:
                return
            
            # Pruning: if current size + remaining candidates can't beat best
            if len(current_set) + (n - next_idx) <= best_size:
                return
            
            if next_idx >= n:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            # Try including point next_idx
            i = next_idx
            # Check if i can be added: no pair (j,k) in current_set such that (j,k) in blocked_pairs_for[i]
            can_add = True
            for (j, k) in blocked_pairs_for[i]:
                if j in current_set and k in current_set:
                    can_add = False
                    break
            
            if can_add:
                bnb(current_set | {i}, next_idx + 1, remaining_upper_bound - 1)
            
            # Try excluding point next_idx
            bnb(current_set, next_idx + 1, remaining_upper_bound - 1)
        
        # Initial upper bound: n
        bnb(frozenset(), 0, n)
        
        return len(best_set)

    def score(points):
        """Return the largest subset size with no three collinear."""
        return max_independent_set_branch_and_bound(points)

    # Initial configuration: two parabolas
    # First parabola: y = x^2 for x in [-10, 9]
    # Second parabola: y = 2*x^2 + 1 for x in [-10, 9] but shifted
    # Actually let's use:
    # Layer 1: (x, x^2) for x in [-10, 9]  -> 20 points, but we need exactly 20 total
    # Let's use 10 points on y=x^2 and 10 on y=2x^2+1
    def make_initial_config():
        pts = []
        # First set: y = x^2
        for x in range(-10, 0):
            pts.append([x, x*x])
        # Second set: y = 2*x^2 + 1
        for x in range(-10, 0):
            pts.append([x, 2*x*x + 1])
        return pts

    # Local search: try swapping points or moving points
    def local_search(points, max_iter=500):
        current = [list(p) for p in points]
        current_score = score(current)
        best = [list(p) for p in current]
        best_score = current_score
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            # Try moving each point to a random nearby position
            for i in range(len(current)):
                if time.time() > deadline - 10.0:
                    break
                
                orig_x, orig_y = current[i]
                # Try several candidate moves
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if abs(new_x) > 100 or abs(new_y) > 100:
                            continue
                        
                        # Check distinctness
                        candidate = [list(p) for p in current]
                        candidate[i] = [new_x, new_y]
                        
                        # Check all points distinct
                        seen = set()
                        valid = True
                        for p in candidate:
                            if tuple(p) in seen:
                                valid = False
                                break
                            seen.add(tuple(p))
                        if not valid:
                            continue
                        
                        # Check no four collinear
                        if has_four_collinear_fast(candidate):
                            continue
                        
                        # Score
                        s = score(candidate)
                        if s < current_score:
                            current = candidate
                            current_score = s
                            if s < best_score:
                                best = [list(p) for p in current]
                                best_score = s
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        return best, best_score

    # Generate initial config
    points = make_initial_config()
    
    # Verify no four collinear in initial config
    if has_four_collinear_fast(points):
        # Fallback: use parabola
        points = [[x, x*x] for x in range(-10, 10)]
    
    best_points = [list(p) for p in points]
    best_score_val = score(best_points)
    
    # Run local search
    best_points, best_score_val = local_search(points, max_iter=200)
    
    # Additional restarts if time permits
    if time.time() < deadline - 30.0:
        for restart in range(5):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb the best solution
            perturbed = [list(p) for p in best_points]
            for _ in range(3):
                idx = random.randint(0, len(perturbed)-1)
                new_x = perturbed[idx][0] + random.randint(-5, 5)
                new_y = perturbed[idx][1] + random.randint(-5, 5)
                if abs(new_x) <= 100 and abs(new_y) <= 100:
                    perturbed[idx] = [new_x, new_y]
            
            # Check validity
            seen = set()
            valid = True
            for p in perturbed:
                if tuple(p) in seen:
                    valid = False
                    break
                seen.add(tuple(p))
            if not valid or has_four_collinear_fast(perturbed):
                continue
            
            p_best, p_score = local_search(perturbed, max_iter=100)
            if p_score < best_score_val:
                best_points = p_best
                best_score_val = p_score
    
    # Final check
    if has_four_collinear_fast(best_points) or len(set(tuple(p) for p in best_points)) != 20:
        return [[x, x*x] for x in range(-10, 10)]
    
    return best_points
