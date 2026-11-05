import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # Helper: Check if 4 points are collinear
    def are_collinear(p1, p2, p3, p4):
        def collinear(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) == (c[0]-a[0])*(b[1]-a[1])
        return collinear(p1, p2, p3) and collinear(p1, p2, p4)

    def is_valid(points):
        # No 4 collinear - early exit
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Check if there's a 4th point on line i-j-k
                    dx1 = points[j][0] - points[i][0]
                    dy1 = points[j][1] - points[i][1]
                    dx2 = points[k][0] - points[i][0]
                    dy2 = points[k][1] - points[i][1]
                    if dx1 * dy2 != dx2 * dy1:
                        continue  # Not collinear
                    # Found 3 collinear, check for 4th
                    for l in range(k+1, n):
                        dx3 = points[l][0] - points[i][0]
                        dy3 = points[l][1] - points[i][1]
                        if dx1 * dy3 == dx2 * dy3:
                            return False
        return True

    # Helper: Compute Score (Largest subset with no 3 collinear)
    # Uses bitmask-based B&B for speed
    def calc_score(points):
        n = len(points)
        if n <= 2:
            return n
        
        # Precompute collinear triples as bitmask pairs
        # For each vertex v, store bitmask of pairs (u,w) that form triple with v
        # Actually, store for each v: list of (u, w) pairs
        pairs_for_vertex = [[] for _ in range(n)]
        triple_count = [0] * n
        
        for i, j, k in itertools.combinations(range(n), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                pairs_for_vertex[i].append((j, k))
                pairs_for_vertex[j].append((i, k))
                pairs_for_vertex[k].append((i, j))
                triple_count[i] += 1
                triple_count[j] += 1
                triple_count[k] += 1
        
        # Order vertices by triple count descending (harder to place first)
        order = sorted(range(n), key=lambda x: -triple_count[x])
        # Map original index to position in order
        pos_in_order = [0] * n
        for pos, idx in enumerate(order):
            pos_in_order[idx] = pos
        
        best_size = 0
        chosen_mask = 0  # Bitmask of chosen vertices
        
        def backtrack(pos, current_size):
            nonlocal best_size, chosen_mask
            if pos == n:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pruning: even if we take all remaining, can't beat best
            if current_size + (n - pos) <= best_size:
                return
            
            v = order[pos]
            
            # Option 1: Exclude v
            backtrack(pos + 1, current_size)
            
            # Option 2: Include v - check if any pair in chosen forms triple with v
            valid = True
            for u, w in pairs_for_vertex[v]:
                if (chosen_mask >> u) & 1 and (chosen_mask >> w) & 1:
                    valid = False
                    break
            
            if valid:
                chosen_mask |= (1 << v)
                backtrack(pos + 1, current_size + 1)
                chosen_mask &= ~(1 << v)
        
        backtrack(0, 0)
        return best_size

    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))
    
    def generate_initial():
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        return points

    def try_local_search(points, score, max_iters=500):
        """Local search: try single-point replacements and swaps"""
        best_pts = list(points)
        best_sc = score
        seen_configs = set()
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            if random.random() < 0.5:
                # Single point replacement
                idx = random.randint(0, 19)
                old_p = best_pts[idx]
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                if new_p in best_pts:
                    continue
                best_pts[idx] = new_p
            else:
                # Swap two points
                i, j = random.sample(range(20), 2)
                best_pts[i], best_pts[j] = best_pts[j], best_pts[i]
            
            if not is_valid(best_pts):
                # Revert
                if random.random() < 0.5:
                    idx = random.randint(0, 19)
                    best_pts[idx] = random_point()
                else:
                    i, j = random.sample(range(20), 2)
                    best_pts[i], best_pts[j] = best_pts[j], best_pts[i]
                continue
            
            new_sc = calc_score(best_pts)
            if new_sc < best_sc:
                best_sc = new_sc
                best_pts = list(best_pts)
        
        return best_pts, best_sc

    best_points = None
    best_score = 21
    
    # Phase 1: Random search for valid configs
    while time.time() < deadline - 30.0:
        points = generate_initial()
        
        if not is_valid(points):
            continue
            
        score = calc_score(points)
        
        if score < best_score:
            best_score = score
            best_points = list(points)
            if best_score <= 9:
                break
    
    # Phase 2: Local search polish
    if best_points is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            # Multiple rounds of local search with different random seeds
            for round_num in range(5):
                if time.time() > deadline - 5.0:
                    break
                best_points, best_score = try_local_search(
                    best_points, best_score, 
                    max_iters=max(100, int(remaining_time / 10))
                )
                if best_score <= 8:
                    break
    
    if best_points is None:
        # Fallback: use a known configuration
        best_points = [[x, x*x % 200 - 100] for x in range(-10, 10)]
        # Ensure validity
        if not is_valid(best_points):
            best_points = generate_initial()
            while not is_valid(best_points):
                best_points = generate_initial()
    
    return [[p[0], p[1]] for p in best_points]
