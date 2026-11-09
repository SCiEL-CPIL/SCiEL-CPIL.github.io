import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Helper to check collinearity
    def collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    def has_four_collinear(points):
        # Check all 4-combinations
        for combo in itertools.combinations(points, 4):
            p1, p2, p3, p4 = combo
            if collinear(p1, p2, p3) and collinear(p1, p2, p4):
                return True
        return False

    def get_lines(points):
        lines = {}
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = points[i], points[j]
                # Normalize line representation (A, B, C) for Ax + By + C = 0
                A = p1[1] - p2[1]
                B = p2[0] - p1[0]
                C = p1[0]*p2[1] - p2[0]*p1[1]
                # Normalize
                g = __import__('math').gcd(A, __import__('math').gcd(B, C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    # Handle 0,0,0? No, distinct points.
                    pass
                # Ensure consistent sign
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A, B, C = -A, -B, -C
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        return lines

    def max_no_three_collinear(points):
        n = len(points)
        lines = get_lines(points)
        # Build hypergraph of collinear triples
        triples = set()
        for line_pts in lines.values():
            if len(line_pts) >= 3:
                for t in itertools.combinations(line_pts, 3):
                    triples.add(t)
        
        # Find largest subset with no triple
        # Use simple branch and bound
        best_size = 0
        used = [False] * n
        
        def backtrack(idx, current_size, current_set):
            nonlocal best_size
            if current_size + (n - idx) <= best_size:
                return
            if idx == n:
                best_size = max(best_size, current_size)
                return
            
            # Option 1: Exclude point idx
            backtrack(idx + 1, current_size, current_set)
            
            # Option 2: Include point idx
            # Check if adding idx creates a triple
            valid = True
            for i, j in itertools.combinations(current_set, 2):
                if (i, j, idx) in triples or (i, idx, j) in triples or (idx, i, j) in triples:
                    valid = False
                    break
            if valid:
                backtrack(idx + 1, current_size + 1, current_set + [idx])

        # The above B&B is too slow for n=20 if called frequently.
        # We need a faster heuristic or a better B&B.
        # Let's use a greedy approach for estimation during search,
        # and exact B&B only for the final candidate.
        
        # Greedy Upper Bound for Independent Set:
        # Actually, we want the *exact* size for the final answer.
        # For the search cost function, we can use the number of triples as a proxy.
        # More triples -> likely smaller independent set.
        return len(triples) # Proxy cost: minimize this? No, we want to MINIMIZE the independent set size.
                            # So we want to MAXIMIZE the number of triples?
                            # Not strictly, but correlated.

    # Let's refine the cost function for search.
    # We want to MINIMIZE the size of the largest no-3-collinear subset.
    # Let's call this size S.
    # We want to find a config with small S.
    
    def exact_max_no_three_collinear(points):
        n = len(points)
        lines = get_lines(points)
        triples = set()
        for line_pts in lines.values():
            if len(line_pts) >= 3:
                for t in itertools.combinations(sorted(line_pts), 3):
                    triples.add(t)
        
        # B&B
        best_size = 0
        # Order points by degree in hypergraph to prune better?
        # Let's just do standard B&B.
        # To speed up, we can precompute for each point which other pairs form a triple with it.
        # conflict[i] = set of pairs (j, k) such that (i, j, k) is a triple.
        # When adding i, we must not have any j, k in current_set such that (j, k) in conflict[i].
        
        conflict = [set() for _ in range(n)]
        for (i, j, k) in triples:
            conflict[i].add((j, k))
            conflict[j].add((i, k))
            conflict[k].add((i, j))
            
        def backtrack(idx, current_size, current_set):
            nonlocal best_size
            if current_size + (n - idx) <= best_size:
                return
            if idx == n:
                best_size = max(best_size, current_size)
                return
            
            # Exclude
            backtrack(idx + 1, current_size, current_set)
            
            # Include
            # Check conflicts
            valid = True
            for j in current_set:
                # Check if there is a k in current_set such that (j, k) in conflict[idx]
                # This is O(N^2) per step, which is slow.
                # Optimization: maintain a set of "forbidden" pairs?
                # Or simply check: for all j, k in current_set, is (j, k) in conflict[idx]?
                # If current_set is small, this is fine.
                pass
            
            # Simpler check:
            for j in current_set:
                for k in current_set:
                    if j == k: continue
                    # Check if (j, k) is a conflict pair for idx
                    if (j, k) in conflict[idx] or (k, j) in conflict[idx]:
                         # Wait, conflict[idx] contains pairs (a, b) such that (idx, a, b) is a triple.
                         # So if (j, k) is in conflict[idx], then (idx, j, k) is a triple.
                         valid = False
                         break
                if not valid: break
            
            if valid:
                backtrack(idx + 1, current_size + 1, current_set + [idx])

        backtrack(0, 0, [])
        return best_size

    # Search Strategy:
    # 1. Start with random points.
    # 2. Ensure no 4 collinear.
    # 3. Use local search to minimize exact_max_no_three_collinear.
    
    best_score = 20
    best_points = None
    
    # Initial candidates
    # Parabola
    pts_parabola = [[x, x*x] for x in range(-10, 10)]
    # Check validity
    if not has_four_collinear(pts_parabola):
        s = exact_max_no_three_collinear(pts_parabola)
        if s < best_score:
            best_score = s
            best_points = pts_parabola
            
    # Random restarts with local search
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Generate random points
        points = []
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in points:
                points.append(p)
        
        # Local Search
        current_points = [list(p) for p in points]
        # Ensure no 4 collinear, else perturb
        while has_four_collinear(current_points):
             # Move a random point
             idx = random.randint(0, 19)
             current_points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
             # Simple loop might be slow if stuck, but random is usually fine
            
        current_score = exact_max_no_three_collinear(current_points)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try moving a random point
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            
            if new_p == old_p:
                continue
                
            # Check if new point causes 4 collinear with any 3 existing
            # We can check all 4-combinations involving the new point
            # Or just replace and check global 4-collinear (expensive)
            # Better: check if new_p is collinear with any pair (i, j) such that there is a k on that line?
            # Actually, just check if new_p makes a 4-collinear set.
            # A 4-collinear set must include new_p. So check all triples of existing points.
            # If new_p is collinear with a triple, it's a 4-collinear set.
            valid_move = True
            for i, j, k in itertools.combinations(range(20), 3):
                if i == idx or j == idx or k == idx: continue
                if collinear(tuple(new_p), tuple(current_points[i]), tuple(current_points[j])) and \
                   collinear(tuple(new_p), tuple(current_points[i]), tuple(current_points[k])):
                    valid_move = False
                    break
            
            if not valid_move:
                continue
                
            # Also check if new_p creates a 4-collinear with 2 existing and another existing?
            # The check above covers "new_p is collinear with 3 existing".
            # What if new_p is collinear with 2 existing, and those 2 are collinear with a 3rd existing?
            # That's the same case.
            # What if new_p is collinear with 1 existing, and that existing is collinear with 2 others?
            # No, 4 collinear means all 4 on one line.
            # So if new_p is on a line with 3 existing points, it's invalid.
            
            # Apply move
            current_points[idx] = new_p
            
            new_score = exact_max_no_three_collinear(current_points)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
            else:
                # Revert
                current_points[idx] = old_p
                
    return best_points

# Run construct
print(construct())
