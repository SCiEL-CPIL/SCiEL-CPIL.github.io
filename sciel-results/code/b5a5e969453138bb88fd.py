import time
import random
import math

def construct():
    start_time = time.time()
    deadline = time.time() + 145  # 5s margin for safety
    
    # Grid dimensions for compact search space
    # A 31x31 grid provides enough variety for 20 points while keeping neighborhood search fast
    GRID = 31
    HALF = GRID // 2  # 15
    # Coordinates range from -15 to 15
    
    # Precompute all points in the grid
    all_pts = []
    for x in range(-HALF, HALF + 1):
        for y in range(-HALF, HALF + 1):
            all_pts.append((x, y))
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    # Helper: Check if any 4 points are collinear in a list of points
    # Optimized: Group points by lines defined by pairs? 
    # For n=20, O(n^4) is 116280 checks. Fast enough for occasional checks.
    # But we call this frequently in SA. 
    # Better: For a given set of 20 points, check if any line has >3 points.
    # We can do this by iterating all pairs (i, j), finding the line, and counting points on it.
    # If any line has count >= 4, invalid.
    def has_4_collinear(points):
        n = len(points)
        if n < 4:
            return False
        # Use a set of normalized line representations to count points
        # Line representation: (A, B, C) for Ax + By + C = 0, normalized
        # To avoid floating point, use integers.
        # For points p1(x1,y1), p2(x2,y2):
        # A = y1 - y2, B = x2 - x1, C = x1*y2 - x2*y1
        # Normalize by GCD of A, B, C and sign
        
        from math import gcd
        
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                
                # Normalize
                g = gcd(abs(A), abs(B))
                if g != 0:
                    g = gcd(g, abs(C))
                    if g != 0:
                        A //= g
                        B //= g
                        C //= g
                    else:
                        # A or B was 0, handle carefully
                        if A == 0 and B == 0:
                            continue
                        if A == 0:
                            if B < 0:
                                A, B, C = -A, -B, -C
                        elif B < 0:
                            A, B, C = -A, -B, -C
                        else:
                            if A < 0:
                                A, B, C = -A, -B, -C
                else:
                    # A=0, B=0 is impossible for distinct points
                    pass
                
                # Ensure a consistent sign for the line representation
                # e.g., first non-zero of A, B is positive
                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C
                
                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
        
        # Check if any line has >= 4 points
        for pts_on_line in line_counts.values():
            if len(pts_on_line) >= 4:
                return True
        return False

    # Helper: Exact branch and bound for largest subset with no 3 collinear
    # This is the true objective function.
    def max_no_3_collinear(points):
        n = len(points)
        
        # Precompute collinearity masks
        # For each point i, and each pair (j, k) with j < k, 
        # bad_pairs[i] contains (j, k) if i, j, k are collinear.
        # Actually, we need: if we add i to S, does it form a triple with any two existing points in S?
        # So for each i, we need to know which pairs (j, k) in S would conflict.
        
        # Let's precompute a list of "conflict pairs" for each point i.
        # conflict[i] is a set of frozensets {j, k} such that i, j, k are collinear.
        
        conflict = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        conflict[i].add(frozenset((j, k)))
                        conflict[j].add(frozenset((i, k)))
                        conflict[k].add(frozenset((i, j)))
        
        best = 0
        
        # Memoization might be hard due to S_mask state, but n=20 is small.
        # Use simple backtracking with pruning.
        
        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = bin(S_mask).count('1')
            
            # Prune: even if we pick all remaining, we can't beat best
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip point start_idx
            backtrack(S_mask, start_idx + 1)
            
            # Option 2: Include point start_idx
            # Check validity: no pair (j, k) in S_mask such that {j, k} in conflict[start_idx]
            valid = True
            # Iterate over pairs in S_mask
            # Optimization: only check pairs (j, k) where j < k and both < start_idx?
            # No, S_mask can contain indices > start_idx? 
            # In standard backtracking, we decide in order 0..n-1.
            # So S_mask only contains bits < start_idx.
            
            for j in range(start_idx):
                if not (S_mask & (1 << j)):
                    continue
                for k in range(j + 1, start_idx):
                    if not (S_mask & (1 << k)):
                        continue
                    if frozenset((j, k)) in conflict[start_idx]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                backtrack(S_mask | (1 << start_idx), start_idx + 1)
                
        backtrack(0, 0)
        return best

    # Heuristic Score for Simulated Annealing
    # We want to MINIMIZE max_no_3_collinear.
    # Proxy: Maximize the number of "wasted" points in lines?
    # Or: Minimize the size of the largest "independent set" in the collinearity graph?
    # A simpler proxy: Count the number of lines with exactly 3 points. 
    # More lines with 3 points means more "constrained" points, potentially reducing the max independent set.
    # Let's try: Score = - (number of triples). 
    # Wait, we want to MINIMIZE the max subset size.
    # If we pack points into lines of 3, the max subset size might drop.
    # Let's use a heuristic: 
    # H = sum over all lines L with |L| >= 3 of (|L| - 2).
    # This rewards lines with many points.
    # Actually, just counting triples might be enough.
    
    def heuristic_score(points):
        # Count distinct lines with >= 3 points
        # Reuse logic from has_4_collinear but count lines
        n = len(points)
        from math import gcd
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                
                g = gcd(abs(A), abs(B))
                if g != 0:
                    g = gcd(g, abs(C))
                    if g != 0:
                        A //= g
                        B //= g
                        C //= g
                else:
                    pass
                
                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C
                
                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
        
        score = 0
        for pts_on_line in line_counts.values():
            if len(pts_on_line) >= 3:
                score += len(pts_on_line)
        
        return score

    # Simulated Annealing
    def simulated_annealing():
        # Initial state: Random 20 points from grid
        # Ensure no 4 collinear initially? 
        # Random points on a small grid might have 4 collinear.
        # We'll penalize invalid states heavily.
        
        current_pts = random.sample(all_pts, 20)
        
        # Check validity
        if has_4_collinear(current_pts):
             # Try to fix or just start with a valid one
             # Let's try a few random restarts to find a valid start
             for _ in range(100):
                 current_pts = random.sample(all_pts, 20)
                 if not has_4_collinear(current_pts):
                     break
        
        current_score = heuristic_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score
        best_exact_score = max_no_3_collinear(best_pts)
        
        T = 10.0
        T_min = 0.01
        alpha = 0.999 # Cooling rate
        
        # Precompute indices for faster access? 
        # We'll pick a random index and a random new point.
        
        steps = 0
        while T > T_min:
            if time.time() > deadline - 5:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            
            # Pick a new point from neighbors
            # Neighbors: points in the grid within Chebyshev distance 1? Or random from grid?
            # Random from grid is faster to implement and explores more.
            # But local moves are better for SA.
            # Let's pick a random point from the grid.
            new_p = random.choice(all_pts)
            
            if new_p == old_p:
                continue
            
            # Check for duplicates
            if new_p in current_pts:
                continue
            
            # Tentative move
            current_pts[idx] = new_p
            
            # Check validity (4-collinear)
            if has_4_collinear(current_pts):
                # Reject immediately or penalize?
                # Rejecting immediately keeps the search in the valid space, which is good.
                current_pts[idx] = old_p
                continue
            
            new_score = heuristic_score(current_pts)
            
            # Calculate delta
            # We want to maximize heuristic_score (more points on lines)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                
                # Update best if this is a good candidate
                # To save time, only compute exact score occasionally or if heuristic is very good
                # Let's compute exact score if heuristic improves the best heuristic, or every 100 steps
                if new_score > best_score or steps % 50 == 0:
                    exact = max_no_3_collinear(current_pts)
                    if exact < best_exact_score:
                        best_exact_score = exact
                        best_pts = current_pts[:]
                        best_score = new_score
                else:
                    # Just update best heuristic pointer
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = current_pts[:]
                        
            else:
                # Revert
                current_pts[idx] = old_p
                
            T *= alpha
            steps += 1
            
        return best_pts, best_exact_score

    # Run SA multiple times with different random seeds/starts
    best_global_pts = None
    best_global_score = 21
    
    for run in range(5):
        if time.time() > deadline - 10:
            break
            
        pts, score = simulated_annealing()
        
        if score < best_global_score:
            best_global_score = score
            best_global_pts = pts
            
        # If we find a very good score, we might stop early?
        # Theoretical lower bound for n=20? 
        # g(20) is likely around 8-10.
        # If we get <= 9, that's excellent.
        if best_global_score <= 9:
             break

    if best_global_pts is None:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
        
    return [list(p) for p in best_global_pts]

# Test harness (optional, for local debugging)
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Valid (no 4 collinear):", not has_4_collinear(pts))
    print("Score:", max_no_3_collinear(pts))
