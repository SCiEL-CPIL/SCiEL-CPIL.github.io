import random
from itertools import combinations

def construct():
    random.seed(42)
    
    # Pool of points
    pool = [[x, y] for x in range(-10, 10) for y in range(-10, 10)]
    
    def is_valid_4(pts):
        n = len(pts)
        # Check all 4-subsets? No, check all lines.
        # Faster: For each pair, count points on line.
        # But N=20, so O(N^4) is 11400 * 20 = 200k ops. Fine.
        # Or simpler: check all 4-tuples for collinearity?
        # Collinearity of 4 points: any 3 are collinear.
        # So just check if any 3 are collinear and form a line with a 4th?
        # Actually, just check if any line has >= 4 points.
        
        # Get all unique lines defined by pairs
        lines = set()
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = pts[i], pts[j]
                # Normalize line equation
                A = p1[1] - p2[1]
                B = p2[0] - p1[0]
                C = p1[0]*p2[1] - p2[0]*p1[1]
                # Normalize (A, B, C)
                from math import gcd
                g = gcd(gcd(A, B), C)
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                # Ensure sign consistency
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A, B, C = -A, -B, -C
                lines.add((A, B, C))
        
        for line in lines:
            A, B, C = line
            count = 0
            for p in pts:
                if A*p[0] + B*p[1] + C == 0:
                    count += 1
            if count >= 4:
                return False
        return True

    def get_triples(pts):
        triples = []
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    if (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) == 0):
                        # Check if exactly 3? No, any triple counts for the "no 3 collinear" constraint.
                        # If 4 are collinear, they contain 4 triples. But we ensure no 4 collinear.
                        # So all triples are exactly 3.
                        triples.append((i, j, k))
        return triples

    def min_hitting_set_size(n_points, triples):
        # Branch and bound to find min points to remove to hit all triples
        # n_points = 20
        # triples = list of (i, j, k)
        
        # If no triples, return 0
        if not triples:
            return 0
            
        # Sort triples by length (all 3)
        # Use recursion
        best = n_points
        
        def backtrack(index, removed, current_size):
            nonlocal best
            if current_size >= best:
                return
            if index == len(triples):
                best = current_size
                return
                
            # If remaining triples can be covered by current_size, prune?
            # Simple pruning: if current_size + 0 >= best, return.
            
            t = triples[index]
            
            # Option 1: Remove one of the points in this triple
            # To reduce branching, we can pick the point that appears in the most remaining triples?
            # For simplicity, just try removing i, j, or k.
            
            # Optimization: If a point is in many triples, removing it is better.
            # Let's just try the 3 options.
            
            # Remove i
            new_removed = removed | (1 << t[0])
            # Filter triples? No, just check if covered.
            # We need to check if all triples from index onwards are covered by new_removed.
            # This is expensive.
            
            # Better approach:
            # The hitting set problem on 20 elements with ~1000 constraints.
            # We can use a simple greedy heuristic for the score estimation in the search loop,
            # and exact only for the final candidates.
            # But we need exact score to compare.
            
            # Let's use a different exact method for N=20.
            # Iterate over all subsets of size k?
            # No, let's stick to B&B.
            
            pass
        
        # This B&B might be slow if called many times.
        # Alternative: Approximate score for search, exact for final.
        # Score approx = 20 - (number of triples) / 3 ? No.
        
        # Let's just return a large number if we can't compute fast, 
        # or use a greedy hitting set size as a proxy.
        # Greedy Hitting Set:
        # Count frequency of each point in triples.
        # Pick point with max freq, remove it, remove all triples containing it.
        # Repeat.
        
        triples_set = [list(t) for t in triples]
        point_freq = [0]*n_points
        for t in triples_set:
            for p in t:
                point_freq[p] += 1
                
        removed = 0
        count = 0
        active_triples = triples_set[:]
        
        while active_triples:
            # Find point in active_triples with max freq
            # Recalculate freq for active
            freq = [0]*n_points
            for t in active_triples:
                for p in t:
                    freq[p] += 1
            max_p = -1
            max_f = -1
            for p in range(n_points):
                if freq[p] > max_f:
                    max_f = freq[p]
                    max_p = p
            if max_f == 0:
                break
            removed |= (1 << max_p)
            count += 1
            active_triples = [t for t in active_triples if max_p not in t]
            
        return count

    def score(pts):
        triples = get_triples(pts)
        return 20 - min_hitting_set_size(20, triples)

    # Initial set: Random valid set
    points = []
    for _ in range(20):
        while True:
            p = random.choice(pool)
            if p not in points:
                points.append(p)
                if is_valid_4(points):
                    break
                else:
                    points.pop()
                    # Try again
                    continue
            else:
                continue

    # Local Search
    current_score = score(points)
    best_points = points[:]
    best_score = current_score
    
    for iteration in range(100): # Limit iterations
        improved = False
        # Try all swaps
        for i in range(20):
            for q in pool:
                if q in points:
                    continue
                new_points = points[:]
                new_points[i] = q
                if is_valid_4(new_points):
                    s = score(new_points)
                    if s < best_score:
                        best_score = s
                        best_points = new_points[:]
                        improved = True
                        # Update current for next swaps?
                        # Or just continue from best?
                        # Greedy local search usually updates immediately.
                        points = best_points[:]
                        current_score = s
                        break
            if improved:
                break
        if not improved:
            break

    return best_points
