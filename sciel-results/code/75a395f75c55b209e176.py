import time
import random
from itertools import combinations
from math import gcd

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        # Normalize
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
            
        # Fix sign
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
            
    ordinary = 0
    for key, pair_count in lines.items():
        if pair_count == 1:
            ordinary += 1
            
    return ordinary

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    timeout = 110 # Safety margin for 120s limit
    
    # Base: 3x3 Grid
    base_points = [[x, y] for x in range(3) for y in range(3)]
    
    # Identify ordinary lines in base_points
    def get_ordinary_lines(pts):
        n = len(pts)
        lines = {}
        
        def get_line_key(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            return (A, B, C)

        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key not in lines:
                    lines[key] = []
                lines[key].append((i, j))
        
        ordinary_lines = []
        for key, pairs in lines.items():
            if len(pairs) == 1:
                i, j = pairs[0]
                ordinary_lines.append((pts[i], pts[j]))
        return ordinary_lines

    ord_lines = get_ordinary_lines(base_points)
    
    # Generate candidate points
    # For each ordinary line (p1, p2), find integer points p3 collinear with p1, p2
    # p3 = p1 + k * (p2 - p1)
    candidates = set()
    for p1, p2 in ord_lines:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        # Try small multiples
        for k in [-2, -1, 1, 2, 3, -3]:
            if k == 0: continue
            # We want a point distinct from p1 and p2
            # If k=1, it's p2. If k=0, it's p1.
            # We want p3 such that {p1, p2, p3} are distinct.
            # p3 = p1 + k*(p2-p1)
            # If k=2, p3 = 2p2 - p1
            # If k=-1, p3 = 2p1 - p2
            px = x1 + k * dx
            py = y1 + k * dy
            if abs(px) <= 10**9 and abs(py) <= 10**9:
                # Check if it's distinct from existing base points
                if [px, py] not in base_points:
                    candidates.add((px, py))
                    
    # Convert to list for easy access
    candidate_list = list(candidates)
    
    # We need to add 3 points to the base 9 points.
    # Total 12 points.
    # We want to minimize ordinary lines.
    
    best_score = count_ordinary_lines(base_points) # This is for 9 points, not comparable directly to 12, but we need 12.
    # Actually, we must return 12 points.
    # Let's find the best 3 candidates to add.
    
    best_points = None
    best_ordinary_count = float('inf')
    
    # If candidate list is too large, we might need to prune, but 12 ordinary lines * ~6 candidates = ~72 candidates.
    # C(72, 3) is ~ 60k. Very fast.
    
    # To be safe with time, we can iterate through combinations
    # But first, let's try a greedy approach to find a good initial set, then local search.
    
    # Greedy: Pick candidate that reduces ordinary lines the most.
    # However, evaluating the full count for each candidate is O(N^2).
    # N=12. 144 pairs. Very fast.
    
    # Let's just iterate all combinations of 3 from candidate_list
    # If candidate_list is huge, we sample.
    # But it should be small.
    
    if len(candidate_list) > 50:
        # Random sampling if too many
        random.shuffle(candidate_list)
        candidate_list = candidate_list[:50]
        
    for i in range(len(candidate_list)):
        if time.time() - start_time > timeout:
            break
        for j in range(i + 1, len(candidate_list)):
            if time.time() - start_time > timeout:
                break
            for k in range(j + 1, len(candidate_list)):
                new_points = base_points + [list(candidate_list[i]), list(candidate_list[j]), list(candidate_list[k])]
                
                # Check distinctness
                if len(set(map(tuple, new_points))) != 12:
                    continue
                    
                score = count_ordinary_lines(new_points)
                if score < best_ordinary_count:
                    best_ordinary_count = score
                    best_points = new_points
                    
    if best_points is None:
        # Fallback: just add 3 random points from a wider grid
        best_points = base_points + [[3,0], [0,3], [3,3]]
        
    # Local Search / Refinement
    # Try to swap one of the added points with another candidate to improve score
    if best_points is not None and time.time() - start_time < timeout - 10:
        current_score = best_ordinary_count
        improved = True
        while improved and time.time() - start_time < timeout - 10:
            improved = False
            # Generate a larger pool of candidates for swapping
            # Include points near the current points
            swap_candidates = set()
            for pt in best_points:
                x, y = pt
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        nx, ny = x + dx, y + dy
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            swap_candidates.add((nx, ny))
            
            # Remove current points from candidates
            for pt in best_points:
                swap_candidates.discard(tuple(pt))
                
            swap_candidates = list(swap_candidates)
            if not swap_candidates:
                break
                
            # Try swapping each of the 3 added points (indices 9, 10, 11)
            # We don't know which are the "added" ones if we modified base, but we know base_points
            # Let's just try swapping any point with any candidate? No, that's expensive.
            # Let's stick to the 3 points we added.
            # Actually, we can just try to replace any point in best_points with a candidate.
            # But 12 * |candidates| evaluations.
            
            # Simpler: Just re-evaluate the best combination found.
            # The combination search above is likely sufficient for a small N.
            
            # Let's do a quick local search on the added points only.
            added_indices = [9, 10, 11]
            for idx in added_indices:
                old_pt = best_points[idx]
                for cand in swap_candidates:
                    if tuple(cand) == tuple(old_pt):
                        continue
                    new_pt_list = best_points.copy()
                    new_pt_list[idx] = list(cand)
                    if len(set(map(tuple, new_pt_list))) != 12:
                        continue
                    new_score = count_ordinary_lines(new_pt_list)
                    if new_score < current_score:
                        current_score = new_score
                        best_points = new_pt_list
                        improved = True
                        break # Restart search with new best
                if improved:
                    break
                    
    return best_points

# EVOLVE-BLOCK-END
