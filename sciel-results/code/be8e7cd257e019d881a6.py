import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin for 150s limit
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def get_line_id(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(points):
        n = len(points)
        # Map line -> count
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i+1, n):
                lid = get_line_id(points[i], points[j])
                line_counts[lid] += 1
                # If we have already seen this line with 2 pairs, adding a 3rd pair means 4 points?
                # No, 3 points on a line have 3 pairs. 4 points have 6 pairs.
                # If line_counts[lid] >= 3, we have at least 3 pairs, which means at least 3 points?
                # Actually, if we have 3 pairs, the points could be {A,B}, {A,C}, {B,C} -> 3 points.
                # If we have 4 pairs, it must be 4 points? No, 3 points have 3 pairs.
                # So if count >= 4, we definitely have >= 4 points?
                # Let's just check if any line has >= 4 points by collecting points.
                pass
        
        # More robust check for 4 collinear
        lines = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                lid = get_line_id(points[i], points[j])
                lines[lid].add(i)
                lines[lid].add(j)
        
        for lid, pts_in_line in lines.items():
            if len(pts_in_line) >= 4:
                return True
        return False

    def max_no_3_collinear(points):
        """
        Exact branch and bound using bitmasks for speed.
        """
        n = len(points)
        if n <= 2:
            return n
            
        # Precompute conflicts: conflict[i] is a bitmask of points j such that 
        # adding i to a set containing j AND k (where i,j,k collinear) is invalid.
        # Actually, simpler: 
        # For each pair (i, j), let collinear_with_ij be the set of k such that i,j,k are collinear.
        # When adding point i to a current set S, it is invalid if there exist j, k in S 
        # such that i, j, k are collinear.
        # This is equivalent to: for each j in S, let L be the line(i, j). If there is any k in S 
        # on L (k != i, k != j), then invalid.
        
        # Precompute for each i, a list of (j, mask_of_k_collinear_with_i_j)
        # But masks are easier if we precompute for each i, a dict: line_id -> mask of points on that line.
        
        # Let's precompute:
        # lines_through_i[i] = dict { line_id: bitmask of points on this line (including i) }
        lines_through = [defaultdict(int) for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                lid = get_line_id(points[i], points[j])
                # We need to know which points are on this line.
                # We can compute this once for all lines.
                pass
                
        # Let's compute all lines first
        all_lines = defaultdict(int) # lid -> bitmask of points
        for i in range(n):
            for j in range(i + 1, n):
                lid = get_line_id(points[i], points[j])
                all_lines[lid] |= (1 << i)
                all_lines[lid] |= (1 << j)
                
        # Now for each point i, and each line containing i, we know the mask.
        # To check if i can be added to S (bitmask S):
        # For every j in S:
        #   Let lid = line(i, j)
        #   Let M = all_lines[lid]
        #   If (M & S) has at least 2 bits set (j and some k), then invalid.
        #   Note: i is not in S yet.
        
        # Precompute line_id for each pair to avoid recomputing
        line_id_pair = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    line_id_pair[i][j] = get_line_id(points[i], points[j])
                    
        best = 0
        full_mask = (1 << n) - 1
        
        # Order points by degree of collinearity? 
        # Points involved in more triples are "harder" to place. 
        # Let's just use index order, but maybe sorting by "potential conflict" helps pruning.
        # Conflict degree: number of other points that form a triple with i and some third point.
        
        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = bin(S_mask).count('1')
            
            # Pruning
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip start_idx
            backtrack(S_mask, start_idx + 1)
            
            # Option 2: Include start_idx
            i = start_idx
            valid = True
            # Check conflicts with points already in S
            # Iterate through set bits in S_mask
            temp_mask = S_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                j = lsb.bit_length() - 1
                lid = line_id_pair[i][j]
                M = all_lines[lid]
                # Check if any other point k in S is on this line
                # M contains i, j, and potentially others.
                # We want to know if (M & S_mask) has more than 1 bit (just j).
                # Since i is not in S_mask, (M & S_mask) contains j and any k.
                # If popcount(M & S_mask) > 1, invalid.
                if (M & S_mask).bit_count() > 1:
                    valid = False
                    break
                temp_mask ^= lsb
                
            if valid:
                backtrack(S_mask | (1 << i), start_idx + 1)
                
        backtrack(0, 0)
        return best

    def local_search(points):
        """
        Try to improve score by swapping points or moving points.
        """
        current_score = max_no_3_collinear(points)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try moving each point to a random new location
            for i in range(len(points)):
                orig = points[i]
                for _ in range(5):
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    new_p = (new_x, new_y)
                    if new_p in points:
                        continue
                    
                    points[i] = new_p
                    if has_4_collinear(points):
                        points[i] = orig
                        continue
                    
                    new_score = max_no_3_collinear(points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        points[i] = orig
                        
        return points, current_score

    # Strategy:
    # 1. Generate initial configurations based on structured patterns.
    # 2. Refine with local search.
    
    best_config = None
    best_score = 21
    
    # Pattern 1: 5x5 grid subset (20 points)
    # The 5x5 grid has many collinearities.
    # Let's try to pick 20 points from a 5x5 grid such that no 4 are collinear.
    # In a 5x5 grid, rows/cols have 5 points. We need to break them into max 3.
    # This is hard to do manually, so let's use a "random subset of grid" approach.
    
    def generate_grid_subset():
        grid_points = [(x, y) for x in range(5) for y in range(5)]
        random.shuffle(grid_points)
        # We need 20 points. 5x5=25.
        # If we take all 25, we have 5 collinear in rows/cols.
        # We must remove points to break 4-collinear.
        # Start with all 25, then remove 5 to break all 4-collinear lines?
        # Or just pick 20 randomly and check.
        pts = []
        attempts = 0
        while len(pts) < 20 and attempts < 1000:
            p = random.choice(grid_points)
            if p in pts:
                continue
            new_pts = pts + [p]
            if not has_4_collinear(new_pts):
                pts.append(p)
            else:
                # Backtrack? No, just try another.
                # This greedy might fail.
                pass
            attempts += 1
        
        # If greedy failed, try random permutation
        if len(pts) < 20:
            random.shuffle(grid_points)
            pts = []
            for p in grid_points:
                if len(pts) == 20: break
                new_pts = pts + [p]
                if not has_4_collinear(new_pts):
                    pts.append(p)
        
        if len(pts) == 20:
            # Scale to fit |x|,|y| <= 100? Grid is 0..4, so it fits.
            return [(x, y) for x, y in pts]
        return None

    # Pattern 2: Random points biased towards lines
    def generate_biased_random():
        pts = []
        # Pick a few "seed" lines
        lines = []
        for _ in range(6):
            m = random.choice([-2, -1, 0, 1, 2])
            c = random.randint(-10, 10)
            lines.append((m, c))
            
        candidates = set()
        for m, c in lines:
            for x in range(-10, 11):
                y = m * x + c
                if abs(y) <= 100:
                    candidates.add((x, y))
                    
        random.shuffle(list(candidates))
        for p in candidates:
            if len(pts) == 20: break
            if p in pts: continue
            new_pts = pts + [p]
            if not has_4_collinear(new_pts):
                pts.append(p)
                
        # Fill rest randomly
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p in pts: continue
            new_pts = pts + [p]
            if not has_4_collinear(new_pts):
                pts.append(p)
                
        if len(pts) == 20:
            return pts
        return None

    # Run multiple strategies
    for i in range(50):
        if time.time() > deadline - 20:
            break
            
        if i % 2 == 0:
            pts = generate_grid_subset()
        else:
            pts = generate_biased_random()
            
        if pts is None or len(pts) != 20:
            continue
            
        # Validate
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
        if len(set(pts)) != 20:
            continue
        if has_4_collinear(pts):
            continue
            
        # Refine with local search
        refined_pts, score = local_search(pts)
        
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in refined_pts]
            
        if best_score <= 10:
            break

    if best_config is None:
        # Fallback: Random valid config
        while True:
            pts = []
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p in pts: continue
                new_pts = pts + [p]
                if not has_4_collinear(new_pts):
                    pts.append(p)
            if len(pts) == 20:
                return [list(p) for p in pts]
                
    return best_config

# EVOLVE-BLOCK-END
