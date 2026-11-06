"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Construct 20 distinct integer points with |x|,|y| <= 100 and no four collinear.
    Minimize the largest subset with no three collinear.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    # Precompute candidate pool of integer points
    # We need a diverse pool. Let's use points in [-50, 50]^2 but that's 10201 points.
    # We need to pick 20 from a pool.
    # A good pool: all integer points in [-10, 10]^2? That's 441 points.
    # Or maybe a bit larger to allow better optimization.
    # Let's use [-20, 20]^2 = 1681 points.
    
    min_coord = -20
    max_coord = 20
    pool = []
    for x in range(min_coord, max_coord + 1):
        for y in range(min_coord, max_coord + 1):
            pool.append((x, y))
    
    # Helper: check if a set of points has no 4 collinear
    def no_four_collinear(points):
        n = len(points)
        # Use sets of normalized line keys
        # For each pair, compute the line they define (normalized)
        # If any line has > 2 points, return False (actually we need > 3 to be invalid? No, "no four collinear" means max collinear is 3. So if a line has 4, it's invalid.)
        # Wait, the task says "no four collinear". This implies that 3 collinear is allowed, but 4 is not.
        # So we check if any line contains 4 or more points.
        
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize (dx, dy)
                g = abs(gcd(dx, dy)) if (dx != 0 or dy != 0) else 1
                if g > 0:
                    dx //= g
                    dy //= g
                else:
                    dx, dy = 0, 0
                # Ensure unique representation for the line
                # The line is defined by direction (dx, dy) and a point (x, y)
                # A better way: line equation ax + by + c = 0
                # a = dy, b = -dx
                # c = -(a*x1 + b*y1)
                a = dy
                b = -dx
                c = -(a * x1 + b * y1)
                # Normalize (a, b, c)
                g = abs(gcd(gcd(a, b), c)) if (a != 0 or b != 0 or c != 0) else 1
                if g > 0:
                    a //= g
                    b //= g
                    c //= g
                # Ensure sign consistency: first non-zero of (a,b,c) should be positive
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        for key, pts in lines.items():
            if len(pts) >= 4:
                return False
        return True

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)

    # Helper: count collinear triples in a set of points
    def count_triples(points):
        n = len(points)
        triples = 0
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(dx, dy)
                if g > 0:
                    dx //= g
                    dy //= g
                a = dy
                b = -dx
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(a, b), c)
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        for key, pts in lines.items():
            k = len(pts)
            if k >= 3:
                triples += k * (k-1) * (k-2) // 6
        return triples

    # Helper: calculate the score (largest subset with no 3 collinear)
    # This is the independence number of the 3-uniform hypergraph of collinear triples.
    # We use a simple greedy + local search for this, or exact B&B if small.
    # Since n=20, we can use a bitset B&B.
    
    def max_independent_set(points):
        n = len(points)
        # Precompute for each point, the set of points it is collinear with (in pairs? No, triples matter).
        # Actually, the constraint is "no three collinear".
        # This is a 3-uniform hypergraph independence problem.
        # Hyperedges are the sets of 3 collinear points.
        
        # Precompute all collinear triples
        triples = []
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(dx, dy)
                if g > 0:
                    dx //= g
                    dy //= g
                a = dy
                b = -dx
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(a, b), c)
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        for key, pts in lines.items():
            k = len(pts)
            if k >= 3:
                # Add all triples from this line
                # We only need to add each triple once.
                # The triples are combinations of 3 from pts.
                # But we should only add them if they are not already added?
                # Since lines are disjoint in terms of "maximal" sets? No, a point can be in multiple lines.
                # However, a triple of points is collinear if and only if they are all on the same line.
                # So we can just iterate over all lines and add all C(k,3) triples.
                # We need to be careful not to add duplicates if a line is processed multiple times?
                # The lines dict keys are unique line equations. So each line is processed once.
                # But can two different line equations define the same set of 3 points? No.
                # So we can just collect all triples.
                pass
        
        # Let's re-evaluate the B&B approach.
        # We want to find the largest subset S such that no 3 points in S are collinear.
        # This is equivalent to: for every line L, |S ∩ L| <= 2.
        
        # Bitmask B&B
        # State: mask of included points, mask of available points.
        # We prune if popcount(included) + popcount(available) <= best.
        
        # Precompute for each point i, the set of points that form a triple with i and another point?
        # No, simpler: for each point i, and each other point j, if i and j are on a line with > 2 points,
        # then we cannot include any third point from that line.
        
        # Let's define conflict[i] as a bitmask of points that conflict with i?
        # No, conflict is 3-way.
        # If we include i and j, we must exclude all k such that i,j,k are collinear.
        
        # Let's precompute:
        # line_mask[i][j] = bitmask of points k such that i, j, k are collinear.
        # Actually, if i and j are collinear with a set of points P, then if we include i and j,
        # we cannot include any k in P \ {i, j}.
        
        # Let's precompute for each pair (i, j), the set of points on the line through i and j.
        # pair_line_mask[i][j]
        
        n = len(points)
        pair_line_mask = [[0] * n for _ in range(n)]
        
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(dx, dy)
                if g > 0:
                    dx //= g
                    dy //= g
                a = dy
                b = -dx
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(a, b), c)
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        for key, pts in lines.items():
            mask = 0
            for p in pts:
                mask |= (1 << p)
            # For all pairs in pts, set pair_line_mask[i][j] = mask
            pts_list = list(pts)
            for i in range(len(pts_list)):
                for j in range(i+1, len(pts_list)):
                    pair_line_mask[pts_list[i]][pts_list[j]] = mask
                    pair_line_mask[pts_list[j]][pts_list[i]] = mask
        
        best = 0
        best_mask = 0
        
        def bnb(included_mask, available_mask, depth):
            nonlocal best, best_mask
            # Pruning
            if bin(included_mask).count('1') + bin(available_mask).count('1') <= best:
                return
            if available_mask == 0:
                if bin(included_mask).count('1') > best:
                    best = bin(included_mask).count('1')
                    best_mask = included_mask
                return
            
            # Choose a point from available_mask
            # Heuristic: choose the point with the most conflicts (highest degree in the hypergraph)
            # Or just the lowest bit.
            # Let's try lowest bit for simplicity first, or highest conflict.
            
            # Find a point i in available_mask
            # To speed up, we can choose the point that removes the most options.
            # For now, let's just pick the lowest set bit.
            i_bit = available_mask & -available_mask
            i = i_bit.bit_length() - 1
            
            # Option 1: Exclude i
            bnb(included_mask, available_mask ^ i_bit, depth + 1)
            
            # Option 2: Include i
            # If we include i, we must remove from available_mask any point j such that
            # there exists a k in available_mask such that i, j, k are collinear?
            # No, the constraint is that no 3 points in the final set are collinear.
            # If we include i, and we already have some points in included_mask, say j,
            # then we must exclude all k such that i, j, k are collinear.
            
            # Let's refine the state to track included points.
            # Actually, the standard B&B for independent set in hypergraphs is tricky.
            # Let's use a simpler recursive formulation:
            # f(S) = max independent set in S.
            # f(S) = max(f(S \ {i}), 1 + f(S \ {i} \ N(i))) where N(i) is the set of points that conflict with i.
            # But conflict is 3-way.
            
            # If we include i, we can still include j if j doesn't form a triple with i and any other included point.
            # This depends on the current included set.
            
            # Let's pass the included set as a bitmask.
            # When we include i, we must remove from available_mask all points j such that
            # there is a k in included_mask such that i, j, k are collinear.
            # Also, if we include i and j, we must remove all k such that i, j, k are collinear.
            
            # So, if we include i:
            # new_available = available_mask ^ i_bit
            # For each j in new_available:
            #   If there exists k in included_mask such that i, j, k are collinear, remove j.
            #   This is equivalent to: for each k in included_mask, remove pair_line_mask[i][k] & new_available.
            
            conflict_mask = 0
            inc_list = [k for k in range(n) if (included_mask >> k) & 1]
            for k in inc_list:
                conflict_mask |= pair_line_mask[i][k]
            
            new_available = (available_mask ^ i_bit) & ~conflict_mask
            
            bnb(included_mask | i_bit, new_available, depth + 1)
        
        bnb(0, (1 << n) - 1, 0)
        return best

    # Initial random solution
    random.seed(42)
    current_points = random.sample(pool, 20)
    if not no_four_collinear(current_points):
        # Retry until valid
        attempts = 0
        while not no_four_collinear(current_points) and attempts < 100:
            current_points = random.sample(pool, 20)
            attempts += 1
    
    current_score = max_independent_set(current_points)
    best_points = current_points
    best_score = current_score
    
    # Local Search
    # Move: Replace one point with another from the pool.
    # Check no_four_collinear.
    # Calculate new score.
    
    pool_set = set(pool)
    
    while time.time() - start_time < DEADLINE:
        improved = False
        # Randomly pick a point to replace
        idx = random.randint(0, 19)
        old_point = current_points[idx]
        
        # Try to replace with a random point from pool
        # To be efficient, we can sample a few candidates
        for _ in range(20):
            if time.time() - start_time > DEADLINE:
                break
            new_point = random.choice(pool)
            if new_point == old_point:
                continue
            
            new_points = current_points.copy()
            new_points[idx] = new_point
            
            # Check distinctness
            if len(set(new_points)) != 20:
                continue
            
            # Check no four collinear
            if not no_four_collinear(new_points):
                continue
            
            new_score = max_independent_set(new_points)
            
            if new_score < current_score:
                current_points = new_points
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points
                break # Accept and restart search from new state
        
        if not improved:
            # Try swapping two points
            i, j = random.sample(range(20), 2)
            p1, p2 = current_points[i], current_points[j]
            for _ in range(20):
                if time.time() - start_time > DEADLINE:
                    break
                n1 = random.choice(pool)
                n2 = random.choice(pool)
                if n1 == p1 or n2 == p2 or n1 == n2:
                    continue
                
                new_points = current_points.copy()
                new_points[i] = n1
                new_points[j] = n2
                
                if len(set(new_points)) != 20:
                    continue
                
                if not no_four_collinear(new_points):
                    continue
                
                new_score = max_independent_set(new_points)
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_points = current_points
                    break

    return [list(p) for p in best_points]

# EVOLVE-BLOCK-END
