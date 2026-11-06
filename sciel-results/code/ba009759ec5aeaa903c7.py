import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Helper functions
    def collinear(p1, p2, p3):
        return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) == 0

    def has_four_collinear(points):
        n = len(points)
        # Check all subsets of 4
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if collinear(points[i], points[j], points[k]) and collinear(points[i], points[j], points[l]):
                            return True
        return False

    def get_triples(points):
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def max_no_three_collinear(points):
        triples = get_triples(points)
        n = len(points)
        
        # Bitmask representation for subsets
        # We want the largest subset S such that no triple (i,j,k) is fully contained in S.
        # This is equivalent to: for every triple, at least one point is NOT in S.
        
        # Branch and Bound
        best_size = 0
        
        # Precompute for each point, which triples it is in
        point_triples = [[] for _ in range(n)]
        for t in triples:
            for p in t:
                point_triples[p].append(t)
                
        # Sort points by number of triples (most constrained first)
        order = sorted(range(n), key=lambda x: len(point_triples[x]), reverse=True)
        
        # We build the subset S.
        # State: current index in order, current subset mask, current size
        # If adding point p to S creates a full triple, we cannot add it.
        
        def dfs(idx, mask, size):
            nonlocal best_size
            if idx == n:
                if size > best_size:
                    best_size = size
                return
            
            # Pruning: even if we add all remaining points, can we beat best?
            if size + (n - idx) <= best_size:
                return
                
            p = order[idx]
            
            # Option 1: Exclude p
            dfs(idx + 1, mask, size)
            
            # Option 2: Include p
            # Check if adding p completes any triple
            valid = True
            for t in point_triples[p]:
                i, j, k = t
                # Check if i, j, k are all in mask (excluding p, since p is being added)
                # Wait, mask contains points added so far.
                # If i, j are in mask, and k is p, then adding p completes it.
                # We need to check if the other two points of any triple containing p are in mask.
                # Actually, simpler: check all triples containing p.
                # If the other two points are in mask, it's invalid.
                pass # Logic below is better

            # Refined Check:
            # A triple is invalid if all 3 are in S.
            # When we add p, we only need to check triples containing p.
            # For each such triple (p, a, b), if a and b are in mask, then adding p is invalid.
            
            is_valid = True
            for t in point_triples[p]:
                other_points = [x for x in t if x != p]
                a, b = other_points
                if (mask & (1 << a)) and (mask & (1 << b)):
                    is_valid = False
                    break
            
            if is_valid:
                dfs(idx + 1, mask | (1 << p), size + 1)

        dfs(0, 0, 0)
        return best_size

    def random_valid_config():
        while True:
            pts = []
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p in pts:
                    continue
                # Check 4-collinear with existing
                valid = True
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        if collinear(p, pts[i], pts[j]):
                            # Check if there is a 3rd existing point on this line
                            # Actually, if p, i, j are collinear, we need to check if any other existing point k is also on this line.
                            # If so, we have 4.
                            # Simpler: just check all existing pairs with p? No, that's O(N^2).
                            # Since N is small, just check if p creates a 4-set.
                            # For each pair (a,b) in pts, if p, a, b collinear, check if any c in pts is also collinear with a,b.
                            pass
                # This check is getting complex. Let's use a simpler validity check on the whole set.
                pts.append(p)
                if has_four_collinear(pts):
                    pts.pop()
                    break
            if len(pts) == 20 and not has_four_collinear(pts):
                return pts

    # Initial configuration
    # Try to find a good one quickly
    best_pts = None
    best_score = 20
    
    for _ in range(5):
        pts = random_valid_config()
        if pts:
            s = max_no_three_collinear(pts)
            if s < best_score:
                best_score = s
                best_pts = pts[:]
        
        if time.time() > deadline:
            break

    if not best_pts:
        # Fallback: Parabola
        return [[x, x * x] for x in range(-10, 10)]

    current_pts = best_pts[:]
    current_score = best_score
    
    # Local Search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_pts[idx]
        
        # Generate candidate
        # Strategy: Find a point that forms a triple with two existing points
        # Pick two other points
        i, j = random.sample([x for x in range(20) if x != idx], 2)
        p1 = current_pts[i]
        p2 = current_pts[j]
        
        # Find integer points on line p1-p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = __import__('math').gcd(abs(dx), abs(dy))
        if g == 0: continue
        
        step_x = dx // g
        step_y = dy // g
        
        # Points are p1 + k * (step_x, step_y)
        # We want k such that the point is in [-100, 100] and not already in set
        # Also, we want to avoid creating 4-collinear.
        
        # Try a few random k values
        found_candidate = False
        for _ in range(10):
            k = random.randint(-50, 50)
            nx = p1[0] + k * step_x
            ny = p1[1] + k * step_y
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                new_p = (nx, ny)
                if new_p in current_pts:
                    continue
                
                # Check validity
                test_pts = current_pts[:]
                test_pts[idx] = new_p
                if not has_four_collinear(test_pts):
                    # Check if it's a new triple (it is by construction, but might be redundant)
                    # Calculate score
                    s = max_no_three_collinear(test_pts)
                    if s < current_score:
                        current_pts = test_pts
                        current_score = s
                        found_candidate = True
                        break
                    elif s == current_score:
                        # Accept with some probability to escape local minima? 
                        # Or just reject to keep it simple.
                        pass
        
        # Also try random moves
        if not found_candidate and random.random() < 0.2:
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            new_p = (nx, ny)
            if new_p not in current_pts:
                test_pts = current_pts[:]
                test_pts[idx] = new_p
                if not has_four_collinear(test_pts):
                    s = max_no_three_collinear(test_pts)
                    if s < current_score:
                        current_pts = test_pts
                        current_score = s

    return [[p[0], p[1]] for p in current_pts]
# EVOLVE-BLOCK-END
