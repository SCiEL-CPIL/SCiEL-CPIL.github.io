import time
import random
import sys

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _max_independent_set(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    This is the score we want to minimize.
    """
    n = len(points)
    
    # Precompute collinear pairs (edges in conflict graph)
    # For each pair (i, j), store the set of k such that i, j, k are collinear.
    collinear_with = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    collinear_with[i][j].add(k)
                    collinear_with[j][i].add(k)

    best = 0
    
    def backtrack(current_set, candidates):
        nonlocal best
        if len(current_set) + len(candidates) <= best:
            return
            
        if not candidates:
            if len(current_set) > best:
                best = len(current_set)
            return

        # Pick a candidate
        # Heuristic: Pick the one with the most conflicts?
        # To speed up, we pick the first one, but we can sort candidates by "degree" (number of conflicts)
        # For n=20, simple branching is fine if pruning is good.
        
        p_idx = candidates[0]
        rest_candidates = candidates[1:]
        
        # Branch 1: Include p_idx
        # New candidates: those in rest_candidates that are not collinear with p_idx and any point in current_set
        new_candidates = []
        for k in rest_candidates:
            can_add = True
            for c in current_set:
                if k in collinear_with[p_idx][c]:
                    can_add = False
                    break
            if can_add:
                new_candidates.append(k)
        
        backtrack(current_set + [p_idx], new_candidates)
        
        # Branch 2: Exclude p_idx
        backtrack(current_set, rest_candidates)

    backtrack([], list(range(n)))
    return best

def _greedy_upper_bound(points):
    """
    Fast greedy heuristic to estimate the maximum independent set size.
    This is not exact but much faster. We use it to guide the search.
    """
    n = len(points)
    # Precompute conflicts: conflict[i] is a set of j such that i, j, and some k form a triple?
    # No, the constraint is "no three collinear".
    # Greedy: repeatedly pick the point that is part of the fewest potential triples?
    # Or simply: pick a random point, remove it and all points that form a triple with it?
    # This is a maximal independent set in the 3-uniform hypergraph sense.
    
    # A simpler bound: 
    # If we have a set of points, the max independent set is at least n/3 (pigeonhole on lines?).
    # Let's just do a greedy construction of a valid set.
    # Start with empty set.
    # While candidates remain:
    #   Pick a point p from candidates.
    #   Add p to set.
    #   Remove from candidates any point q such that p, q, and some r in set form a triple?
    #   Actually, if we add p, we must ensure no triple is formed.
    #   If we have set S, and we add p, we must remove from candidates any q such that 
    #   there exists r in S such that p, q, r are collinear.
    
    candidates = list(range(n))
    current_set = []
    
    # Precompute collinear triples for fast lookup
    # triple_conflict[p][q] = set of r such that p, q, r are collinear
    # We can compute this on the fly or precompute.
    # For speed in the heuristic, let's precompute a global list of triples.
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j+1, n):
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    triples.append((i, j, k))
    
    # Map point to triples it's in
    point_to_triples = [set() for _ in range(n)]
    for t in triples:
        for idx in t:
            point_to_triples[idx].add(t)
            
    # Greedy:
    # While candidates:
    #   Pick p with min conflicts?
    #   Add p.
    #   For each q in candidates:
    #      If there is a triple (p, q, r) where r is in current_set, remove q.
    #      Wait, if r is in current_set, and p is added, then q cannot be added because p,q,r would be collinear.
    #      Also, if there is a triple (q, r, s) where r, s are in current_set, q is already removed?
    #      Actually, if current_set is valid, no two points in it form a triple with any other point?
    #      No. It's possible i, j in S and k not in S are collinear.
    #      If we add p, and there exists r in S such that p, r, q are collinear, then q cannot be added.
    
    candidates = set(range(n))
    S = []
    
    while candidates:
        # Pick a point. Heuristic: pick one with fewest "dangerous" interactions?
        # Just pick the first one for speed.
        p = next(iter(candidates))
        S.append(p)
        candidates.remove(p)
        
        # Remove all q in candidates such that there exists r in S where p, q, r are collinear.
        to_remove = set()
        for r in S:
            # Find all q such that p, r, q are collinear
            # We can precompute collinear_with[p][r]
            # But we didn't precompute that in this function.
            # Let's just check all q in candidates.
            for q in candidates:
                x1, y1 = points[p]
                x2, y2 = points[r]
                x3, y3 = points[q]
                dx = x2 - x1
                dy = y2 - y1
                if dx * (y3 - y1) == dy * (x3 - x1):
                    to_remove.add(q)
                    break
        
        candidates -= to_remove
    
    return len(S)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Generate initial configuration
    # Strategy: Place points on a grid-like structure to encourage triples.
    # Try to place points on horizontal lines y = 0, 2, 4, ...
    # 7 lines * 3 points = 21 points. We need 20.
    # So 6 lines * 3 points = 18, plus 2 points elsewhere.
    
    def generate_initial_structured():
        pts = []
        # Define lines
        lines_y = [0, 2, 4, 6, 8, 10]
        for y in lines_y:
            # Place 3 points on this line
            # Random x coordinates, distinct
            xs = random.sample(range(-10, 11), 3)
            for x in xs:
                pts.append([x, y])
        
        # Add 2 more points
        # Ensure no 4 collinear
        while len(pts) < 20:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            p = [x, y]
            if p in pts:
                continue
            # Check validity
            temp_pts = pts + [p]
            if _check_no_four_collinear(temp_pts):
                pts.append(p)
        
        return pts

    current_points = generate_initial_structured()
    current_score = _max_independent_set(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    # Use greedy upper bound for fast evaluation during SA
    # Only compute exact score for the best few candidates
    
    def random_point():
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        return [x, y]

    while time.time() < deadline:
        if T < T_min:
            break
            
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        new_p = random_point()
        
        if new_p == old_p:
            continue
        if new_p in current_points:
            continue
            
        current_points[idx] = new_p
        
        if not _check_no_four_collinear(current_points):
            current_points[idx] = old_p
            continue
            
        # Fast evaluation
        new_greedy = _greedy_upper_bound(current_points)
        
        # We need a comparable current_greedy
        # To avoid recomputing current_greedy every time, we can store it
        # But for simplicity, let's just use the exact score if it's cheap enough?
        # _max_independent_set is the bottleneck.
        # Let's compute exact score only if new_greedy is promising?
        # Or just use greedy for the delta, and verify at the end.
        
        # Let's maintain a running greedy score for current state
        # This is tricky because we don't have current_greedy stored.
        # Let's just compute exact score for the candidate if time permits?
        # No, that's too slow.
        
        # Alternative: Use a fixed number of SA steps with greedy, then refine.
        # For now, let's just accept if new_greedy < best_greedy_found?
        # We need to track best_greedy.
        
        # Let's change strategy:
        # 1. Run SA minimizing greedy upper bound.
        # 2. Keep the best 10 candidates by greedy bound.
        # 3. At the end, compute exact score for the best candidate.
        
        # This requires storing candidates.
        # For simplicity in this rewrite, let's just do random restarts of the structured generator
        # and local search, and compute exact score for the best few.
        
        # Actually, the current program already does SA.
        # The problem is speed.
        # Let's just do many random restarts of the structured generator and pick the best exact score.
        
        # Break out of SA and do restarts
        break

    # Restart strategy
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        pts = generate_initial_structured()
        # Local search on this point
        for step in range(100):
            if time.time() > deadline - 10:
                break
            idx = random.randint(0, 19)
            old_p = pts[idx]
            new_p = random_point()
            if new_p == old_p or new_p in pts:
                continue
            pts[idx] = new_p
            if not _check_no_four_collinear(pts):
                pts[idx] = old_p
                continue
            # Fast check: if greedy bound is worse than current best, reject?
            # We don't have current best greedy.
            # Let's just accept if valid, and compute exact score at the end of each restart.
            
        # Compute exact score for this restart's result
        score = _max_independent_set(pts)
        if score < best_score:
            best_score = score
            best_points = pts[:]
            
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
