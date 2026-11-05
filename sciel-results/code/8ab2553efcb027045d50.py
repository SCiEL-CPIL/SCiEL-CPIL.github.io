"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Precompute all pairwise squared distances
    all_points = list(itertools.product(range(N), repeat=2))
    n_pts = len(all_points)
    pt_idx = {p: i for i, p in enumerate(all_points)}
    
    # dist2[i][j] = squared distance between point i and point j
    dist2 = [[0]*n_pts for _ in range(n_pts)]
    for i in range(n_pts):
        x1, y1 = all_points[i]
        for j in range(i+1, n_pts):
            x2, y2 = all_points[j]
            d2 = (x1-x2)**2 + (y1-y2)**2
            dist2[i][j] = d2
            dist2[j][i] = d2
    
    # For each point p and each pair (a,b) in the set, we need to check if
    # dist2[p][a] == dist2[p][b] or dist2[p][a] == dist2[a][b] or dist2[p][b] == dist2[a][b]
    
    # A point p is "blocked" by a set S if for every p not in S, there exist a,b in S
    # such that triangle (p,a,b) is isosceles.
    
    # Strategy: Use a different approach. Instead of random greedy, use a
    # constraint-satisfaction / exact cover style approach with bitmasks.
    
    # For each point p, define blocked_by[p] as a bitmask of points that "block" p.
    # Actually, let's think differently.
    
    # A set S is no-isosceles if for all distinct a,b,c in S, the triangle is not isosceles.
    # S is maximal if for all p not in S, there exist a,b in S such that (p,a,b) is isosceles.
    
    # Let's use a bitmask representation. n_pts = 256, so we need 256-bit integers.
    # Python handles big integers natively.
    
    # For each point p, define iso_mask[p] as the set of all points q such that
    # there exists some r where (p, q, r) is isosceles. Wait, that's not quite right.
    
    # Let's define: for a candidate set S (as a bitmask), and a point p not in S,
    # p can be added to S iff for all a, b in S, the triangle (p, a, b) is NOT isosceles.
    # This is equivalent to: for all a, b in S, dist2[p][a] != dist2[p][b] AND 
    # dist2[p][a] != dist2[a][b] AND dist2[p][b] != dist2[a][b].
    
    # This is hard to check with bitmasks directly.
    
    # Alternative strategy: Use simulated annealing or local search with a different
    # neighborhood. Start with a valid no-isosceles set (not necessarily maximal), then
    # try to add points to make it maximal while keeping the size small, or try to
    # remove points and re-add different ones.
    
    # Actually, let's try a fundamentally different approach:
    # 1. Find a large no-isosceles set using a greedy approach (maximize size).
    # 2. Then, try to find a minimal subset that is still maximal.
    #    This is like finding a hitting set / dominating set problem.
    
    # Wait, the task is to find a MINIMAL MAXIMAL no-isosceles set.
    # "Minimal" here means minimum size among all maximal sets.
    
    # Let's try a different construction: use symmetry and structure.
    # For N=16, maybe there's a pattern.
    
    # Let's try: pick points on a sparse grid, e.g., only even coordinates, or
    # use a lattice structure that avoids isosceles triangles.
    
    # Actually, let's use a smarter local search:
    # 1. Start with a maximal no-isosceles set S (from greedy).
    # 2. Try to swap: remove p from S, add q not in S, such that S\{p}∪{q} is still
    #    a valid no-isosceles set AND is maximal. If the size stays the same, we might
    #    enable further reductions.
    # 3. More importantly, try to find a smaller maximal set by searching over subsets.
    
    # Let's implement a branch-and-bound style search for small sets.
    # But the search space is huge.
    
    # Different strategy: Use the fact that a maximal no-isosceles set S has the property
    # that every point outside S is "covered" by some pair in S. We can formulate this
    # as: find a set S such that (1) S is no-isosceles, and (2) for every p not in S,
    # there exist a,b in S with (p,a,b) isosceles.
    
    # Let's try a genetic algorithm / evolutionary approach.
    
    # Actually, let's try a much simpler but different approach:
    # Use a deterministic construction based on number theory.
    # For example, pick points (x, y) where x and y satisfy some condition that
    # prevents isosceles triangles.
    
    # Let's try: pick points where x+y is even (checkerboard). But that might still
    # have isosceles triangles.
    
    # Let's try a random restart with a different objective: minimize the number of
    # points in a maximal set using a penalty-based approach.
    
    # Here's a different strategy: 
    # 1. Generate a random no-isosceles set (not necessarily maximal) by adding points
    #    in random order, skipping those that create isosceles triangles.
    # 2. Complete it to a maximal set by adding points in random order.
    # 3. Try to minimize by removing points that are "redundant" for maximality.
    
    # Let's implement a faster isosceles check using precomputed distances.
    
    def is_iso_idx(pi, aj, bj):
        """Check if triangle formed by points at indices pi, aj, bj is isosceles."""
        d_ab = dist2[aj][bj]
        d_pa = dist2[pi][aj]
        d_pb = dist2[pi][bj]
        return d_pa == d_pb or d_pa == d_ab or d_pb == d_ab
    
    def can_add(pi, chosen_indices):
        """Check if point pi can be added to the set of chosen indices."""
        for i in range(len(chosen_indices)):
            for j in range(i+1, len(chosen_indices)):
                if is_iso_idx(pi, chosen_indices[i], chosen_indices[j]):
                    return False
        return True
    
    def is_maximal(chosen_indices):
        """Check if the set is maximal (no point can be added)."""
        chosen_set = set(chosen_indices)
        for pi in range(n_pts):
            if pi in chosen_set:
                continue
            if can_add(pi, chosen_indices):
                return False
        return True
    
    # Greedy construction with random order
    def greedy_random():
        order = list(range(n_pts))
        random.shuffle(order)
        chosen = []
        for pi in order:
            if can_add(pi, chosen):
                chosen.append(pi)
        return chosen
    
    # Try to minimize a maximal set by removing points
    def minimize_maximal(chosen):
        current = list(chosen)
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                test = current[:i] + current[i+1:]
                if is_maximal(test):
                    current = test
                    improved = True
                    break
        return current
    
    best = None
    best_size = float('inf')
    
    while time.time() < deadline:
        # Generate a random maximal set
        candidate = greedy_random()
        
        # Make it maximal if it isn't already
        # Actually, greedy_random already produces a maximal set because it tries
        # all points and adds any that can be added. So it's maximal.
        
        # Minimize it
        minimized = minimize_maximal(candidate)
        
        if len(minimized) < best_size:
            best_size = len(minimized)
            best = minimized
        
        # If we find a very small set, we can stop early
        if best_size <= 20:
            break
    
    # Convert indices back to points
    if best is None:
        # Fallback: return an empty set or a trivial set
        # Actually, we need at least some points. Let's return a single point.
        return [all_points[0]]
    
    return [all_points[i] for i in best]
# EVOLVE-BLOCK-END
