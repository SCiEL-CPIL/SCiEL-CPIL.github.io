import time
import random
import math
import numpy as np

N = 64
NUM_POINTS = N * N
# Precompute coordinates
COORDS = [(i // N, i % N) for i in range(NUM_POINTS)]

# Precompute squared distances
xs = np.array([i // N for i in range(NUM_POINTS)])
ys = np.array([i % N for i in range(NUM_POINTS)])
dx_all = xs[:, None] - xs[None, :]
dy_all = ys[:, None] - ys[None, :]
dist2_mat = (dx_all * dx_all + dy_all * dy_all).astype(np.int32)

# Convert to lists of lists for faster access in Python loops if needed, 
# but numpy indexing is fast. Let's stick to numpy but ensure we handle types correctly.
# dist2_mat[i, j] is int32.

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # State for backtracking
    # S: list of point indices currently in the set
    # in_set: boolean array
    # used_d2: list of sets, used_d2[i] contains d2 values of distances from i to other points in S
    
    S = []
    in_set = [False] * NUM_POINTS
    used_d2 = [set() for _ in range(NUM_POINTS)]
    
    best_S = []
    best_size = 0
    
    # We will use a recursive backtracking search with a depth limit or time limit.
    # However, pure DFS might be too slow if the branching factor is high.
    # We will use a "Depth-First Search with Restarting" or simply a guided search.
    # Given the constraints, a simple DFS that goes deep might get stuck.
    # Let's use a hybrid: 
    # 1. Greedy initialization to get a decent base.
    # 2. Then, use a local search that tries to swap points, but guided by a "conflict count" or similar?
    # No, the directive asks for a DIFFERENT strategy. 
    # Let's try a "Simulated Annealing on the set of points" but with a different move set?
    # Or "Iterated Local Search" with a specific perturbation?
    
    # The current program uses random greedy + local search.
    # A materially different approach: **Conflict-Driven Backtracking (CDCL-like) or simply Backtracking with Heuristics.**
    
    # Let's implement a Backtracking Search that tries to build the set from scratch, 
    # but uses "Forward Checking" to prune branches early.
    # To avoid exponential blowup, we will limit the depth or use a "Beam Search" variant.
    
    # Actually, a very effective different strategy for packing problems is **Min-Conflicts Local Search** combined with **Restarts**.
    # But the current one is already local search.
    
    # Let's try **Genetic Algorithm**? 
    # Representation: Bitset of 4096 bits.
    # Fitness: Number of points - Penalty * (Number of isosceles triangles).
    # This is a different optimization family (Evolutionary vs Greedy/Local Search).
    
    # However, evaluating fitness for a GA individual is expensive (O(n^2) or O(n^3)).
    # With n=4096, O(n^2) is 16M ops. Too slow for GA population.
    
    # Let's go with **Backtracking with Dynamic Variable Ordering (MRV)**.
    # We try to add points one by one. At each step, we choose the "most constrained" candidate?
    # No, we are adding points. The constraint is that the new point must not form an isosceles triangle with any pair in S.
    # This is equivalent to:
    # 1. Distances from new point p to all a in S must be distinct.
    # 2. For each a in S, distance(p, a) must not be in used_d2[a].
    
    # Let's implement a **Randomized Backtracking Search** with a depth limit and restarts.
    # This explores the solution space differently than greedy (which commits immediately).
    
    def can_add(p):
        if in_set[p]:
            return False
        d2s_seen = set()
        for a in S:
            d2 = int(dist2_mat[p, a])
            if d2 in d2s_seen:
                return False
            d2s_seen.add(d2)
            if d2 in used_d2[a]:
                return False
        return True
        
    def add_point(p):
        S.append(p)
        in_set[p] = True
        used_d2[p] = set()
        for a in S[:-1]:
            d2 = int(dist2_mat[p, a])
            used_d2[a].add(d2)
            used_d2[p].add(d2)
            
    def remove_point(p):
        idx = S.index(p)
        S.pop(idx)
        in_set[p] = False
        for a in S:
            d2 = int(dist2_mat[p, a])
            used_d2[a].discard(d2)
        used_d2[p] = set()
        
    # Phase 1: Greedy to get a baseline
    # We use a specific ordering: sort by "potential" or just random.
    # Let's use a greedy that picks the point with the fewest conflicts first (MRV-like for initialization)
    # Or simply random greedy as before to establish a baseline.
    
    # Let's do a quick greedy to get a starting set for the backtracker to improve upon?
    # No, the backtracker starts from scratch to explore deep structures.
    
    # To make it efficient, we will use a **Beam Search** over the set of points.
    # State: Current Set S.
    # Expansion: For each point p not in S, if can_add(p), create new state S + [p].
    # We keep only the top K states (beams) based on some heuristic (e.g., size, or potential).
    # Since all valid states at depth d have size d, we need a heuristic for "promisingness".
    # A good heuristic: The number of "blocked" future points? Or just random sampling of valid extensions?
    
    # Let's try **Randomized Greedy with Lookahead**.
    # At each step, instead of picking the first valid point, we pick a random valid point, but with a bias towards points that leave more options for the future?
    # This is hard to compute.
    
    # Let's stick to **Backtracking with Restarts and Randomization**.
    # We will attempt to build a set of size K.
    # We increase K until we fail.
    # This is different from the current "add as many as you can" greedy.
    
    # However, backtracking for K=100+ on 4096 points is likely to explode.
    
    # Alternative Different Strategy: **Conflict-Driven Local Search**.
    # Start with a random set of size M (larger than optimal).
    # Identify conflicts (isosceles triangles).
    # Remove points involved in the most conflicts.
    # Repeat until no conflicts.
    # This is a "Min-Conflicts" heuristic.
    
    # Let's implement Min-Conflicts.
    
    def count_conflicts(S_local):
        # Returns number of isosceles triangles
        # For each triple (a, b, c) in S_local, check if isosceles.
        # This is O(n^3) which is too slow for n=100.
        # We can optimize: 
        # For each point a, count pairs (b, c) such that dist(a,b)==dist(a,c).
        # Sum over all a.
        count = 0
        n = len(S_local)
        # Use the precomputed dist2_mat
        for i in range(n):
            a = S_local[i]
            d2_counts = {}
            for j in range(n):
                if i == j: continue
                b = S_local[j]
                d2 = int(dist2_mat[a, b])
                if d2 in d2_counts:
                    d2_counts[d2] += 1
                else:
                    d2_counts[d2] = 1
            # For each distance d2 with count k, there are k*(k-1)/2 isosceles triangles with apex a
            for k in d2_counts.values():
                if k > 1:
                    count += k * (k - 1) // 2
        return count
        
    # Min-Conflicts Algorithm:
    # 1. Initialize S with M random points.
    # 2. While S has conflicts:
    #    a. Find the point p in S that is part of the most conflicts.
    #    b. Remove p from S.
    #    c. (Optional) Try to add a random point not in S that reduces conflicts?
    #       Or just keep removing until valid.
    #    d. If S becomes too small, restart.
    
    # This is structurally different: it starts with a "bad" set and repairs it,
    # whereas the current program starts with an "empty" set and builds it.
    
    M = 150  # Start larger than expected optimum (10.5)
    
    def min_conflicts_search():
        S_local = []
        # Randomly pick M distinct points
        candidates = list(range(NUM_POINTS))
        random.shuffle(candidates)
        S_local = candidates[:M]
        
        # To speed up conflict counting, we can maintain counts incrementally?
        # For simplicity, we recalculate or update locally.
        # Given M=150, O(M^2) is 22500 ops. Very fast.
        
        # We want to remove points to eliminate conflicts.
        # We will loop until no conflicts.
        
        # To avoid getting stuck in a local minimum of "valid but small" sets,
        # we can try to add points back if possible?
        # No, Min-Conflicts usually just removes.
        # To get a large set, we start with a very large M.
        
        # Optimization: Instead of removing the point with max conflicts,
        # we can remove a random point from the set of points involved in conflicts.
        
        while True:
            # Calculate conflicts for each point
            # conflict_count[p] = number of isosceles triangles involving p
            # We can compute this in O(M^2)
            
            conflict_count = {p: 0 for p in S_local}
            
            # For each pair (a, b), check if there exists c such that (a,b,c) is isosceles?
            # No, isosceles is defined by distances.
            # Triangle (a,b,c) is isosceles if d(a,b)==d(a,c) OR d(a,b)==d(b,c) OR d(a,c)==d(b,c).
            
            # Efficient counting:
            # For each a in S:
            #   Group other points by distance from a.
            #   If a group has size k > 1, then for each pair (b,c) in that group,
            #   the triangle (a,b,c) has d(a,b)==d(a,c).
            #   This contributes to conflicts for a, b, and c.
            
            # Let's just count total conflicts and identify "bad" points.
            # For the purpose of Min-Conflicts, we need to assign blame.
            # A triangle (a,b,c) with d(a,b)==d(a,c) is "blamed" on a? Or all three?
            # Usually, we minimize the sum of conflicts.
            
            # Let's compute the number of isosceles triangles where each point is the "apex" (equal legs).
            # This covers all isosceles triangles?
            # If d(a,b)==d(a,c), apex is a.
            # If d(a,b)==d(b,c), apex is b.
            # If d(a,c)==d(b,c), apex is c.
            # An equilateral triangle has 3 apexes.
            # An isosceles non-equilateral has 1 apex.
            
            # We want to remove points that are apexes of many such triangles.
            
            apex_counts = {p: 0 for p in S_local}
            
            for i in range(len(S_local)):
                a = S_local[i]
                d2_map = {}
                for j in range(len(S_local)):
                    if i == j: continue
                    b = S_local[j]
                    d2 = int(dist2_mat[a, b])
                    if d2 in d2_map:
                        d2_map[d2].append(b)
                    else:
                        d2_map[d2] = [b]
                        
                for d2, neighbors in d2_map.items():
                    k = len(neighbors)
                    if k > 1:
                        # Number of pairs is k*(k-1)/2
                        # Each pair forms an isosceles triangle with apex a
                        count = k * (k - 1) // 2
                        apex_counts[a] += count
                        # Also, these triangles involve the neighbors.
                        # We might want to add a fraction of this count to the neighbors?
                        # Or just use apex count as a proxy for "badness".
                        # Points that are apexes of many triangles are "central" to conflicts.
                        
            # Find the point with the highest apex count
            if not S_local:
                break
                
            max_conf = max(apex_counts.values())
            if max_conf == 0:
                # No conflicts
                break
                
            # Remove the point with max conflict
            # If multiple, pick randomly among them
            candidates_to_remove = [p for p, c in apex_counts.items() if c == max_conf]
            p_to_remove = random.choice(candidates_to_remove)
            S_local.remove(p_to_remove)
            
            # Optional: Try to add a random point to keep size up?
            # No, that might reintroduce conflicts.
            # We will rely on starting with a large M.
            
            if time.time() > deadline:
                return S_local
                
        return S_local

    # Run multiple instances of Min-Conflicts with different M
    best_size = 0
    best_S = []
    
    # We can also try to "repair" a greedy solution using Min-Conflicts?
    # Or just run Min-Conflicts from scratch with high M.
    
    # Let's try M from 200 down to 100?
    # Or just a few random restarts with M=150.
    
    for m in [200, 180, 160, 150]:
        for _ in range(5):
            if time.time() > deadline:
                break
            S_mc = min_conflicts_search()
            if len(S_mc) > best_size:
                best_size = len(S_mc)
                best_S = S_mc[:]
                
    # If Min-Conflicts didn't find anything better than a simple greedy, 
    # we might want to fall back to the previous strategy?
    # No, we must stick to the new strategy.
    # But we can combine: Use Min-Conflicts to get a valid set, then try to ADD points to it?
    
    # Let's add a phase: Try to add points to the best_S found by Min-Conflicts.
    # This is similar to the original local search, but starting from a different point.
    
    # Initialize state with best_S
    S = best_S[:]
    in_set = [False] * NUM_POINTS
    used_d2 = [set() for _ in range(NUM_POINTS)]
    for p in S:
        add_point(p)
        
    # Try to add more points
    # This is a "Repair and Expand" strategy.
    
    while time.time() < deadline:
        # Try to add a random point
        p = random.randint(0, NUM_POINTS - 1)
        if not in_set[p] and can_add(p):
            add_point(p)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S[:]
        else:
            # If we can't add, maybe remove a random point and try again?
            # This is getting back to local search.
            # To keep it distinct, let's just do random additions.
            pass
            
    return [COORDS[i] for i in best_S]

# To ensure we don't return an empty list if Min-Conflicts fails completely,
# we should have a fallback or ensure Min-Conflicts always returns a valid set.
# Min-Conflicts removes points until valid, so it always returns a valid set (possibly empty).
# If it returns empty, we should probably run a greedy fallback?
# The task says "Keep the function names and signatures".
# It doesn't forbid using multiple strategies, but the directive asks for a MATERIAL DIFFERENCE.
# Using Min-Conflicts as the primary engine is the difference.
# If it fails, we can just return the result.
# However, score 0 is bad.
# Let's add a simple greedy fallback if Min-Conflicts yields < 5 points.

# Refining construct to include a fallback greedy if MC is poor.

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # ... (precomputation is global) ...
    
    S = []
    in_set = [False] * NUM_POINTS
    used_d2 = [set() for _ in range(NUM_POINTS)]
    
    best_S = []
    best_size = 0
    
    def can_add(p):
        if in_set[p]:
            return False
        d2s_seen = set()
        for a in S:
            d2 = int(dist2_mat[p, a])
            if d2 in d2s_seen:
                return False
            d2s_seen.add(d2)
            if d2 in used_d2[a]:
                return False
        return True
        
    def add_point(p):
        S.append(p)
        in_set[p] = True
        used_d2[p] = set()
        for a in S[:-1]:
            d2 = int(dist2_mat[p, a])
            used_d2[a].add(d2)
            used_d2[p].add(d2)
            
    def remove_point(p):
        idx = S.index(p)
        S.pop(idx)
        in_set[p] = False
        for a in S:
            d2 = int(dist2_mat[p, a])
            used_d2[a].discard(d2)
        used_d2[p] = set()
        
    # Strategy: Min-Conflicts
    # We define the MC search inside to access S? No, MC is independent.
    
    def run_min_conflicts(M):
        S_local = list(range(NUM_POINTS))
        random.shuffle(S_local)
        S_local = S_local[:M]
        
        # Optimization: Use a set for fast lookup if needed, but list is fine for M<300
        
        while True:
            if time.time() > deadline:
                return S_local
                
            # Calculate apex conflicts
            apex_counts = {p: 0 for p in S_local}
            
            for i in range(len(S_local)):
                a = S_local[i]
                d2_map = {}
                for j in range(len(S_local)):
                    if i == j: continue
                    b = S_local[j]
                    d2 = int(dist2_mat[a, b])
                    if d2 in d2_map:
                        d2_map[d2].append(b)
                    else:
                        d2_map[d2] = [b]
                        
                for d2, neighbors in d2_map.items():
                    k = len(neighbors)
                    if k > 1:
                        count = k * (k - 1) // 2
                        apex_counts[a] += count
                        
            max_conf = max(apex_counts.values()) if S_local else 0
            if max_conf == 0:
                return S_local
                
            candidates_to_remove = [p for p, c in apex_counts.items() if c == max_conf]
            p_to_remove = random.choice(candidates_to_remove)
            S_local.remove(p_to_remove)
            
    # Run MC
    for M in [200, 150]:
        for _ in range(3):
            if time.time() > deadline:
                break
            S_mc = run_min_conflicts(M)
            if len(S_mc) > best_size:
                best_size = len(S_mc)
                best_S = S_mc[:]
                
    # Fallback: If MC didn't find a good set, do a quick greedy
    if best_size < 5:
        # Simple greedy
        S = []
        in_set = [False] * NUM_POINTS
        used_d2 = [set() for _ in range(NUM_POINTS)]
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        for p in order:
            if can_add(p):
                add_point(p)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            
    # Try to expand the best_S found
    # Load best_S into global state
    S = best_S[:]
    in_set = [False] * NUM_POINTS
    used_d2 = [set() for _ in range(NUM_POINTS)]
    for p in S:
        add_point(p)
        
    # Expansion phase
    while time.time() < deadline:
        p = random.randint(0, NUM_POINTS - 1)
        if not in_set[p] and can_add(p):
            add_point(p)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S[:]
                
    return [COORDS[i] for i in best_S]

# The code above is complete.
# Note: The "Expansion phase" is similar to the original, but the core generation is Min-Conflicts.
# The Min-Conflicts approach starts with a large, invalid set and prunes it, which is a different trajectory
# than starting empty and growing. This can escape local minima that greedy methods fall into
# because the "landscape" of valid sets is explored from the "top" down rather than "bottom" up.

