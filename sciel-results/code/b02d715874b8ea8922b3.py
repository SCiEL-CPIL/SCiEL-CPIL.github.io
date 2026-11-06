"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

# Precompute all grid points
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances between all pairs of points
dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute for each point p, and each distance d, the list of points at distance d from p
# This allows O(1) lookup of points at same distance.
# But for the conflict graph, we need to know if (p, a, b) is isosceles.
# A triangle (p, a, b) is isosceles if:
# 1. dist(p, a) == dist(p, b)
# 2. dist(p, a) == dist(a, b)
# 3. dist(p, b) == dist(a, b)

# Let's define a "conflict" between a candidate point p and a pair (a, b) in the current set.
# However, the standard Independent Set formulation on a graph G=(V, E) where E contains edges (i, j) 
# if points i and j "conflict" is not directly applicable because the constraint is on triples.
# 
# Wait, the prompt suggests: "Model the problem as a maximum independent set in a conflict graph where edges connect points that form an isosceles triangle with any third grid point".
# This phrasing is slightly ambiguous. Usually, for "no isosceles triangles", we look for a set S such that no triple in S is isosceles.
# This is not a standard independent set problem on a simple graph of pairs, because the constraint is 3-wise.
# However, we can reformulate:
# A set S is valid iff for all p in S, the set of distances from p to other points in S are all distinct, AND for all pairs (a, b) in S, the distance d(a, b) is not equal to d(p, a) or d(p, b) for any p in S? No, that's getting complicated.
#
# Let's stick to the local search strategy which is known to work well for this specific "maximal no-isosceles" task, but optimize the internal checks significantly.
# The current program uses O(k^2) checks for `can_add_point`.
# We can precompute for each point p, a list of "forbidden" points relative to any other point?
#
# Actually, let's look at the "Maximal" constraint.
# A set S is maximal if for every point q not in S, S U {q} is NOT valid (i.e., contains an isosceles triangle).
# This means for every q not in S, there exist a, b in S such that (q, a, b) is isosceles.
#
# Strategy:
# 1. Generate a maximal valid set using greedy.
# 2. Minimize it: Remove points if the set remains maximal and valid.
# 3. Use Simulated Annealing or Randomized Local Search to find smaller maximal sets.
#    Moves:
#    - Remove a point p from S. If S\{p} is valid (always) and maximal, accept.
#    - If S\{p} is not maximal, we are in an "invalid" state (non-maximal).
#    - To escape, we can add a point q that restores maximality? No, adding a point might break validity.
#    - Better: Treat the state as just "Valid Set". Objective: Minimize size, Subject to Maximal.
#    - Penalty method: Energy = size + Lambda * (number of points that CAN be added).
#    - If we can add a point, the set is not maximal. We want 0 addable points.
#    - So, we search for a valid set with 0 addable points, minimizing size.
#
# Let's implement a fast validity and maximality checker.
# To check if a set S is valid:
#   For each p in S, check if distances to other points in S are unique.
#   For each pair (a, b) in S, check if d(a, b) is not equal to d(p, a) or d(p, b) for any p in S?
#   Actually, if we maintain S as a set of indices, we can check validity incrementally.
#
# Let's use the following optimization for `can_add_point(p, S)`:
# 1. Check if p is at a unique distance from all points in S.
#    If not, invalid.
# 2. For each a in S, let d_pa = dist2[p][a].
#    Check if d_pa is equal to dist2[a][b] for any b in S.
#    If yes, invalid.
#
# We can precompute for each point a, a set of distances to all other points?
# No, we need to check if d_pa is in {dist2[a][b] for b in S}.
#
# Let's try to speed up the local search by precomputing "conflict pairs" for a given point p.
# For a fixed p, and a fixed a, the condition "dist(p, a) == dist(a, b)" defines a set of points b.
# Let's precompute `same_dist_from_a[p][a]`? No, that's too big.
#
# Let's precompute for each pair (p, a), the list of points b such that dist(p, a) == dist(a, b).
# Let `conflict_b[p][a]` be a set of b's.
# Then, when checking if we can add p to S:
# 1. Check uniqueness of distances from p to S.
# 2. For each a in S, check if S intersects `conflict_b[p][a]`.
#    If yes, invalid.
#
# This reduces the check to O(k) set intersections (if we represent S as a set).
# Precomputation:
# For p in range(N*N):
#   for a in range(N*N):
#     if p == a: continue
#     d_pa = dist2[p][a]
#     # Find all b such that dist2[a][b] == d_pa
#     # We can iterate b or use a precomputed map for a: dist -> list of b's.
#
# Let's build `dist_map[a][d]` = list of b such that dist2[a][b] == d.
# This is efficient.

# Precompute distance maps
dist_map = [defaultdict(list) for _ in range(NUM_POINTS)]
for a in range(NUM_POINTS):
    for b in range(NUM_POINTS):
        if a == b:
            continue
        d = dist2[a][b]
        dist_map[a][d].append(b)

def can_add_point_fast(p_idx, S_set):
    """
    Check if p_idx can be added to S_set (a set of indices) without creating an isosceles triangle.
    Uses precomputed dist_map for speed.
    """
    # 1. Check uniqueness of distances from p to points in S
    seen_dists = set()
    for a in S_set:
        d_pa = dist2[p_idx][a]
        if d_pa in seen_dists:
            return False
        seen_dists.add(d_pa)
        
    # 2. Check for conflicts: dist(p, a) == dist(a, b)
    # For each a in S, we check if any b in S is in dist_map[a][d_pa]
    for a in S_set:
        d_pa = dist2[p_idx][a]
        # Get list of b's at distance d_pa from a
        candidates_b = dist_map[a][d_pa]
        # Check intersection with S_set
        # Since S_set is a set, we can iterate candidates_b and check membership
        for b in candidates_b:
            if b in S_set:
                return False
    return True

def is_maximal_fast(S_list):
    """
    Check if no point can be added to S_list.
    """
    S_set = set(S_list)
    for p_idx in range(NUM_POINTS):
        if p_idx in S_set:
            continue
        if can_add_point_fast(p_idx, S_set):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # We will use multiple restarts with different random seeds
    # and a local search procedure.
    
    for restart in range(200): # Increase restarts if time permits
        if time.time() > deadline:
            break
            
        random.seed(restart * 1000 + 7)
        
        # 1. Greedy Construction
        # Random order
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        current_set = []
        current_set_set = set()
        
        for p_idx in order:
            if can_add_point_fast(p_idx, current_set_set):
                current_set.append(p_idx)
                current_set_set.add(p_idx)
                
        # 2. Minimize: Remove points if possible
        # We want to find a minimal maximal set.
        # Iteratively try to remove each point. If the set remains maximal, remove it.
        
        # To speed up, we can shuffle the order of removal attempts.
        changed = True
        while changed:
            changed = False
            # Try to remove points in random order
            indices_to_try = list(range(len(current_set)))
            random.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if time.time() > deadline:
                    break
                
                p_to_remove = current_set[idx]
                
                # Trial removal
                trial_set = [x for i, x in enumerate(current_set) if i != idx]
                trial_set_set = set(trial_set)
                
                if is_maximal_fast(trial_set):
                    # Accept removal
                    current_set = trial_set
                    current_set_set = trial_set_set
                    changed = True
                    # Restart the loop because the set changed
                    break
                    
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # 3. Local Search / Perturbation
        # Try to escape local minima by removing a point, then repairing.
        # If removal makes it non-maximal, we need to add points back to restore maximality.
        # We want to add as few points as possible, and preferably points that allow further removals.
        
        for _ in range(50): # Local search steps
            if time.time() > deadline:
                break
            
            if not current_set:
                break
                
            # Pick a random point to remove
            p_to_remove = random.choice(current_set)
            trial_set = [x for x in current_set if x != p_to_remove]
            trial_set_set = set(trial_set)
            
            # Check if it's still maximal
            if is_maximal_fast(trial_set):
                # Great, we found a smaller maximal set
                current_set = trial_set
                current_set_set = trial_set_set
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                continue
            
            # It's not maximal. We need to add points to make it maximal.
            # Greedy repair: Find all points that can be added.
            # Add one randomly? Or add the one that "covers" the most non-maximality?
            # Let's just add a random valid point until maximal.
            
            repair_set = list(trial_set)
            repair_set_set = set(repair_set)
            
            # Find all addable points
            addable = []
            for q in range(NUM_POINTS):
                if q not in repair_set_set:
                    if can_add_point_fast(q, repair_set_set):
                        addable.append(q)
                        
            if not addable:
                # Should not happen if trial_set is valid, but just in case
                continue
                
            # Add points randomly until maximal
            # Limit the number of adds to avoid blowing up the size
            max_adds = 3 
            added_count = 0
            while added_count < max_adds:
                if is_maximal_fast(repair_set):
                    break
                if not addable:
                    break
                    
                # Pick a random addable point
                q = random.choice(addable)
                repair_set.append(q)
                repair_set_set.add(q)
                added_count += 1
                
                # Update addable list? This is expensive.
                # Let's just recompute addable if we want to be precise.
                # For speed, we'll just try to add until maximal or max_adds reached.
                # If we reach max_adds and it's not maximal, we discard this trial.
                
            if is_maximal_fast(repair_set):
                # Now minimize the repaired set
                changed = True
                while changed:
                    changed = False
                    indices_to_try = list(range(len(repair_set)))
                    random.shuffle(indices_to_try)
                    for idx in indices_to_try:
                        if time.time() > deadline:
                            break
                        p_to_remove = repair_set[idx]
                        t2 = [x for i, x in enumerate(repair_set) if i != idx]
                        if is_maximal_fast(t2):
                            repair_set = t2
                            changed = True
                            break
                            
                if len(repair_set) <= len(current_set):
                    current_set = repair_set
                    current_set_set = set(current_set)
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)

    if best_set is None:
        # Fallback
        current_set = []
        current_set_set = set()
        for p_idx in range(NUM_POINTS):
            if can_add_point_fast(p_idx, current_set_set):
                current_set.append(p_idx)
                current_set_set.add(p_idx)
        return [POINTS[i] for i in current_set]
        
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
