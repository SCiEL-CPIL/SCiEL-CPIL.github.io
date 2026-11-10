"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Precompute coordinates and squared distances
# Points are indexed by 0 to N*N-1
# point index i corresponds to (i // N, i % N)
num_points = N * N
coords = np.array([(i // N, i % N) for i in range(num_points)])

# dist[i][j] is squared distance between point i and point j
# Using int32 for memory efficiency, though int64 is safer. 16x16 grid max dist is 2*15^2 = 450. int32 is plenty.
dist = np.zeros((num_points, num_points), dtype=np.int32)

# Compute distances using broadcasting
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist = np.sum(diff**2, axis=2).astype(np.int32)

# Precompute isosceles triples?
# A triple (i, j, k) is isosceles if dist[i,j] == dist[i,k] or dist[i,j] == dist[j,k] or dist[i,k] == dist[j,k].
# For the hypergraph approach, we want to find a Maximum Independent Set in the 3-uniform hypergraph.
# This is equivalent to finding a Maximum Set S such that no triple in S is isosceles.
# Wait, the task asks for a MINIMAL MAXIMAL no-isosceles set.
# "Minimal" means inclusion-wise minimal? Or minimum size among maximal sets?
# "Minimum maximal no-isosceles set" usually means:
# 1. The set S is "maximal" with respect to the property "no isosceles triangles". i.e., you cannot add any point.
# 2. Among all such maximal sets, we want the one with the minimum number of points.
#
# The current program does:
# 1. Greedy construction of a maximal set (randomized order).
# 2. Local search: remove a point, refill greedily, keep if smaller.
#
# The inspiration suggests:
# "Formulate the problem as a Maximum Independent Set in the 3-uniform hypergraph of isosceles triples... ensuring the resulting independent set is maximal by construction."
# Wait. A set S is an independent set in the hypergraph of isosceles triples if NO triple in S is isosceles.
# If we find a MAXIMUM independent set, it is the largest possible set with no isosceles triangles.
# But we want the MINIMUM size among MAXIMAL sets.
# A maximal independent set is one where you can't add any more vertices without violating the independence constraint.
# So we are looking for a Minimum Maximal Independent Set (MMIS) in the 3-uniform hypergraph.
#
# This is a different problem than Maximum Independent Set.
# However, the strategy says: "solved via a branch-and-bound algorithm... ensuring the resulting independent set is maximal by construction."
# This phrasing is slightly ambiguous. Does it mean we find a maximal independent set and try to minimize its size?
# Or does it mean we use the structure to guide the search?
#
# Let's re-read the strategy carefully:
# "Formulate the problem as a Maximum Independent Set in the 3-uniform hypergraph of isosceles triples, solved via a branch-and-bound algorithm that uses bitset-based conflict propagation to prune infeasible branches and a greedy lower-bound heuristic to tighten the search, ensuring the resulting independent set is maximal by construction."
#
# Actually, finding a *Maximum* Independent Set gives the *largest* set. We want the *smallest* maximal set.
# These are dual problems in a sense, but not directly solvable by just finding a MIS.
# However, often "Maximum Independent Set" in this context of "no isosceles" might be a misnomer in the strategy description, or it implies we are searching the space of independent sets.
#
# Let's stick to the goal: Minimize |S| subject to S being a maximal no-isosceles set.
#
# A better structural approach for MMIS:
# 1. Use a branch-and-bound search over the vertices.
# 2. State: (included, excluded, undecided).
# 3. Constraints:
#    - If we include a vertex v, we must exclude any vertex u that forms an isosceles triple with two vertices already in 'included'? No, that's for 2-uniform (graph). For 3-uniform, if we include v, it doesn't immediately exclude others unless we already have two others.
#    - Actually, the constraint is: No triple (a,b,c) in 'included' is isosceles.
#    - Maximality constraint: Every vertex in 'excluded' must be "blocked" by 'included'. i.e., for every e in 'excluded', there exists a pair (a,b) in 'included' such that (e,a,b) is isosceles.
#
# This is a very hard constraint to check during search.
#
# Alternative interpretation of the strategy:
# Maybe the strategy implies using the hypergraph structure to do a smart local search or construction?
# "Bitset-based conflict propagation" suggests representing the set of "conflicting" vertices as bitsets.
#
# Let's look at the current program's weakness:
# It uses a randomized greedy start + local search (remove/refill).
# The local search is O(N^3) per step.
#
# Let's try a different local search move:
# Instead of "remove one, refill all", try "swap one for another" or "remove one, add one specific one".
#
# But the directive asks for a STRUCTURAL change based on the inspiration.
# The inspiration mentions "branch-and-bound".
#
# Let's implement a Branch and Bound for MMIS.
#
# Variables: x_i in {0, 1, -1} (In, Out, Undecided).
# We want to minimize sum(x_i).
#
# Pruning:
# 1. If sum(x_i) >= current_best, prune.
# 2. If we have a set of 'In' vertices, check if it's already isosceles-free. If not, prune.
# 3. Maximality check: For the final solution, every 'Out' vertex must be blocked.
#    During search, we can't easily check maximality until the end.
#    However, we can enforce that if a vertex is 'Out', it MUST be blocked.
#    This makes it a constraint satisfaction problem.
#
# Let's define "blocked(v, S)": True if there exist a, b in S such that (v,a,b) is isosceles.
#
# Branching rule:
# Pick an undecided vertex v.
# Branch 1: Include v.
#    - Check if S ∪ {v} is isosceles-free. If not, backtrack.
#    - Recurse.
# Branch 2: Exclude v.
#    - Check if v is blocked by S. If not, we cannot exclude v yet (it would violate maximality).
#    - If blocked, recurse.
#
# This is a valid B&B for MMIS.
#
# Optimization:
# Use bitsets for S (In set).
# Precompute for each vertex v, the set of pairs (a,b) that block v? No, that's too big.
# Precompute for each vertex v, the set of vertices u such that (v, u, w) is isosceles for some w?
#
# Let's precompute `conflict_pairs[v]`? No.
# Let's precompute `blocks[v]` as a list of pairs?
# For a given v, and a set S, v is blocked if there exist a, b in S such that isosceles(v, a, b).
#
# To make this fast, we can precompute for each v, a matrix `M_v[a][b]` which is 1 if (v,a,b) is isosceles.
# Then v is blocked by S if there exist a, b in S such that M_v[a][b] == 1.
# This is equivalent to checking if the submatrix M_v[S, S] contains a 1.
#
# With bitsets, we can represent S as an integer or a numpy array of uint64.
# But checking "does submatrix contain 1" is still O(|S|^2) in the worst case.
#
# However, we can optimize the branching order.
# Heuristic: Branch on vertices that are "hard" to block? Or "easy"?
#
# Given the time limit (300s) and N=16 (256 points), a full B&B might be too slow if the branching factor is high.
# The current greedy+local search is likely faster for finding a *good* solution, but B&B might find the *optimal* or near-optimal one.
#
# Let's combine:
# 1. Use the greedy+local search to get a good initial upper bound (best_size).
# 2. Run a limited B&B to try to beat that bound.
#
# B&B Implementation Details:
# - State: `in_set` (frozenset or sorted list), `out_set`, `undecided`.
# - To speed up, maintain `in_set` as a list and a set for O(1) lookup.
# - `is_isosceles_free(in_set)`: Check all triples in `in_set`. O(k^3).
# - `is_blocked(v, in_set)`: Check if any pair in `in_set` blocks v. O(k^2).
#
# This is still slow for k ~ 100.
#
# Let's use the bitset idea for `is_blocked`.
# For each v, precompute `blockers[v]` as a list of bitmasks?
# Actually, for each v, we can precompute a 2D boolean array `Isos[v]` where `Isos[v][a][b]` is True if (v,a,b) is isosceles.
# Then `is_blocked(v, S)` is `any(Isos[v][a][b] for a in S for b in S if a != b)`.
#
# To make it faster, we can precompute for each v and each a, the set of b's that form an isosceles with (v,a,b).
# `B[v][a]` = set of b such that (v,a,b) is isosceles.
# Then `is_blocked(v, S)` is `any((B[v][a] & S) for a in S)`.
# This is O(k * cost_of_intersection).
# If we use Python sets, intersection is fast.
#
# Let's precompute `B[v]` as a list of sets.
# `B[v][a]` is a set of integers b.
#
# Number of triples: 256^3 / 6 ~ 2.8M.
# Storing this: 256 * 256 sets. Each set has on average 2.8M / (256*256) ~ 42 elements?
# Wait, for a fixed v and a, how many b are isosceles with (v,a,b)?
# The locus of points equidistant from v and a is the perpendicular bisector of va.
# In a 16x16 grid, this line contains at most 16-32 points.
# So the sets are small.
#
# This representation is feasible.
#
# Algorithm:
# 1. Precompute `B[v][a]` sets.
# 2. Greedy + Local Search to get `best_size`.
# 3. B&B:
#    - Global `best_set`, `best_size`.
#    - Recursive function `search(in_list, out_list, undecided_list)`.
#    - Prune if `len(in_list) + min_required_to_block_out` >= `best_size`.
#      How to estimate `min_required_to_block_out`?
#      If `out_list` is empty, 0.
#      Otherwise, we need enough points in `in_list` to block all of `out_list`.
#      Lower bound: `len(out_list)`? No, one point can block many.
#      Simple bound: 0.
#      Better bound: If `out_list` has points that are not blocked by `in_list`, we MUST include more points.
#      Actually, if `v` is in `out_list` and not blocked by `in_list`, we have a contradiction?
#      No, we can still add points to `in_list` later that block it.
#      But if `undecided` is empty, and `v` is not blocked, then this branch is invalid.
#
#    - Branching:
#      Pick `v` from `undecided`.
#      Option 1: Add `v` to `in_list`.
#        - Check if `in_list ∪ {v}` is isosceles-free.
#          - This requires checking (v, a, b) for all a, b in `in_list`.
#          - Use `B[v][a]`? No, `B[v][a]` tells us which b's form isosceles with (v,a,b).
#          - So we need to check if `B[v][a]` intersects `in_list` for any `a` in `in_list`.
#          - If yes, conflict.
#        - If no conflict, recurse.
#      Option 2: Add `v` to `out_list`.
#        - Check if `v` is blocked by `in_list`.
#          - Check if `B[v][a]` intersects `in_list` for any `a` in `in_list`.
#          - If not blocked, we CANNOT put `v` in `out_list` yet.
#          - So this branch is invalid (prune).
#
#    - This forces us to include points to block others. This naturally leads to smaller sets?
#    - Actually, it forces us to find a set that blocks everything we exclude.
#
#    - Ordering:
#      To minimize size, we should try to EXCLUDE points first?
#      If we exclude a point, we are forced to include blockers.
#      If we include a point, we don't have that constraint immediately.
#      So, branching on "Exclude" first might lead to contradictions faster?
#      Or branching on "Include" first might build a valid set faster?
#      
#      For MMIS, we want to exclude as many as possible.
#      So try "Exclude" first.
#      If "Exclude" fails (not blocked), we MUST "Include".
#      So the search is forced.
#
#      This looks like a CSP solver.
#
#    - Time limit:
#      If we run out of time, return the best found.
#
# Let's implement this.

# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

N = 16
num_points = N * N
coords = np.array([(i // N, i % N) for i in range(num_points)])

# Precompute distances
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist = np.sum(diff**2, axis=2).astype(np.int32)

# Precompute B[v][a] = set of b such that (v, a, b) is isosceles
# Note: (v,a,b) is isosceles if d(v,a)==d(v,b) or d(v,a)==d(a,b) or d(v,b)==d(a,b)
# We need to be careful with duplicates and order.
# B[v] is a list of length num_points, each element is a set of ints.
B = [ [set() for _ in range(num_points)] for _ in range(num_points) ]

for v in range(num_points):
    for a in range(num_points):
        if v == a: continue
        d_va = dist[v][a]
        for b in range(num_points):
            if b == v or b == a: continue
            d_vb = dist[v][b]
            d_ab = dist[a][b]
            if d_va == d_vb or d_va == d_ab or d_vb == d_ab:
                B[v][a].add(b)
                # Symmetry: (v,b,a) is also isosceles
                B[v][b].add(a)

def is_conflict(v, in_set):
    """Check if adding v to in_set creates an isosceles triangle."""
    # Check if any pair (a, b) in in_set forms an isosceles triangle with v.
    # This is equivalent to checking if B[v][a] intersects in_set for any a in in_set.
    # But B[v][a] contains b's. So if b is in in_set, conflict.
    in_list = list(in_set)
    for a in in_list:
        if B[v][a] & in_set:
            return True
    return False

def is_blocked(v, in_set):
    """Check if v is blocked by in_set (i.e., cannot be added without creating isosceles)."""
    # v is blocked if there exist a, b in in_set such that (v, a, b) is isosceles.
    # This is the same condition as is_conflict(v, in_set).
    # Wait.
    # is_conflict(v, S) means S ∪ {v} is NOT an independent set.
    # is_blocked(v, S) means v CANNOT be added to S to maintain independence.
    # These are the same thing.
    return is_conflict(v, in_set)

def construct():
    start_time = time.time()
    time_limit = 295
    
    # 1. Greedy + Local Search to get a good initial solution
    best_size = num_points
    best_set = set()
    
    random.seed(42)
    
    # Run a few greedy restarts to get a good upper bound
    for _ in range(20):
        order = list(range(num_points))
        random.shuffle(order)
        current_set = set()
        for p in order:
            if not is_conflict(p, current_set):
                current_set.add(p)
        
        # Local Search: Remove and Refill
        improved = True
        while improved:
            improved = False
            removals = list(current_set)
            random.shuffle(removals)
            for p_remove in removals:
                new_set = set(current_set)
                new_set.remove(p_remove)
                # Refill
                for q in range(num_points):
                    if q not in new_set and not is_conflict(q, new_set):
                        new_set.add(q)
                if len(new_set) < len(current_set):
                    current_set = new_set
                    improved = True
                    break
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        if time.time() - start_time > time_limit * 0.3:
            break

    # 2. Branch and Bound to try to improve
    # State: in_set, out_set, undecided
    # We use a recursive search with a time limit.
    
    global_best_size = best_size
    global_best_set = best_set
    search_count = 0
    deadline = time.time() + (time_limit - (time.time() - start_time))
    
    # Heuristic for branching:
    # Pick an undecided vertex.
    # Try to exclude it first (to minimize size).
    # If exclusion is valid (blocked), recurse.
    # If exclusion is invalid (not blocked), we MUST include it.
    # So we only branch if exclusion is valid.
    # If exclusion is valid, we have two choices: Exclude or Include.
    # We try Exclude first.
    
    def search(in_set, out_set, undecided):
        nonlocal global_best_size, global_best_set, search_count
        
        if time.time() > deadline:
            return
        
        search_count += 1
        
        # Prune: if current size + min possible additions >= global_best_size
        # Min possible additions: 0 (if we can exclude all remaining)
        # But we must block all out_set.
        # If out_set has elements not blocked by in_set, we need more in_set.
        # For now, simple bound: len(in_set) >= global_best_size -> prune
        if len(in_set) >= global_best_size:
            return
        
        if not undecided:
            # Check if all out_set are blocked
            valid = True
            for v in out_set:
                if not is_blocked(v, in_set):
                    valid = False
                    break
            if valid:
                # This is a valid maximal set
                if len(in_set) < global_best_size:
                    global_best_size = len(in_set)
                    global_best_set = set(in_set)
            return
        
        # Pick a vertex to branch on
        # Heuristic: Pick a vertex that is "easy" to block? Or "hard"?
        # If a vertex is not blocked by in_set, we MUST include it.
        # So check if any undecided vertex is not blocked.
        # If yes, we are forced to include it. No branch.
        forced_include = None
        for v in undecided:
            if not is_blocked(v, in_set):
                forced_include = v
                break
        
        if forced_include is not None:
            # Must include forced_include
            if is_conflict(forced_include, in_set):
                return # Conflict, backtrack
            new_in = set(in_set)
            new_in.add(forced_include)
            new_undecided = set(undecided)
            new_undecided.remove(forced_include)
            search(new_in, out_set, new_undecided)
            return
        
        # All undecided vertices are blocked by in_set.
        # We can choose to exclude one or include one.
        # Try to exclude one first (to keep size small).
        # Pick a random undecided vertex to exclude.
        v = random.choice(list(undecided))
        
        # Branch 1: Exclude v
        new_out = set(out_set)
        new_out.add(v)
        new_undecided = set(undecided)
        new_undecided.remove(v)
        search(in_set, new_out, new_undecided)
        
        if time.time() > deadline:
            return
        
        # Branch 2: Include v
        if is_conflict(v, in_set):
            return
        new_in = set(in_set)
        new_in.add(v)
        new_undecided = set(undecided)
        new_undecided.remove(v)
        search(new_in, out_set, new_undecided)

    # Initialize B&B
    # Start with empty in_set, empty out_set, all undecided.
    # But we know we need at least some points.
    # To speed up, we can seed with a few points from the greedy solution?
    # Or just start from scratch.
    # Starting from scratch might be slow.
    # Let's try starting from the greedy solution's "core"?
    # No, B&B needs to explore the space.
    
    # To make B&B more effective, we can use the greedy solution to guide the initial state?
    # No, that biases the search.
    # Let's just run it.
    
    # Optimization: Use a list for undecided for faster iteration?
    # Sets are fine for small N.
    
    search(set(), set(), set(range(num_points)))
    
    # Return the best found
    if global_best_set:
        return [tuple(c) for c in coords[list(global_best_set)]]
    else:
        return [tuple(c) for c in coords[list(best_set)]]

# EVOLVE-BLOCK-END
