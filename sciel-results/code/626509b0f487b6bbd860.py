import time
import random
import itertools
import math
import sys

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute blocking relationships
# Two points i and j "block" point p if the triangle (p, i, j) is isosceles.
# We need to find a minimal set S such that for every p not in S, there exists (i, j) in S 
# such that p is blocked by i and j.
# This is equivalent to: S is a Dominating Set in the graph G where 
# edge (i, j) exists if there is some p blocked by i, j? No.
# Let's look at the definition again.
# p is blocked by S if there exist i, j in S such that {p, i, j} is isosceles.
# This means S must contain at least one pair from the set of "witness pairs" W_p for every p not in S.
# This is a complex constraint.
#
# Alternative View:
# Let's define a graph H on the grid points.
# For each point p, let B_p be the set of pairs {i, j} such that {p, i, j} is isosceles.
# We want to choose S such that:
# 1. S is independent in the "isosceles triangle graph" (no 3 points in S form an isosceles triangle).
# 2. For every p not in S, S intersects at least one pair in B_p.
#
# The second condition is "S is a 2-tuple dominating set" in a hypergraph sense.
#
# Let's try a different structural approach: Branch and Bound on the selection of S.
# We want to minimize |S|.
#
# Pruning Strategy:
# 1. If we have selected S, we can check if it's valid (no iso triangles).
# 2. We can check if it's maximal (all p not in S are blocked).
# 3. If not maximal, we MUST add points.
#
# Let's use a recursive search:
# def solve(selected, available, blocked_points):
#   - selected: set of points chosen so far
#   - available: set of points that can still be chosen (don't form iso with selected)
#   - blocked_points: set of points already blocked by selected pairs
#
#   If all points are either in selected or in blocked_points:
#       return selected (valid and maximal)
#   
#   We need to block the remaining unblocked points.
#   Let U = All Points - selected - blocked_points.
#   For each p in U, we must add at least one pair from B_p to selected.
#   Actually, we add points one by one.
#   
#   Heuristic: Pick a point p in U that has the fewest "options" to be blocked.
#   The options are pairs {i, j} in B_p such that i, j are in available (or selected).
#   Wait, if i is in selected, we only need to add j.
#   If i, j are both available, we need to add both? No, we add points to S.
#   If we add i to S, it might block p if there is j in S.
#   
#   This state space is huge.
#
# Let's stick to the "Minimum Maximal No-Isosceles Set" via Iterated Local Search but with a 
# much faster evaluation using bitsets and a stronger perturbation.
# The previous attempt failed because `is_maximal` was O(N^3) or worse.
#
# Optimization:
# Represent S as a bitset (Python int).
# Precompute for each point p a bitset `BLOCKED_BY[p]`? No, blocking requires a PAIR.
# 
# Let's precompute `ISO_MASK[i][j]`? No.
# Let's precompute `ISO_WITH[i]` = bitset of points k such that {i, k, m} is iso for some m? No.
#
# Let's go back to the pair-based blocking.
# For a set S (bitset), a point p is blocked if there exist i, j in S such that (i, j) is in WITNESS_PAIRS[p].
# This is equivalent to: The set of pairs in S must hit every WITNESS_PAIRS[p] for p not in S.
#
# Let's try a different formulation:
# A set S is maximal no-isosceles iff:
# 1. No 3 points in S form an isosceles triangle.
# 2. For every p not in S, there are i, j in S such that {p, i, j} is isosceles.
#
# We can use a greedy algorithm to build S, then minimize it.
# The key is fast checking.
#
# Let's use NumPy for vectorized distance checks if possible, but Python sets with precomputed lists are often faster for small N.
#
# Let's optimize the local search loop.
# 1. Precompute `ISO_TRIPLES_SET`: set of frozensets of 3 indices.
# 2. Precompute `WITNESS_PAIRS_BITSET[p]`: For each p, a list of pairs (i, j).
#    To check if p is blocked by S:
#    Check if any (i, j) in WITNESS_PAIRS[p] has both i and j in S.
#    This is O(len(WITNESS_PAIRS[p])) per p.
#    Total O(N_PTS * AvgPairs). AvgPairs is roughly O(N). So O(N^2) per maximality check.
#    N_PTS = 256. 256 * ~100 = 25k ops. This is fast in Python if optimized.
#
# Let's implement a highly optimized ILS.

# Precomputation
IDX_TO_POINT = {i: p for i, p in enumerate(ALL_POINTS)}
POINT_TO_IDX = {p: i for i, p in enumerate(ALL_POINTS)}

# Distance squared matrix
DIST2 = [[0]*N_PTS for _ in range(N_PTS)]
for i in range(N_PTS):
    pi = ALL_POINTS[i]
    for j in range(i+1, N_PTS):
        pj = ALL_POINTS[j]
        d2 = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2
        DIST2[i][j] = d2
        DIST2[j][i] = d2

# WITNESS_PAIRS[p]: list of (i, j) such that {p, i, j} is isosceles.
# We only need i, j != p.
WITNESS_PAIRS = [[] for _ in range(N_PTS)]
# Also, for fast "can_add" check:
# can_add(p, S) is False if there exist i, j in S such that {p, i, j} is isosceles.
# This is exactly the same condition as "p is blocked by S"!
# So, `can_add(p, S)` is equivalent to `NOT is_blocked(p, S)`.
# And `is_maximal(S)` is equivalent to `ALL p not in S: is_blocked(p, S)`.

# So we just need one function: `is_blocked(p, S)`.
# S is a set of indices.

# To speed up `is_blocked(p, S)`:
# We iterate through WITNESS_PAIRS[p].
# If (i, j) in WITNESS_PAIRS[p] and i in S and j in S, return True.

# Let's also precompute for each p, the witness pairs as a list of tuples.
# And maybe a set of (i, j) for faster lookup? No, list iteration is fine.

# Build WITNESS_PAIRS
# For each p, for each pair (i, j) with i < j, i != p, j != p:
# Check if {p, i, j} is isosceles.
# This is O(N^3) precomputation. 256^3 = 16M. Might be slow in pure Python (10-20s).
# Let's optimize precomputation.
# For a fixed p, we can iterate i, j.
# Or, for each pair (i, j), find all p that are blocked by (i, j).
# This is what we did before. Let's do that.

BLOCKING_PAIRS = [[] for _ in range(N_PTS)] # BLOCKING_PAIRS[p] = list of (i, j) that block p

# Iterate all pairs (i, j)
for i in range(N_PTS):
    pi = ALL_POINTS[i]
    for j in range(i+1, N_PTS):
        pj = ALL_POINTS[j]
        d2_ij = DIST2[i][j]
        
        # Find all p such that {p, i, j} is isosceles.
        # Conditions:
        # 1. d2(pi, p) == d2(pj, p)  (p on perp bisector)
        # 2. d2(pi, p) == d2_ij      (p on circle i)
        # 3. d2(pj, p) == d2_ij      (p on circle j)
        
        # We can iterate all p and check.
        for p in range(N_PTS):
            if p == i or p == j:
                continue
            pp = ALL_POINTS[p]
            d2_ip = DIST2[i][p]
            d2_jp = DIST2[j][p]
            
            if d2_ip == d2_jp or d2_ip == d2_ij or d2_jp == d2_ij:
                BLOCKING_PAIRS[p].append((i, j))

# Deduplicate? The conditions might overlap (e.g. equilateral). 
# If a pair blocks p via multiple conditions, it's still just one pair.
# We want unique pairs.
for p in range(N_PTS):
    BLOCKING_PAIRS[p] = list(set(BLOCKING_PAIRS[p]))

# Now we have BLOCKING_PAIRS[p].
# Function to check if p is blocked by S (set of indices)
def is_blocked(p, S):
    for i, j in BLOCKING_PAIRS[p]:
        if i in S and j in S:
            return True
    return False

# Function to check if S is maximal
def is_maximal(S):
    for p in range(N_PTS):
        if p not in S:
            if not is_blocked(p, S):
                return False
    return True

# Function to check if S is valid (no isosceles triangles)
# S is valid if for all p in S, p is NOT blocked by S\{p}?
# No. p is blocked by S if there exist i, j in S such that {p, i, j} is iso.
# If p is in S, and there exist i, j in S such that {p, i, j} is iso, then S is INVALID.
# So, S is valid iff for all p in S, is_blocked(p, S\{p}) is False?
# Wait. is_blocked(p, S) checks if there is a pair in S blocking p.
# If p is in S, and we check is_blocked(p, S), we might find a pair (i, j) in S.
# If such a pair exists, {p, i, j} is iso, so S is invalid.
# So, S is valid iff for all p in S, is_blocked(p, S) is False?
# Note: is_blocked uses pairs (i, j) from BLOCKING_PAIRS[p].
# BLOCKING_PAIRS[p] contains pairs (i, j) where i, j != p.
# So if is_blocked(p, S) is True, it means there are i, j in S (and i, j != p) such that {p, i, j} is iso.
# This is exactly the invalidity condition.
# So:
# S is Valid iff for all p in S, NOT is_blocked(p, S).
# S is Maximal iff for all p not in S, is_blocked(p, S).

# Combined check:
def is_valid_and_maximal(S):
    for p in range(N_PTS):
        if p in S:
            if is_blocked(p, S):
                return False
        else:
            if not is_blocked(p, S):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Greedy Initialization
    # We want a small set.
    # Random order, add if valid.
    # Then try to remove.
    
    def greedy_init():
        S = set()
        order = list(range(N_PTS))
        random.shuffle(order)
        for p in order:
            # Check if adding p keeps S valid.
            # Adding p is valid if p is not blocked by S.
            # i.e., NOT is_blocked(p, S).
            if not is_blocked(p, S):
                S.add(p)
        return S

    best_S = greedy_init()
    best_size = len(best_S)
    
    # Local Improvement: Remove points
    # If we remove p from S, S' = S \ {p}.
    # S' is valid if S was valid (removing points can't create iso triangles).
    # S' is maximal if all points not in S' are blocked by S'.
    # Points not in S' are (All \ S) U {p}.
    # Points in All \ S were blocked by S. Are they blocked by S'?
    # Maybe not, if they were blocked by a pair involving p.
    # And p itself must be blocked by S'.
    
    def try_remove(S):
        # Try to remove a random point
        if not S:
            return S
        p = random.choice(list(S))
        S_new = S - {p}
        # Check maximality of S_new
        if is_maximal(S_new):
            return S_new
        return S

    # Iterated Local Search
    current_S = best_S
    no_improve_count = 0
    
    while time.time() < deadline:
        # Perturb: Remove 1-3 random points
        k = random.randint(1, 3)
        S_perturbed = set(current_S)
        to_remove = random.sample(list(S_perturbed), min(k, len(S_perturbed)))
        for p in to_remove:
            S_perturbed.remove(p)
        
        # Repair: Add points to restore maximality
        # While not maximal, add a point that is not blocked.
        # To minimize size, we should add points that block many others?
        # Or just add any valid point.
        # Note: Adding a point p is valid if p is not blocked by current S.
        # If p is not blocked, it can be added.
        
        max_iter = 100
        for _ in range(max_iter):
            if is_maximal(S_perturbed):
                break
            # Find a point that can be added
            # A point p can be added if p not in S_perturbed and NOT is_blocked(p, S_perturbed).
            # If no such point exists, we are in a bad state?
            # If S is not maximal, there exists p not in S such that p is NOT blocked.
            # So there is always a point that can be added.
            addable = []
            for p in range(N_PTS):
                if p not in S_perturbed:
                    if not is_blocked(p, S_perturbed):
                        addable.append(p)
            if not addable:
                # Should not happen
                break
            # Choose one. Random is okay.
            # Better: Choose one that blocks the most other addable points?
            # For now, random.
            p_add = random.choice(addable)
            S_perturbed.add(p_add)
            
        # Local Improvement: Try to remove points
        improved = True
        while improved:
            improved = False
            # Try removing each point
            # Randomize order
            pts = list(S_perturbed)
            random.shuffle(pts)
            for p in pts:
                if time.time() > deadline:
                    break
                S_test = S_perturbed - {p}
                if is_maximal(S_test):
                    S_perturbed = S_test
                    improved = True
                    break # Restart loop as set changed
            
        if len(S_perturbed) < best_size:
            best_S = S_perturbed
            best_size = len(best_S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck for a while, restart from best?
            if no_improve_count > 50:
                current_S = best_S
                no_improve_count = 0
            else:
                current_S = S_perturbed
                
    return [IDX_TO_POINT[i] for i in best_S]

# Run construct once to ensure it works? No, the evaluator calls it.
# But we can do a quick sanity check in local execution if needed.
# The function `construct` is the entry point.

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s, Size: {len(res)}")
    # Verify
    S_idx = set(POINT_TO_IDX[p] for p in res)
    print("Valid and Maximal:", is_valid_and_maximal(S_idx))
