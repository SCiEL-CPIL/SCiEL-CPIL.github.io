import numpy as np
import random
import time
import itertools

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute distance matrix
D = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        d = dx * dx + dy * dy
        D[i, j] = d
        D[j, i] = d

def is_isosceles_idx(v, a, b):
    d_va = D[v, a]
    d_vb = D[v, b]
    d_ab = D[a, b]
    return d_va == d_vb or d_va == d_ab or d_vb == d_ab

# Precompute conflict bitsets: for each point v, which points a conflict with v
# A point a conflicts with v if there exists some b such that (v,a,b) forms an isosceles triangle
# Actually, we need: for a set S to be maximal, every point not in S must form an isosceles triangle with some pair in S
# So for each v, precompute which pairs (a,b) would block v
# Better: for each v, precompute the set of points that "cover" v (i.e., for each point a, does a help block v?)

# For maximality: v is blocked by S if there exist a,b in S such that (v,a,b) is isosceles
# Equivalently: v is NOT blocked by S if for all a,b in S, (v,a,b) is NOT isosceles
# So v is blocked iff S contains at least one pair (a,b) from the set of pairs that block v

# Precompute for each v: the list of pairs (a,b) that would block v
# And for each v: the set of points that are "compatible" with v in the sense that they can form blocking pairs

# Actually, let's think about this differently.
# We want a minimal maximal independent set in the 3-uniform hypergraph.
# Independent set: no three points form an isosceles triangle
# Maximal: every point not in the set forms an isosceles triangle with some pair in the set

# For the branch-and-bound:
# State: (included, excluded, undecided)
# included: points we've decided to include
# excluded: points we've decided to exclude  
# undecided: remaining points

# Constraints:
# 1. No three included points form an isosceles triangle
# 2. Every excluded point must be "blocked" by included points (forms isosceles with some pair in included)
# 3. Every undecided point must either be includable or be blocked

# Let's precompute:
# conflict[a][b] = set of points v such that (v,a,b) is isosceles
# This tells us: if a and b are both included, then all v in conflict[a][b] cannot be included

# For efficiency, use bitsets (Python ints as bit vectors)
NUM = NUM_POINTS
all_mask = (1 << NUM) - 1

# Precompute conflict bitsets
# conflict_mask[v] = bitmask of points that conflict with v (i.e., if v is included, these points cannot be in the set along with some other point)
# Actually, the conflict is 3-way: (v,a,b) is isosceles means v,a,b cannot all be in the set
# So if v and a are both in the set, then b cannot be in the set for any b in conflict[v][a]

# Let's precompute: for each pair (v,a), the set of b such that (v,a,b) is isosceles
# conflict_pair[v][a] = bitmask of b values

conflict_pair = [[0] * NUM for _ in range(NUM)]
for v in range(NUM):
    for a in range(v + 1, NUM):
        mask = 0
        for b in range(NUM):
            if b == v or b == a:
                continue
            if is_isosceles_idx(v, a, b):
                mask |= (1 << b)
        conflict_pair[v][a] = mask
        conflict_pair[a][v] = mask

# For maximality: point v is blocked by set S if there exist a,b in S such that (v,a,b) is isosceles
# Equivalent: v is blocked by S iff S intersects with the "blocking set" for v
# But the blocking depends on pairs, not single points
# v is blocked by S iff there exist a,b in S such that (v,a,b) is isosceles
# This is equivalent to: the set S contains at least one edge from the 2-uniform hypergraph H_v
# where H_v has edges (a,b) for all a,b such that (v,a,b) is isosceles

# For the search, let's use a different approach:
# Greedy construction with smart ordering, then local search with bitset operations

def greedy_construct(order, deadline):
    """Build a maximal independent set greedily in the given order."""
    S = []
    S_mask = 0
    for p in order:
        if time.time() > deadline:
            break
        # Check if p can be added: no isosceles triangle with any pair in S
        can_add = True
        for i in range(len(S)):
            a = S[i]
            # Check if (p, a, any other in S) forms isosceles
            # p and a are in S, so for each b in S, check if (p,a,b) is isosceles
            # Equivalent: conflict_pair[p][a] & S_mask != 0
            if conflict_pair[p][a] & S_mask:
                can_add = False
                break
        if can_add:
            S.append(p)
            S_mask |= (1 << p)
    return S, S_mask

def is_maximal_bitset(S, S_mask):
    """Check if S is maximal using bitset operations."""
    for v in range(NUM):
        if S_mask & (1 << v):
            continue
        # v is not in S, check if v is blocked by S
        # v is blocked if there exist a,b in S such that (v,a,b) is isosceles
        # This means: the set of points in S that form isosceles triangles with v
        # For each a in S, check if conflict_pair[v][a] & S_mask != 0
        blocked = False
        # Iterate through points in S
        temp_mask = S_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & (-temp_mask)
            a = lsb.bit_length() - 1
            if conflict_pair[v][a] & S_mask:
                blocked = True
                break
            temp_mask ^= lsb
        if not blocked:
            return False
    return True

def local_search_remove(S, S_mask, deadline):
    """Try to remove points from S while maintaining maximality."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_copy = S[:]
        random.shuffle(S_copy)
        for p in S_copy:
            if time.time() > deadline:
                return S, S_mask, False
            # Try removing p
            new_S = [x for x in S if x != p]
            new_S_mask = S_mask & ~(1 << p)
            if not new_S:
                continue
            # Check if new_S is still maximal
            if is_maximal_bitset(new_S, new_S_mask):
                S = new_S
                S_mask = new_S_mask
                improved = True
                break
    return S, S_mask, improved

def local_search_swap(S, S_mask, deadline):
    """Try to swap points in S to reduce size."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_copy = S[:]
        random.shuffle(S_copy)
        for p in S_copy:
            if time.time() > deadline:
                return S, S_mask, False
            # Try removing p and adding some other point
            # First, find points that could replace p
            # A point q can be added if it doesn't form isosceles with any pair in S\{p}
            # And after adding q, the set must still be maximal
            
            # For now, just try simple removals
            new_S = [x for x in S if x != p]
            new_S_mask = S_mask & ~(1 << p)
            if not new_S:
                continue
            if is_maximal_bitset(new_S, new_S_mask):
                S = new_S
                S_mask = new_S_mask
                improved = True
                break
    return S, S_mask, improved

def construct():
    best_set = None
    best_size = float('inf')
    deadline = time.time() + 285  # Safety margin
    
    random.seed(42)
    
    # Multiple randomized greedy trials
    num_trials = 20
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if trial % 3 == 0:
            order = list(range(NUM))
            random.shuffle(order)
        elif trial % 3 == 1:
            # Priority by degree (points with more conflicts first)
            # Points that are in many isosceles triangles should be considered first
            # Actually, we want to include points that block many others
            # Let's try: sort by number of points they conflict with
            degrees = []
            for p in range(NUM):
                count = 0
                for a in range(NUM):
                    if a != p:
                        count += bin(conflict_pair[p][a]).count('1')
                degrees.append((count, p))
            degrees.sort(key=lambda x: -x[0])
            order = [p for _, p in degrees]
            random.shuffle(order)
        else:
            # Random with some structure
            order = list(range(NUM))
            random.shuffle(order)
        
        S, S_mask = greedy_construct(order, deadline)
        
        # Local search
        S, S_mask, _ = local_search_remove(S, S_mask, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
            # If we found a good solution, do more intensive local search
            if best_size <= 20 and time.time() < deadline - 30:
                # Multiple rounds of local search
                for _ in range(5):
                    if time.time() > deadline - 10:
                        break
                    S, S_mask, improved = local_search_remove(S, S_mask, deadline - 10)
                    if not improved:
                        break

    if best_set is None:
        # Fallback
        S, _ = greedy_construct(list(range(NUM)), deadline)
        best_set = S

    return [POINTS[i] for i in best_set]
