import time
import random
import itertools
import sys
from collections import defaultdict

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = N * N
DEADLINE = 305.0  # Safety margin from 330s

# Precompute distance squared
def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# Precompute for each point p, the set of pairs (a,b) such that (p,a,b) is isosceles
# and p, a, b are distinct.
# We store this as a set of frozenset pairs for fast lookup.
# IsoPairs[p] = set of frozenset({a, b})

print("Precomputing isosceles structures...", file=sys.stderr)
start_time = time.time()

# For efficiency, we precompute:
# 1. IsoPairs[p]: set of pairs {a,b} in POINTS x POINTS (a!=b, a!=p, b!=p)
#    such that (p,a,b) is isosceles.
# 2. We also need to know, for a given set S, which points p are "blocked" by S.
#    p is blocked if there exist a,b in S such that {a,b} in IsoPairs[p].
#    But we also need S to be independent (no isosceles triangles within S).

# Let's use bitmasks for faster operations? 256 bits fits in a Python int or a few ints.
# But let's stick to sets for clarity first, and optimize if needed.

# Precompute IsoPairs
IsoPairs = {p: set() for p in POINTS}

# To speed up precomputation, iterate over pairs (a,b) and find p.
# For a fixed pair (a,b), which p form an isosceles triangle with (a,b)?
# The locus of points p such that (p,a,b) is isosceles is the union of:
# - Perpendicular bisector of ab (where pa=pb)
# - Circle centered at a with radius ab (where pa=ab)
# - Circle centered at b with radius ab (where pb=ab)
#
# But iterating over all p for each pair is O(N^6) = 16^6 = 16M, which might be slow in Python.
# 16M * constant. Let's try to optimize or just run it and see.
# Actually, 16M iterations in Python might take ~10-20 seconds. That's acceptable.

# Let's optimize the inner loop.
# For each pair (a, b):
#   d_ab = dist2(a,b)
#   for p in POINTS:
#      if p == a or p == b: continue
#      d_pa = dist2(p,a)
#      d_pb = dist2(p,b)
#      if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
#         add {a,b} to IsoPairs[p]

# This is O(P * P * P) = 16M.
# We can precompute dist2 for all pairs? 256*256 = 65536 distances. That's small.
# Let's precompute a dist2 matrix.
# dist2_matrix[i][j] = dist2(POINTS[i], POINTS[j])

dist2_matrix = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
point_index = {p: i for i, p in enumerate(POINTS)}

for i in range(NUM_POINTS):
    pi = POINTS[i]
    for j in range(NUM_POINTS):
        pj = POINTS[j]
        dist2_matrix[i][j] = (pi[0]-pj[0])**2 + (pi[1]-pj[1])**2

# Now precompute IsoPairs using indices for speed?
# Let's use indices for internal representation to speed up set operations?
# Or just use tuples. Tuples are hashable and fast enough for small sizes.

# Let's try the direct iteration with precomputed dist2 matrix.
for i in range(NUM_POINTS):
    a = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        b = POINTS[j]
        d_ab = dist2_matrix[i][j]
        # For each p, check if (p,a,b) is isosceles
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            p = POINTS[k]
            d_pa = dist2_matrix[k][i]
            d_pb = dist2_matrix[k][j]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                # (p, a, b) is isosceles
                # Add {a, b} to IsoPairs[p]
                pair = frozenset((a, b))
                IsoPairs[p].add(pair)

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

# Now we have IsoPairs.
# We also need to check if a set S is "no-isosceles".
# This means no three points in S form an isosceles triangle.
# Equivalent to: For all a,b,c in S, {a,b} is NOT in IsoPairs[c].
# Or: For all c in S, no pair in S is in IsoPairs[c].

# We also need to check maximality.
# S is maximal if for every p not in S, p is "blocked" by S.
# p is blocked if there exist a,b in S such that {a,b} in IsoPairs[p].

# Let's define helper functions.

def _is_no_isosceles(S):
    """Check if S contains no isosceles triangle."""
    S_list = list(S)
    for c in S_list:
        # For each pair a,b in S, check if {a,b} in IsoPairs[c]
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                if a == c or b == c:
                    continue
                if frozenset((a, b)) in IsoPairs[c]:
                    return False
    return True

def _is_maximal(S):
    """Check if S is maximal (no point can be added)."""
    S_set = set(S)
    for p in POINTS:
        if p in S_set:
            continue
        # Check if p is blocked by S
        blocked = False
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                if frozenset((a, b)) in IsoPairs[p]:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return False
    return True

# The problem is to find a minimal size maximal no-isosceles set.
# This is a combinatorial optimization problem.
# Let's try a local search / greedy approach.

# Strategy:
# 1. Start with a random maximal set (by adding points until maximal).
# 2. Try to remove points one by one, checking if the set remains maximal and no-isosceles.
#    Wait, removing a point from a maximal set makes it non-maximal usually?
#    No, if we remove p, S' = S \ {p}.
#    S' is no-isosceles (subset of no-isosceles).
#    But S' might not be maximal.
#    We want S' to be maximal.
#    So we can only remove p if S' is still maximal.
#    This is a very restrictive condition.
#    
#    Actually, the definition of "maximal no-isosceles set" is:
#    A set S is maximal if S is no-isosceles and for every p not in S, S U {p} is NOT no-isosceles.
#    So if we remove p from S, S' = S \ {p}.
#    Is S' maximal?
#    We need to check if for every q not in S', S' U {q} is not no-isosceles.
#    This is expensive to check.
#    
#    Alternative strategy:
#    Construct a set S that is no-isosceles.
#    Then, add points to make it maximal? No, adding points violates no-isosceles.
#    
#    Let's rethink.
#    We want a set S such that:
#    1. No isosceles triangles in S.
#    2. For every p not in S, there exist a,b in S such that (p,a,b) is isosceles.
#    
#    This is equivalent to finding a "hitting set" of isosceles triangles, but with the constraint that the selected points themselves don't form isosceles triangles.
#    
#    Let's try a greedy construction:
#    Start with S = {}.
#    While S is not maximal:
#       Find a point p that can be added to S without violating no-isosceles.
#       Among all such p, choose the one that "blocks" the most new points.
#       Add p to S.
#    
#    "Blocks" q if (q, a, b) is isosceles for some a,b in S U {p}.
#    Since S is no-isosceles, the only new isosceles triangles involving q must involve p and some a in S.
#    So p blocks q if there exists a in S such that (q, p, a) is isosceles.
#    Also, if S is empty, p blocks nothing?
#    If S = {p}, does p block q?
#    We need a,b in S such that (q,a,b) is isosceles.
#    If S={p}, we need (q,p,p) isosceles. But q,p,p are not distinct.
#    Usually, isosceles triangles require 3 distinct points.
#    If distinct points are required, then S={p} blocks nothing.
#    We need at least 2 points in S to block any point.
#    
#    So, let's start with a pair (a,b).
#    Which pair is best?
#    A pair (a,b) blocks q if (q,a,b) is isosceles.
#    So we want a pair (a,b) that forms isosceles triangles with many q.
#    
#    Let's precompute for each pair (a,b), the number of q such that (q,a,b) is isosceles.
#    This is the "coverage" of the pair.
    
# Precompute pair coverage
print("Precomputing pair coverage...", file=sys.stderr)
pair_coverage = {}
for i in range(NUM_POINTS):
    a = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        b = POINTS[j]
        pair = frozenset((a, b))
        count = 0
        for p in POINTS:
            if p == a or p == b:
                continue
            if pair in IsoPairs[p]:
                count += 1
        pair_coverage[pair] = count

# Sort pairs by coverage descending
sorted_pairs = sorted(pair_coverage.items(), key=lambda x: x[1], reverse=True)

# Now, greedy construction:
# 1. Pick a high-coverage pair (a,b) to start.
# 2. S = {a, b}.
# 3. Check if S is maximal.
# 4. If not, find a point p to add.
#    p must be such that S U {p} is no-isosceles.
#    Among valid p, pick the one that maximizes the number of newly blocked points.
#    Newly blocked points are those q not in S U {p} that are blocked by S U {p} but not by S.
#    Since S was not blocking q, the new blocking must involve p.
#    So p blocks q if there exists c in S such that (q, p, c) is isosceles.
#    (Note: c can be a or b).
#    Also, if S has only a,b, then p blocks q if (q,p,a) or (q,p,b) is isosceles.

def _greedy_construct():
    # Pick a random top pair
    top_pairs = sorted_pairs[:50]
    random.shuffle(top_pairs)
    pair, _ = top_pairs[0]
    a, b = list(pair)
    S = {a, b}
    
    # Check if S is no-isosceles (trivially yes for 2 points)
    # Check if S is maximal
    while not _is_maximal(S):
        # Find candidates p
        candidates = []
        for p in POINTS:
            if p in S:
                continue
            # Check if S U {p} is no-isosceles
            valid = True
            S_list = list(S)
            for c in S_list:
                # Check if (p, c, x) is isosceles for any x in S
                # Actually, we just need to check if p forms an isosceles triangle with any pair in S.
                # Since S has size k, there are k*(k-1)/2 pairs.
                # We can check if any pair in S is in IsoPairs[p].
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        if frozenset((S_list[i], S_list[j])) in IsoPairs[p]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                candidates.append(p)
                
        if not candidates:
            # Stuck, return current S (might not be maximal, but let's hope it is)
            break
            
        # Score each candidate
        best_p = None
        best_score = -1
        best_cands = []
        
        for p in candidates:
            score = 0
            # Count how many points q are blocked by S U {p} but not by S
            for q in POINTS:
                if q in S or q == p:
                    continue
                # Check if q is blocked by S
                blocked_by_S = False
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        if frozenset((S_list[i], S_list[j])) in IsoPairs[q]:
                            blocked_by_S = True
                            break
                    if blocked_by_S:
                        break
                if blocked_by_S:
                    continue
                
                # Check if q is blocked by S U {p}
                # This means there exist a,b in S U {p} such that (q,a,b) is isosceles.
                # Since q was not blocked by S, the pair {a,b} must involve p.
                # So we need c in S such that (q, p, c) is isosceles.
                # i.e., {p, c} in IsoPairs[q].
                blocked_by_Sp = False
                for c in S:
                    if frozenset((p, c)) in IsoPairs[q]:
                        blocked_by_Sp = True
                        break
                if blocked_by_Sp:
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_cands = [p]
            elif score == best_score:
                best_cands.append(p)
                
        if best_cands:
            p = random.choice(best_cands)
            S.add(p)
            
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = float('inf')
    
    # Try multiple random initializations
    for _ in range(100):
        if time.time() > DEADLINE:
            break
            
        S = _greedy_construct()
        if S is None:
            continue
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Local search: try to remove points
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            S_list = list(best_set)
            random.shuffle(S_list)
            for p in S_list:
                if time.time() > DEADLINE:
                    break
                S_temp = best_set - {p}
                if len(S_temp) < 2:
                    continue
                if _is_no_isosceles(S_temp) and _is_maximal(S_temp):
                    best_set = S_temp
                    best_size -= 1
                    improved = True
                    break
                    
    if best_set is None:
        # Fallback: return a valid maximal set
        # Just pick the first valid pair and expand
        S = _greedy_construct()
        if S:
            return list(S)
        return []
        
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
