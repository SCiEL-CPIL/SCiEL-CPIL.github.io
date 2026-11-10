import itertools
import random
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist[i][j] = d
        dist[j][i] = d

# For each point k, precompute which pairs (a,b) with a,b != k would form isosceles with k
# Actually, let's precompute conflict[k] = set of points that cannot coexist with k in a no-isosceles set
# Two points a and b can coexist if there's no c such that (a,b,c) is isosceles
# But actually, the constraint is on triples. Let me think differently.

# A set S is no-isosceles if for all distinct a,b,c in S, no two of dist[a][b], dist[a][c], dist[b][c] are equal.
# For adding point k to set S: k conflicts with S if there exist a,b in S such that (k,a,b) is isosceles.

# Precompute for each k, the set of "forbidden pairs" - but that's expensive.
# Instead, for each k, precompute which points a have the property that dist[k][a] equals dist[a][b] for some b.
# This is still complex. Let's use a different approach.

# For each point k, compute conflict_mask[k] as a bitset where bit a is set if k and a "conflict" in the sense
# that they cannot both be in a no-isosceles set. But that's not quite right either because the constraint is on triples.

# Actually, two points can always coexist (a 2-point set has no triangles). The constraint only kicks in for 3+ points.
# So the "conflict" is really a 3-way constraint. 

# Let me precompute: for each pair (a,b), the set of points c such that (a,b,c) is isosceles.
# Then when checking if k can be added to S, for each pair (a,b) in S, check if k is in the isosceles set for (a,b).

# Better: for each k, precompute iso_conflict[k] as a bitset. Bit a is set in iso_conflict[k] if... 
# Hmm, this doesn't directly work because the isosceles condition depends on three points.

# Let me try: for each k, and for each a, compute the set of b such that (k,a,b) is isosceles.
# Then k can be added to S iff for all a in S, no b in S satisfies that (k,a,b) is isosceles.

# Precompute: iso[k][a] = bitset of b such that (k,a,b) forms isosceles triangle
# where k,a,b are distinct.

iso = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for k in range(NUM_POINTS):
    for a in range(NUM_POINTS):
        if k == a:
            continue
        dk = dist[k][a]
        mask = 0
        for b in range(NUM_POINTS):
            if b == k or b == a:
                continue
            da = dist[k][b]
            dab = dist[a][b]
            if dk == da or dk == dab or da == dab:
                mask |= (1 << b)
        iso[k][a] = mask
        iso[a][k] = mask  # symmetric in the sense that if (k,a,b) is iso, (a,k,b) is iso

def can_add_bitset(k, S_mask):
    """Check if point k can be added to set represented by S_mask (bitset)."""
    for a_bit in range(NUM_POINTS):
        if not (S_mask & (1 << a_bit)):
            continue
        a = a_bit
        # Check if any b in S_mask has (k,a,b) isosceles
        if iso[k][a] & S_mask:
            return False
    return True

def is_maximal_bitset(S_mask):
    """Check if set S_mask is maximal (no point can be added)."""
    for k in range(NUM_POINTS):
        if S_mask & (1 << k):
            continue
        if can_add_bitset(k, S_mask):
            return False
    return True

def greedy_maximal_bitset(order):
    """Construct a maximal set using the given order."""
    S_mask = 0
    for k in order:
        if can_add_bitset(k, S_mask):
            S_mask |= (1 << k)
    return S_mask

def mask_to_list(mask):
    return [i for i in range(NUM_POINTS) if mask & (1 << i)]

def shrink_bitset(S_mask):
    """Try to remove points from S_mask while maintaining maximality."""
    improved = True
    while improved:
        improved = False
        for p in range(NUM_POINTS):
            if not (S_mask & (1 << p)):
                continue
            S_trial = S_mask & ~(1 << p)
            if is_maximal_bitset(S_trial):
                S_mask = S_trial
                improved = True
                break
    return S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    best_mask = None
    best_size = NUM_POINTS + 1
    
    rng = random.Random(42)
    
    # Initial greedy with row-major
    base_order = list(range(NUM_POINTS))
    S_base = greedy_maximal_bitset(base_order)
    S_base_shrunk = shrink_bitset(S_base)
    best_mask = S_base_shrunk
    best_size = bin(S_base_shrunk).count('1')
    
    # Randomized greedy attempts
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        S = greedy_maximal_bitset(order)
        S_shrunk = shrink_bitset(S)
        
        sz = bin(S_shrunk).count('1')
        if sz < best_size:
            best_mask = S_shrunk
            best_size = sz
    
    return [POINTS[i] for i in mask_to_list(best_mask)]
