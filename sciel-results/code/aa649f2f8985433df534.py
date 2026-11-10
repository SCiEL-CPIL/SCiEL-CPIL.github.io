"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
TIME_LIMIT = 320.0  # Safety margin below 330s kill

# Precompute all squared distances between grid points
# Point index: i = x * N + y
# Total points: N*N
NUM_POINTS = N * N

# Precompute squared distance matrix: dist2[i, j]
# dist2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
xs = np.arange(N)
ys = np.arange(N)
# Create coordinate arrays for all points
coords = np.array([(x, y) for x in range(N) for y in range(N)])
# coords shape: (NUM_POINTS, 2)

# Compute all pairwise squared distances
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = (diff ** 2).sum(axis=2)  # shape: (NUM_POINTS, NUM_POINTS)

# For each pair of points (a, b), compute the set of points c that would form an isosceles triangle
# with a and b. This is the key optimization.
#
# A triangle (a, b, c) is isosceles if:
#   d(a,c) == d(a,b)  OR  d(b,c) == d(a,b)  OR  d(a,c) == d(b,c)
#
# For a fixed pair (a, b), the "forbidden" points are those c where any of these hold.
# We can precompute for each pair (a, b) a bitmask of forbidden points.

# For each pair (a, b), compute forbidden mask
# forbidden[a, b] is a bitmask (as a set of point indices) of points c that cannot be added
# if both a and b are already in the set.

# Actually, let's think differently. For a candidate point c to be added to a set S,
# for every pair (a, b) in S, the triangle (a, b, c) must NOT be isosceles.
# So c is forbidden if there exists a pair (a, b) in S such that (a, b, c) is isosceles.

# Let's precompute for each point c and each pair (a, b) whether (a, b, c) is isosceles.
# Then for a given set S, c is valid iff for all pairs (a,b) in S, (a,b,c) is NOT isosceles.

# Alternative: for each point c, precompute which pairs (a, b) would make (a, b, c) isosceles.
# Then c can be added iff no such pair is entirely in S.

# Let's precompute: for each c, the set of pairs (a, b) such that (a, b, c) is isosceles.
# This is a large structure. Let's use a different approach.

# For each pair (a, b), precompute the set of c's that form isosceles with a, b.
# iso_pairs[a, b] = set of c such that (a, b, c) is isosceles.

# Then for a set S, a candidate c is valid iff for all pairs (a, b) in S, c not in iso_pairs[a, b].
# Equivalently, c is invalid if there exists a pair (a, b) in S such that c in iso_pairs[a, b].

# Let's precompute iso_pairs as a 2D array of sets, or better, use bitmasks.

# Use numpy bitmasks: for each pair (a, b), a bitmask of NUM_POINTS bits.
# But NUM_POINTS = 256, so we need 256-bit masks. Python ints can handle this.

# Precompute: for each pair (a, b), a Python int bitmask of forbidden c's.
# iso_mask[a, b] = bitmask where bit c is set if (a, b, c) is isosceles.

# This is O(NUM_POINTS^3) = 256^3 = ~16 million operations, which is feasible.

# Let's compute iso_mask
iso_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

for a in range(NUM_POINTS):
    for b in range(a + 1, NUM_POINTS):
        mask = 0
        for c in range(NUM_POINTS):
            if c == a or c == b:
                continue
            da = dist2[a, c]
            db = dist2[b, c]
            dab = dist2[a, b]
            if da == db or da == dab or db == dab:
                mask |= (1 << c)
        iso_mask[a][b] = mask
        iso_mask[b][a] = mask

# Now, for a given set S (represented as a set of point indices), a candidate c is valid iff:
# For all pairs (a, b) in S, c is not in iso_mask[a][b].
# Equivalently, the union of iso_mask[a][b] for all pairs (a,b) in S does not contain c.

# Let's define: for a set S, the "forbidden set" F(S) = union of iso_mask[a][b] for all pairs (a,b) in S.
# Then c can be added iff c not in F(S) and c not in S.

# We can maintain F(S) incrementally. When we add a point p to S, the new forbidden set is:
# F(S ∪ {p}) = F(S) ∪ { union of iso_mask[p][q] for q in S }

# Let's implement a randomized greedy approach with multiple restarts.

def _can_add(c, S, forbidden_mask):
    """Check if point c can be added to set S.
    c: point index
    S: set of point indices
    forbidden_mask: int bitmask of all points that are forbidden (i.e., would form isosceles with some pair in S)
    """
    if c in S:
        return False
    if (forbidden_mask >> c) & 1:
        return False
    return True


def _greedy_construct(order, deadline):
    """Construct a maximal no-isosceles set using a given point ordering."""
    S = set()
    forbidden_mask = 0  # Union of iso_mask[a][b] for all pairs (a,b) in S
    
    for c in order:
        if time.time() > deadline:
            break
        if _can_add(c, S, forbidden_mask):
            # Add c to S
            # Update forbidden_mask: for each q in S, add iso_mask[c][q]
            for q in S:
                forbidden_mask |= iso_mask[c][q]
            S.add(c)
    
    # Now S is a no-isosceles set, but is it maximal?
    # A set is maximal if no further point can be added.
    # We need to check if S is maximal. If not, we need to add more points.
    # But wait, the greedy process above only adds points in the given order.
    # A set is maximal if for every point not in S, it cannot be added.
    # Let's check maximality.
    
    # Check if S is maximal
    for c in range(NUM_POINTS):
        if c not in S:
            if _can_add(c, S, forbidden_mask):
                # Not maximal, add c
                for q in S:
                    forbidden_mask |= iso_mask[c][q]
                S.add(c)
    
    return list(S)


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5.0  # 5 second safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # First, do a simple greedy in natural order to get a baseline
    natural_order = list(range(NUM_POINTS))
    S = _greedy_construct(natural_order, deadline)
    if len(S) < best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Now do randomized restarts
    rng = random.Random(42)
    
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Create a random ordering
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        S = _greedy_construct(order, deadline)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Also try a few variations: reverse, sorted by some heuristic
        if time.time() < deadline:
            # Try reverse of natural order
            if attempt % 3 == 0:
                order_rev = list(range(NUM_POINTS - 1, -1, -1))
                S = _greedy_construct(order_rev, deadline)
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = S[:]
        
        if time.time() < deadline:
            # Try ordering by x-coordinate then y-coordinate (different from natural)
            if attempt % 5 == 0:
                order_alt = list(range(NUM_POINTS))
                # Sort by y then x
                order_alt.sort(key=lambda idx: (idx % N, idx // N))
                S = _greedy_construct(order_alt, deadline)
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = S[:]
    
    if best_set is None:
        # Fallback: return a single point (trivially no-isosceles and maximal? No, not maximal)
        # Actually, a single point is not maximal. We need to return a maximal set.
        # Let's just do one more greedy if we didn't get any.
        S = _greedy_construct(list(range(NUM_POINTS)), deadline)
        return S
    
    return best_set


# Convert point indices to (x, y) tuples for the return value
def _index_to_point(idx):
    x = idx // N
    y = idx % N
    return (x, y)


# Override construct to return the correct format
_construct_original = construct

def construct():
    result_indices = _construct_original()
    return [_index_to_point(idx) for idx in result_indices]
