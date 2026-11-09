import numpy as np
import itertools
import time
import random
import math

N = 16
POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=int)
NUM_POINTS = len(POINTS)

# Precompute squared distances between all pairs of points
diff = POINTS[:, np.newaxis, :] - POINTS[np.newaxis, :, :]
D = np.sum(diff * diff, axis=2)

# For each point p, and for each pair (a,b), determine if {p,a,b} forms an isosceles triangle
# iso_conflict[p, a, b] = True if adding p to a set containing both a and b creates isosceles
# We also need: for point p, if we have set S, p conflicts with S if:
#   1. There exist a,b in S such that d(p,a) == d(p,b) [apex at p]
#   2. There exist a,b in S such that d(p,a) == d(a,b) [apex at a]
#   3. There exist a,b in S such that d(p,b) == d(a,b) [apex at b]
# Note: cases 2 and 3 are symmetric.

# Precompute for each point p, the set of distance values from p to all other points
# And for each point a, the set of distances from a to all other points

# For fast conflict checking with bitmask:
# conflict_mask[p] is a 2D boolean array where conflict_mask[p][a][b] = True
# if {p, a, b} forms an isosceles triangle (for a != b, a != p, b != p)

# Actually, let's precompute: for each p, and for each a, the set of b such that {p,a,b} is isosceles
# This is a 3D boolean array: ISO[p, a, b]

ISO = np.zeros((NUM_POINTS, NUM_POINTS, NUM_POINTS), dtype=bool)

for p in range(NUM_POINTS):
    for a in range(NUM_POINTS):
        if p == a:
            continue
        for b in range(NUM_POINTS):
            if p == b or a == b:
                continue
            d_pa = D[p, a]
            d_pb = D[p, b]
            d_ab = D[a, b]
            # Check if any two distances are equal
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                ISO[p, a, b] = True
                ISO[p, b, a] = True
                ISO[a, p, b] = True
                ISO[a, b, p] = True
                ISO[b, p, a] = True
                ISO[b, a, p] = True

# For each point p, precompute which pairs (a,b) cause a conflict
# conflict_pairs[p] = set of (a,b) pairs such that ISO[p,a,b] is True
# For bitmask operations, we'll use a different approach

def _check_conflict_bitmask(p_idx, mask):
    """
    Check if adding point p_idx to the set represented by bitmask 'mask' creates an isosceles triangle.
    mask is a Python int where bit i is set if point i is in the set.
    """
    # Get indices of points in the set
    s_indices = [i for i in range(NUM_POINTS) if (mask >> i) & 1]
    if len(s_indices) < 2:
        return False
    
    # Check using precomputed ISO
    for a in s_indices:
        for b in s_indices:
            if a == b:
                continue
            if ISO[p_idx, a, b]:
                return True
    return False

def _is_maximal_bitmask(mask):
    """
    Check if the set represented by bitmask 'mask' is maximal.
    """
    for p_idx in range(NUM_POINTS):
        if (mask >> p_idx) & 1:
            continue
        if not _check_conflict_bitmask(p_idx, mask):
            return False
    return True

def _greedy_maximal(order):
    """
    Greedily build a maximal set by adding points in the given order.
    Returns a bitmask.
    """
    mask = 0
    for p_idx in order:
        if not _check_conflict_bitmask(p_idx, mask):
            mask |= (1 << p_idx)
    return mask

def _simulated_annealing(initial_mask, time_limit, rng):
    """
    Simulated annealing with swap moves: remove one point, add another.
    The objective is to minimize the size of the set while maintaining maximality.
    We allow temporary non-maximal states with a penalty.
    """
    current_mask = initial_mask
    current_size = bin(current_mask).count('1')
    
    # Energy function: size + penalty * (number of non-maximal points)
    # A set is non-maximal if there exists a point outside that can be added
    # penalty should be large enough to strongly prefer maximal sets
    
    def energy(mask):
        size = bin(mask).count('1')
        # Count how many points outside can be added
        addable = 0
        for p in range(NUM_POINTS):
            if (mask >> p) & 1:
                continue
            if not _check_conflict_bitmask(p, mask):
                addable += 1
        # If addable > 0, the set is not maximal
        # We want to minimize size, but strongly penalize non-maximality
        return size + 100 * addable
    
    current_energy = energy(current_mask)
    best_mask = current_mask
    best_energy = current_energy
    
    T = 50.0  # Initial temperature
    cooling = 0.9995
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            break
        
        # Generate a move: swap one point out and one point in
        in_points = [i for i in range(NUM_POINTS) if (current_mask >> i) & 1]
        out_points = [i for i in range(NUM_POINTS) if not (current_mask >> i) & 1]
        
        if not in_points or not out_points:
            break
        
        remove_idx = rng.choice(in_points)
        add_idx = rng.choice(out_points)
        
        new_mask = (current_mask & ~(1 << remove_idx)) | (1 << add_idx)
        new_energy = energy(new_mask)
        
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_mask = new_mask
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_mask = current_mask
        
        T *= cooling
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    best_mask = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Generate diverse initial maximal sets using different strategies
    
    # Strategy 1: Random order greedy
    for restart in range(50):
        if time.time() > deadline - 60:
            break
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        mask = _greedy_maximal(order)
        size = bin(mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = mask
    
    # Strategy 2: Greedy with "blocking power" heuristic
    # Points that block many other points when added are preferred
    # Compute blocking power: for each point p, how many other points q
    # would conflict with p if both were in a set with some common neighbor?
    # Simpler: just use random but with different seeds
    
    for restart in range(50):
        if time.time() > deadline - 60:
            break
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        mask = _greedy_maximal(order)
        size = bin(mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Simulated annealing on the best found
    if best_mask is not None and time.time() < deadline - 20:
        remaining = deadline - 20 - time.time()
        # Run SA with time budget
        sa_result = _simulated_annealing(best_mask, remaining, rng)
        sa_size = bin(sa_result).count('1')
        if sa_size < best_size:
            best_size = sa_size
            best_mask = sa_result
    
    # Phase 3: Final local search (removal only) on best
    if best_mask is not None and time.time() < deadline - 5:
        current = best_mask
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            in_points = [i for i in range(NUM_POINTS) if (current >> i) & 1]
            rng.shuffle(in_points)
            for p in in_points:
                new_mask = current & ~(1 << p)
                if _is_maximal_bitmask(new_mask):
                    current = new_mask
                    improved = True
                    break
        if bin(current).count('1') < best_size:
            best_size = bin(current).count('1')
            best_mask = current
    
    if best_mask is None:
        # Fallback
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        best_mask = _greedy_maximal(order)
    
    # Convert bitmask to list of (x,y) tuples
    result = []
    for i in range(NUM_POINTS):
        if (best_mask >> i) & 1:
            result.append(tuple(POINTS[i]))
    
    return result
