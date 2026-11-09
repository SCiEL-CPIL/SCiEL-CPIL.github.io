import numpy as np
import itertools
import time
import random

N = 16
POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=int)
NUM_POINTS = len(POINTS)

# Precompute squared distances between all pairs of points
# D[i, j] = squared distance between POINTS[i] and POINTS[j]
diff = POINTS[:, np.newaxis, :] - POINTS[np.newaxis, :, :]
D = np.sum(diff * diff, axis=2)

# Precompute for each point p, and each other point a, the set of points b such that d(p,a) == d(a,b)
# This is the "apex a" conflict: triangle p-a-b is isosceles with apex a.
# For a given p and a, d_pa is fixed. We need b such that D[a,b] == D[p,a].
# Let's precompute a boolean matrix or bitsets for efficiency.
# Since NUM_POINTS = 256, we can use integers as bitsets.

# conflict_bitset[p, a] is a bitmask of all b such that D[a,b] == D[p,a]
# Note: b can be p itself (D[a,p] == D[p,a]), but that's fine, we just need to check if any b in S is in this set.
# Actually, if b=p, then the triangle is degenerate or p is in S, which is handled by the "apex p" check or just not adding p.
# The condition for conflict when adding p to S is:
# 1. Apex p: exists A, B in S such that D[p,A] == D[p,B].
# 2. Apex A: exists A, B in S such that D[p,A] == D[A,B].
# 3. Apex B: exists A, B in S such that D[p,B] == D[B,A]. (Same as 2, just swap A and B).

# Let's precompute:
# For each pair (p, a), the set of b such that D[a,b] == D[p,a].
# Let's call this iso_set[p, a].

# We can represent iso_set[p, a] as an integer bitmask.
# 256 bits fits in 4 64-bit integers, or we can use Python's arbitrary precision integers.
# Python integers are efficient for bit operations.

iso_bitsets = np.zeros((NUM_POINTS, NUM_POINTS), dtype=object)

for p in range(NUM_POINTS):
    for a in range(NUM_POINTS):
        if p == a:
            continue
        d_pa = D[p, a]
        # Find all b such that D[a, b] == d_pa
        # D[a, :] is the row of distances from a
        # We need indices b where D[a, b] == d_pa
        # Note: b can be any point.
        # Let's use numpy to find indices
        b_indices = np.where(D[a, :] == d_pa)[0]
        # Create bitmask
        mask = 0
        for b in b_indices:
            mask |= (1 << b)
        iso_bitsets[p, a] = mask

def _is_conflict(p_idx, s_indices):
    """
    Check if point p_idx conflicts with set s_indices.
    Returns True if adding p_idx to s_indices creates an isosceles triangle.
    """
    if len(s_indices) < 2:
        return False
    
    # 1. Check apex p_idx: any two points in S at same distance from p_idx?
    d_p = D[p_idx, s_indices]
    if len(np.unique(d_p)) < len(d_p):
        return True
        
    # 2. Check apex A in S: for any A in S, is there a B in S such that D[p_idx, A] == D[A, B]?
    # This is equivalent to: does the bitmask of S intersect with iso_bitsets[p_idx, A]?
    # Note: iso_bitsets[p_idx, A] includes B such that D[A,B] == D[p_idx,A].
    # If B is in S, then we have a conflict.
    # We need to check if any B in S is in iso_bitsets[p_idx, A].
    # Note: B can be p_idx? No, p_idx is not in S.
    # B can be A? D[A,A]=0. D[p_idx,A] > 0. So A is not in iso_bitsets unless D[p_idx,A]==0 which is false.
    
    s_mask = 0
    for s in s_indices:
        s_mask |= (1 << s)
        
    for a_idx in s_indices:
        # Check if S intersects iso_bitsets[p_idx, a_idx]
        if s_mask & iso_bitsets[p_idx, a_idx]:
            return True
            
    return False

def _is_maximal(s_indices):
    """
    Check if set s_indices is maximal.
    Maximal means no point outside s_indices can be added without conflict.
    """
    s_set = set(s_indices)
    s_mask = 0
    for s in s_indices:
        s_mask |= (1 << s)
        
    for p_idx in range(NUM_POINTS):
        if p_idx in s_set:
            continue
        # Check if p_idx can be added
        # 1. Apex p_idx
        d_p = D[p_idx, s_indices]
        if len(d_p) >= 2:
            if len(np.unique(d_p)) < len(d_p):
                continue # Conflict, so p_idx cannot be added. Good for maximality.
            # 2. Apex A in S
            conflict_found = False
            for a_idx in s_indices:
                if s_mask & iso_bitsets[p_idx, a_idx]:
                    conflict_found = True
                    break
            if conflict_found:
                continue # Conflict, so p_idx cannot be added. Good for maximality.
        
        # If we reach here, p_idx can be added.
        # So the set is NOT maximal.
        return False
    return True

def _greedy_random_order(order):
    """
    Greedily add points in the given order.
    Returns the resulting maximal set.
    """
    current = []
    current_set = set()
    for p_idx in order:
        if not _is_conflict(p_idx, current):
            current.append(p_idx)
            current_set.add(p_idx)
    return current

def _local_search(s_indices, max_iters=1000, rng=None):
    """
    Local search: try to remove points.
    If removing a point keeps the set maximal, remove it.
    """
    if rng is None:
        rng = random.Random()
        
    current = list(s_indices)
    current_set = set(current)
    
    for _ in range(max_iters):
        improved = False
        indices_to_check = current[:]
        rng.shuffle(indices_to_check)
        
        for p_idx in indices_to_check:
            if p_idx not in current_set:
                continue
                
            # Try removing p_idx
            new_set = [x for x in current if x != p_idx]
            
            # Check if new_set is maximal
            if _is_maximal(new_set):
                current = new_set
                current_set.remove(p_idx)
                improved = True
            else:
                pass
                
        if not improved:
            break
            
    return current

def _simulated_annealing(s_indices, start_temp=10.0, end_temp=0.1, max_iters=5000, rng=None):
    """
    Simulated annealing to minimize set size.
    Moves: remove a point, add a point, swap a point.
    Objective: minimize size, subject to being a valid no-isosceles set.
    Note: The set must remain maximal? No, usually SA works on the space of valid sets, 
    and we check maximality at the end or during the move acceptance.
    Actually, if we just remove points, we might lose maximality.
    If we add points, we increase size.
    The goal is minimum MAXIMAL set.
    So the state space should be maximal no-isosceles sets.
    Moves should transform one maximal set to another.
    A simple move: Remove a point p. If the set is still maximal, accept.
    If not maximal, we must add some points to make it maximal again?
    This is complex.
    
    Alternative: 
    State: A no-isosceles set (not necessarily maximal).
    Objective: Minimize size + Penalty if not maximal.
    But "maximal" is a hard constraint for the final answer.
    
    Let's stick to the local search that only removes points if maximality is preserved.
    This is a "descent" method.
    
    To escape local minima, we could allow adding a point and removing two, etc.
    But let's try a simpler "kick" move:
    Remove a random point (even if it breaks maximality), then greedily add points to restore maximality.
    This might lead to a different local minimum.
    """
    if rng is None:
        rng = random.Random()
        
    current = list(s_indices)
    current_set = set(current)
    current_size = len(current)
    
    best = current[:]
    best_size = current_size
    
    temp = start_temp
    cool_factor = (end_temp / start_temp) ** (1.0 / max_iters)
    
    for i in range(max_iters):
        # Choose a move
        move_type = rng.choice(['remove', 'kick'])
        
        if move_type == 'remove':
            if not current:
                continue
            p_idx = rng.choice(current)
            new_set = [x for x in current if x != p_idx]
            # Check if new_set is a valid no-isosceles set (it is, subset of valid set)
            # Check if new_set is maximal
            if _is_maximal(new_set):
                new_size = len(new_set)
                delta = new_size - current_size # -1
                if delta < 0 or rng.random() < np.exp(-delta / temp):
                    current = new_set
                    current_set = set(current)
                    current_size = new_size
                    if current_size < best_size:
                        best = current[:]
                        best_size = current_size
        else: # kick
            if not current:
                continue
            # Remove a random point
            p_idx = rng.choice(current)
            new_set = [x for x in current if x != p_idx]
            # Greedily add points to make it maximal
            # Use a random order for adding
            remaining = [x for x in range(NUM_POINTS) if x not in set(new_set)]
            rng.shuffle(remaining)
            for r in remaining:
                if not _is_conflict(r, new_set):
                    new_set.append(r)
            
            new_size = len(new_set)
            delta = new_size - current_size
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                current = new_set
                current_set = set(current)
                current_size = new_size
                if current_size < best_size:
                    best = current[:]
                    best_size = current_size
                    
        temp *= cool_factor
        
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Randomized Greedy Restarts + Local Search
    num_restarts = 0
    while time.time() < deadline - 60:
        num_restarts += 1
        
        # Random order of points
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        # Greedy construction
        current = _greedy_random_order(order)
        
        # Local search to minimize
        minimized = _local_search(current, max_iters=200, rng=rng)
        
        if len(minimized) < best_len:
            best_len = len(minimized)
            best_set = minimized
            # print(f"New best: {best_len} at restart {num_restarts}")
            
    # Phase 2: Simulated Annealing on the best found
    if best_set is not None and time.time() < deadline - 10:
        # Run SA on the best set
        # We can run multiple SA instances
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            sa_result = _simulated_annealing(best_set, start_temp=5.0, end_temp=0.01, max_iters=2000, rng=rng)
            if len(sa_result) < best_len:
                best_len = len(sa_result)
                best_set = sa_result
                
    # Phase 3: Final local search on the best found
    if best_set is not None and time.time() < deadline - 2:
        final_best = _local_search(best_set, max_iters=1000, rng=rng)
        if len(final_best) < best_len:
            best_set = final_best
            best_len = len(final_best)
            
    # Convert indices back to (x, y) tuples
    if best_set is None:
        # Fallback: just return a valid maximal set
        best_set = _greedy_random_order(list(range(NUM_POINTS)))
        
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-START
# This block is just to satisfy the prompt structure if needed, but the code above is complete.
# EVOLVE-BLOCK-END
