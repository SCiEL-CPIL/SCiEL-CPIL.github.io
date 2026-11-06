import random
import time
import math

N = 64
GRID_SIZE = N * N
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

# Pre-compute squared distances between all pairs of grid points
# Using a flat list of arrays or list of lists. 
# For speed in Python, list of lists of ints is okay, but numpy might be faster for vectorized checks?
# However, we need individual checks for validity.
# Let's stick to precomputed lists but optimize the validity check.

DIST = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

def precompute_distances():
    for i in range(GRID_SIZE):
        x1, y1 = ALL_POINTS[i]
        for j in range(i + 1, GRID_SIZE):
            x2, y2 = ALL_POINTS[j]
            dx = x1 - x2
            dy = y1 - y2
            d_sq = dx * dx + dy * dy
            DIST[i][j] = d_sq
            DIST[j][i] = d_sq

precompute_distances()

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Helper to check if a set of indices is valid
    # S is a list of indices
    # Condition: For all a in S, all d(a, b) for b in S, b!=a are distinct.
    def is_valid_set(S):
        if len(S) < 3:
            return True
        for a in S:
            seen_dists = set()
            for b in S:
                if a == b:
                    continue
                d = DIST[a][b]
                if d in seen_dists:
                    return False
                seen_dists.add(d)
        return True

    # Helper to check if adding k to S is valid
    # We only need to check:
    # 1. d(k, a) is not equal to d(k, b) for any a, b in S (k's distances to S are unique)
    # 2. For any a in S, d(k, a) is not equal to d(a, b) for any b in S, b!=a 
    #    Wait, condition 2 is: For a fixed a in S, the new distance d(k,a) must not equal any existing distance d(a,b) where b in S, b!=a.
    #    Also, for a fixed b in S, the new distance d(k,b) must not equal any existing distance d(b,a) where a in S, a!=b.
    #    Actually, if we ensure that for EVERY a in S, the set of distances from a to S remains unique after adding k, it's valid.
    #    This is O(|S|^2) if done naively.
    #    Optimization:
    #    For each a in S:
    #       Existing distances from a to S\{a} are stored or can be checked.
    #       New distance is d_ka.
    #       We need d_ka not in {d(a,b) for b in S, b!=a}.
    #    This is O(|S|^2) total for the check.
    
    def can_add(k, S):
        m = len(S)
        if m == 0:
            return True
        
        # Check 1: Distances from k to S must be unique
        seen_dists_from_k = set()
        for a in S:
            d_ka = DIST[k][a]
            if d_ka in seen_dists_from_k:
                return False
            seen_dists_from_k.add(d_ka)
            
        # Check 2: For each a in S, d_ka must not be in {d(a,b) for b in S, b!=a}
        # And for each b in S, d_kb must not be in {d(b,a) for a in S, a!=b}
        # Note: d_ka is the distance from k to a.
        # For vertex a, the new edge is (a,k) with weight d_ka.
        # The existing edges from a are (a,b) with weights d_ab.
        # So we check if d_ka == d_ab for any b in S, b!=a.
        
        for a in S:
            d_ka = DIST[k][a]
            for b in S:
                if a == b:
                    continue
                if DIST[a][b] == d_ka:
                    return False
        return True

    # Initial greedy construction
    def greedy_construct():
        S = []
        candidates = list(range(GRID_SIZE))
        random.shuffle(candidates)
        for k in candidates:
            if can_add(k, S):
                S.append(k)
        return S

    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Simulated Annealing
    # State: S (list of indices)
    # Energy: -len(S)
    # Moves:
    # 1. Add a random point not in S
    # 2. Remove a random point from S
    # 3. Swap a point in S with a point not in S
    
    current_set = best_set[:]
    current_energy = -len(current_set)
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-4
    
    in_set = [False] * GRID_SIZE
    for i in current_set:
        in_set[i] = True
        
    # Precompute neighbors? No, just use in_set.
    
    while time.time() < deadline and T > min_T:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add
            if len(current_set) < GRID_SIZE:
                # Pick a random point not in set
                # Efficient way: keep a list of available points? 
                # For now, random sampling with retry is okay if set is small, 
                # but if set is large, we need a list of available.
                # Let's maintain an available list for speed.
                pass # We'll implement a simpler swap/remove/add strategy below
                
        # Let's use a simpler strategy: 
        # 1. Pick random index i from 0..GRID_SIZE-1
        # 2. If i in S, try to remove it.
        # 3. If i not in S, try to add it.
        # 4. Occasionally, do a swap: pick i in S, j not in S, try to swap.
        
        # To make "add" and "remove" efficient, we maintain `current_set` and `in_set`.
        
        if len(current_set) == 0:
             # Restart
             current_set = greedy_construct()
             for i in current_set: in_set[i] = True
             for i in range(GRID_SIZE):
                 if not in_set[i]:
                     # This loop is O(N^2). Better to rebuild in_set.
                     pass
             in_set = [False] * GRID_SIZE
             for i in current_set: in_set[i] = True
             continue
             
        r = random.random()
        
        if r < 0.4:
            # Add
            # Pick random candidate
            attempts = 0
            while attempts < 10:
                k = random.randint(0, GRID_SIZE - 1)
                if not in_set[k]:
                    break
                attempts += 1
            else:
                continue
                
            if can_add(k, current_set):
                new_set = current_set + [k]
                new_energy = -len(new_set)
                delta = new_energy - current_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_set = new_set
                    in_set[k] = True
                    current_energy = new_energy
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                        
        elif r < 0.7:
            # Remove
            if current_set:
                idx = random.randint(0, len(current_set) - 1)
                k = current_set[idx]
                new_set = current_set[:idx] + current_set[idx+1:]
                new_energy = -len(new_set)
                delta = new_energy - current_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_set = new_set
                    in_set[k] = False
                    current_energy = new_energy
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                        
        else:
            # Swap
            if current_set:
                # Pick one from set
                idx_in = random.randint(0, len(current_set) - 1)
                k_in = current_set[idx_in]
                
                # Pick one not in set
                attempts = 0
                k_out = -1
                while attempts < 10:
                    k_out = random.randint(0, GRID_SIZE - 1)
                    if not in_set[k_out]:
                        break
                    attempts += 1
                else:
                    continue
                
                # Create new set: remove k_in, add k_out
                # Check validity of new set
                # We can check can_add(k_out, current_set without k_in)
                # And check that removing k_in doesn't break anything? 
                # Removing a point always preserves the "no isosceles" property for the remaining set?
                # Yes, if a set is valid, any subset is valid.
                # So we only need to check if k_out can be added to (current_set \ {k_in}).
                
                # Efficient check:
                # Let S' = current_set \ {k_in}
                # Check if k_out can be added to S'.
                
                # We can reuse can_add but it expects a list.
                # S' is a list.
                S_prime = current_set[:idx_in] + current_set[idx_in+1:]
                
                if can_add(k_out, S_prime):
                    new_set = S_prime + [k_out]
                    new_energy = -len(new_set)
                    delta = new_energy - current_energy
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_set = new_set
                        in_set[k_in] = False
                        in_set[k_out] = True
                        current_energy = new_energy
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
                            
        T *= cooling
        
    return [ALL_POINTS[i] for i in best_set]
