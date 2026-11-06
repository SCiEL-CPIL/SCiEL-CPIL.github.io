import time
import random
import numpy as np

N = 512
DEADLINE = 55.0

def construct():
    start_time = time.time()
    
    # 1. Precompute 4-AP conflict graph
    # ap_sets[x] = list of indices (0-based) that form a 4-AP with x
    # Since the graph is undirected, if y is in ap_sets[x], x is in ap_sets[y]
    ap_sets = [[] for _ in range(N)]
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 0-based indices
            i_a, i_b, i_c, i_e = a - 1, b - 1, c - 1, e - 1
            ap_sets[i_a].extend([i_b, i_c, i_e])
            ap_sets[i_b].extend([i_a, i_c, i_e])
            ap_sets[i_c].extend([i_a, i_b, i_e])
            ap_sets[i_e].extend([i_a, i_b, i_c])
    
    # Convert to numpy arrays for fast operations
    ap_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap_sets]
    
    # 2. Greedy Initialization
    # We maintain a bitset (numpy boolean array) for the current set
    # and a conflict_count array for fast validity checks
    # bitset[i] is True if i+1 is in the set
    # conflict_count[i] is the number of elements in the set that conflict with i
    
    bits = np.zeros(N, dtype=bool)
    conflict_count = np.zeros(N, dtype=np.int32)
    
    # Greedy: add elements in increasing order if valid
    for x in range(N):
        if conflict_count[x] == 0:
            bits[x] = True
            # Update conflict counts for neighbors
            for y in ap_arrays[x]:
                if not bits[y]:
                    conflict_count[y] += 1
    
    best_bits = bits.copy()
    best_size = bits.sum()
    
    # 3. Simulated Annealing
    # We use a fast O(1) validity check: conflict_count[x] == 0
    # Swaps: remove y, add x.
    # New conflict_count for z:
    #   if z == x: 0 (since it was valid to add, but we must check if removing y affects it? No, y is removed, so it only helps x)
    #   Actually, let's just compute the delta.
    #   Removing y: conflict_count[z] -= 1 if z in ap_arrays[y]
    #   Adding x: conflict_count[z] += 1 if z in ap_arrays[x]
    
    # To speed up swaps, we can precompute or just do it on the fly.
    # On the fly is O(deg(x) + deg(y)), which is small.
    
    current_temp = 1.0
    cooling_rate = 0.9999
    min_temp = 0.01
    
    current_size = bits.sum()
    
    while time.time() - start_time < DEADLINE:
        if current_temp < min_temp:
            break
            
        # Pick a random element y to remove
        if current_size == 0:
            break
        set_indices = np.nonzero(bits)[0]
        y = set_indices[random.randint(0, len(set_indices) - 1)]
        
        # Pick a random element x to add (must not be in set)
        # We can just pick a random index and check
        x = random.randint(0, N - 1)
        if bits[x]:
            continue
            
        # Check if swap is valid
        # If we remove y and add x, is the set 4-AP free?
        # This is valid if:
        # 1. x has no conflicts with the set minus {y}
        #    conflict_count[x] - (1 if y in ap_arrays[x] else 0) == 0
        # 2. y has no conflicts with the set minus {y} ... wait, y is removed, so this is trivially true for y itself.
        #    But we need to ensure that removing y doesn't break anything? No, removing an element never creates a 4-AP.
        #    So the only condition is that x is valid in the set with y removed.
        
        is_y_neighbor = (y in ap_arrays[x]) # This is O(deg(x))
        # Faster: use a set or a precomputed matrix? N=512, so a 512x512 boolean matrix is 256KB, very fast.
        
        # Let's use a matrix for O(1) neighbor check.
        # Actually, we can just check `conflict_count[x]`.
        # If y is a neighbor of x, then `conflict_count[x]` includes y.
        # So we need `conflict_count[x] - (1 if is_y_neighbor else 0) == 0`.
        
        # To make is_y_neighbor O(1), let's use a numpy matrix or just a list of sets.
        # Let's stick to the list of arrays and use a simple check. 
        # Wait, `y in ap_arrays[x]` is O(deg(x)). deg(x) is at most ~170. That's fine.
        
        # Optimization: Precompute a 512x512 boolean matrix `is_neighbor`
        # This takes 512*512 = 262144 bools = 256KB. Very fast to build.
        
        # Let's rebuild the neighbor check using a matrix for speed in the inner loop.
        
        if 'is_neighbor' not in locals():
            is_neighbor = np.zeros((N, N), dtype=bool)
            for i in range(N):
                if len(ap_arrays[i]) > 0:
                    is_neighbor[i, ap_arrays[i]] = True
        
        is_neigh = is_neighbor[x, y]
        
        # Valid swap condition:
        # x must have 0 conflicts in (Set - {y})
        # conflict_count[x] counts conflicts in Set.
        # If y is a neighbor of x, y is one of those conflicts.
        # So conflicts in (Set - {y}) = conflict_count[x] - (1 if is_neigh else 0)
        
        if conflict_count[x] - (1 if is_neigh else 0) == 0:
            # Swap is valid
            # Perform swap
            bits[y] = False
            bits[x] = True
            current_size += 0 # Size is constant
            
            # Update conflict counts
            # For all z in ap_arrays[y]: conflict_count[z] -= 1 (if z is in set? No, conflict_count is for ALL elements, but we only care about those NOT in set for adding, and those IN set for validity? 
            # Actually, conflict_count[z] = number of elements in Set that conflict with z.
            # So if y is removed, for all z in ap_arrays[y], conflict_count[z] decreases by 1.
            # If x is added, for all z in ap_arrays[x], conflict_count[z] increases by 1.
            
            # We must be careful: if z is y, conflict_count[y] changes. But y is not in set, so it doesn't matter for validity of adding other elements, but it matters if we want to add y back later.
            # So we update for ALL z in ap_arrays[y] and ap_arrays[x].
            
            for z in ap_arrays[y]:
                conflict_count[z] -= 1
            for z in ap_arrays[x]:
                conflict_count[z] += 1
            
            if current_size > best_size:
                best_bits = bits.copy()
                best_size = current_size
        else:
            # Invalid swap. 
            # In SA, we might accept invalid states? No, we are constrained to 4-AP free sets.
            # So we just reject it.
            # To allow SA to escape local minima, we need to consider moves that temporarily reduce size?
            # Or use a "kick" move: remove 2, add 1?
            # Or just rely on the randomness of x and y to find valid swaps.
            # If no valid swaps are found, we are stuck.
            # Let's add a "kick" phase if we are stuck for a while.
            
            # For now, let's just try another swap.
            pass
        
        # Cooling
        current_temp *= cooling_rate
        
        # Periodic restart if stuck?
        # If best_size hasn't changed in a long time, maybe try a random restart?
        # Or a "remove 1" move to allow adding 2?
        
    # Convert best_bits to list of 1-based integers
    result = np.nonzero(best_bits)[0] + 1
    return list(result)

# Pre-compute is_neighbor matrix to avoid re-computing in the loop if we re-enter construct()
# But construct() is called once.
# We'll do it inside construct() to keep it self-contained.

# Wait, I put `if 'is_neighbor' not in locals()` inside the loop. That's inefficient.
# Let's move it out.

def construct():
    start_time = time.time()
    
    # 1. Precompute 4-AP conflict graph
    ap_sets = [[] for _ in range(N)]
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            i_a, i_b, i_c, i_e = a - 1, b - 1, c - 1, e - 1
            ap_sets[i_a].extend([i_b, i_c, i_e])
            ap_sets[i_b].extend([i_a, i_c, i_e])
            ap_sets[i_c].extend([i_a, i_b, i_e])
            ap_sets[i_e].extend([i_a, i_b, i_c])
            
    ap_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap_sets]
    
    is_neighbor = np.zeros((N, N), dtype=bool)
    for i in range(N):
        if len(ap_arrays[i]) > 0:
            is_neighbor[i, ap_arrays[i]] = True
            
    # 2. Greedy Initialization
    bits = np.zeros(N, dtype=bool)
    conflict_count = np.zeros(N, dtype=np.int32)
    
    for x in range(N):
        if conflict_count[x] == 0:
            bits[x] = True
            for y in ap_arrays[x]:
                if not bits[y]:
                    conflict_count[y] += 1
    
    best_bits = bits.copy()
    best_size = bits.sum()
    
    # 3. Simulated Annealing
    current_temp = 1.0
    cooling_rate = 0.9999
    min_temp = 0.01
    
    current_size = bits.sum()
    stuck_counter = 0
    
    while time.time() - start_time < DEADLINE:
        if current_temp < min_temp:
            break
            
        if current_size == 0:
            break
            
        set_indices = np.nonzero(bits)[0]
        y = set_indices[random.randint(0, len(set_indices) - 1)]
        x = random.randint(0, N - 1)
        
        if bits[x]:
            continue
            
        is_neigh = is_neighbor[x, y]
        
        if conflict_count[x] - (1 if is_neigh else 0) == 0:
            # Valid swap
            bits[y] = False
            bits[x] = True
            
            for z in ap_arrays[y]:
                conflict_count[z] -= 1
            for z in ap_arrays[x]:
                conflict_count[z] += 1
                
            if current_size > best_size:
                best_bits = bits.copy()
                best_size = current_size
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            stuck_counter += 1
            
        # If stuck for too long, try a "remove 1" move to potentially allow adding 2 later
        # Or just reset temperature?
        if stuck_counter > 1000:
            # Remove a random element to break the deadlock
            if current_size > 1:
                y = set_indices[random.randint(0, len(set_indices) - 1)]
                bits[y] = False
                current_size -= 1
                for z in ap_arrays[y]:
                    conflict_count[z] -= 1
            stuck_counter = 0
            current_temp = max(current_temp, 0.5) # Reheat
            
        current_temp *= cooling_rate
        
    result = np.nonzero(best_bits)[0] + 1
    return list(result)
