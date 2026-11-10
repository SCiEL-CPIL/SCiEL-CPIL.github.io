# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # Safety margin of 40s before 630s kill

    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    forbidden_set = forbidden

    # Build forbidden neighbor bitmask for each element
    # mask[i] has bit j set iff (i-j) mod m is in forbidden
    mask = [0] * m
    for i in range(m):
        bits = 0
        for j in range(m):
            d = (i - j) % m
            if d in forbidden_set:
                bits |= (1 << j)
        mask[i] = bits
    
    # Also build the "available" mask: for a given set S (as bitmask),
    # the elements that can be added are those not in S and not conflicting
    # available = full_mask & ~S & ~conflict_mask(S)
    # where conflict_mask(S) = OR of mask[i] for i in S
    
    full_mask = (1 << m) - 1

    def greedy_bitmask(order):
        """Greedy using bitmask operations."""
        S = 0  # bitmask of selected elements
        for cand in order:
            if S & (1 << cand):
                continue
            # Check if cand conflicts with any selected
            if S & mask[cand]:
                continue
            S |= (1 << cand)
        return S

    def count_bits(x):
        return bin(x).count('1')

    # Phase 1: Multiple greedy initializations
    best_S = 0
    best_size = 0
    
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
    ]
    
    # Stride-based orders
    for stride in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        for offset in range(stride):
            orders.append(list(range(offset, m, stride)))
    
    # Random orders
    rng_init = random.Random(42)
    for seed in range(50):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
    
    for order in orders:
        S = greedy_bitmask(order)
        sz = count_bits(S)
        if sz > best_size:
            best_size = sz
            best_S = S
    
    # Phase 2: Local search with bitmask
    # Try to add elements greedily
    def local_search_improve(S):
        """Try to add more elements to S using bitmask."""
        improved = True
        while improved:
            improved = False
            # Find all candidates that can be added
            # A candidate c can be added if:
            #   (1 << c) & S == 0  (not already in S)
            #   mask[c] & S == 0   (no conflict)
            for c in range(m):
                bit = 1 << c
                if S & bit:
                    continue
                if S & mask[c]:
                    continue
                S |= bit
                improved = True
                break
        return S

    best_S = local_search_improve(best_S)
    best_size = count_bits(best_S)

    # Phase 3: Simulated annealing / random restarts with bitmask
    rng_sa = random.Random(999)
    temp = 1.0
    cooling = 0.9995
    
    # Precompute list of bits for fast iteration
    bits_list = [1 << i for i in range(m)]
    
    for iteration in range(200000):
        if time.time() > deadline - 30:
            break
        
        move_type = rng_sa.random()
        
        if move_type < 0.4:
            # Try to add a random element
            candidates = []
            for c in range(m):
                bit = bits_list[c]
                if best_S & bit:
                    continue
                if best_S & mask[c]:
                    continue
                candidates.append(c)
            if candidates:
                cand = rng_sa.choice(candidates)
                new_S = best_S | bits_list[cand]
                if count_bits(new_S) > best_size:
                    best_S = new_S
                    best_size = count_bits(best_S)
        
        elif move_type < 0.8:
            # Swap: remove one, add one
            if best_size == 0:
                continue
            # Pick random element in S to remove
            in_elements = [i for i in range(m) if best_S & bits_list[i]]
            if not in_elements:
                continue
            to_remove = rng_sa.choice(in_elements)
            new_S = best_S & ~bits_list[to_remove]
            
            # Try to add a random element not in new_S
            candidates = []
            for c in range(m):
                bit = bits_list[c]
                if new_S & bit:
                    continue
                if new_S & mask[c]:
                    continue
                candidates.append(c)
            if candidates:
                to_add = rng_sa.choice(candidates)
                new_S |= bits_list[to_add]
                sz = count_bits(new_S)
                if sz >= best_size:
                    best_S = new_S
                    best_size = sz
        
        else:
            # Remove one element (to escape local optima)
            if best_size > 0:
                in_elements = [i for i in range(m) if best_S & bits_list[i]]
                if in_elements:
                    to_remove = rng_sa.choice(in_elements)
                    best_S &= ~bits_list[to_remove]
                    best_size -= 1
        
        temp *= cooling
    
    # Final greedy fill
    best_S = local_search_improve(best_S)
    best_size = count_bits(best_S)

    # Phase 4: Branch and bound for maximum independent set
    # Use a recursive approach with bitset pruning
    # Graph: vertices 0..204, edge between i,j iff (i-j)%m in forbidden
    # We want max independent set.
    
    # Precompute adjacency as bitmask: adj[i] = mask[i] (same as before)
    adj = mask  # adj[i] has bit j set iff i and j are adjacent (conflict)
    
    # For B&B, we use the standard algorithm:
    # max_independent_set(vertices, current_size):
    #   if vertices is empty: return current_size
    #   upper_bound = current_size + count_bits(vertices)
    #   if upper_bound <= best_found: return best_found
    #   pick a vertex v from vertices
    #   # Branch 1: include v
    #   #   new_vertices = vertices & ~(adj[v] | (1<<v))
    #   #   result1 = max_independent_set(new_vertices, current_size + 1)
    #   # Branch 2: exclude v
    #   #   new_vertices = vertices & ~(1<<v)
    #   #   result2 = max_independent_set(new_vertices, current_size)
    #   return max(result1, result2)
    
    # This can be expensive, so we use it with a time limit and good initial bound.
    
    global_best_S = best_S
    global_best_size = best_size
    bbb_time_limit = deadline - 5  # Leave 5s for final processing
    bbb_start = time.time()
    
    # Use a stack-based B&B to avoid recursion depth issues
    # State: (vertices_bitmask, current_size)
    # We use a priority queue-like approach: process states with higher upper bound first
    
    import heapq
    
    # Max-heap by upper bound (negate for min-heap)
    # We'll use a list and always pick the state with highest potential
    
    # Actually, let's use a simpler iterative deepening approach
    # Start with the current best and try to improve via targeted search
    
    # Alternative: use a different B&B formulation
    # Color-based upper bound for independent set (like Bron-Kerbosch with coloring)
    
    # For now, let's do a targeted local search with more aggressive moves
    # Try all pairs of removals and see if we can add 2+ elements
    
    for iter2 in range(50000):
        if time.time() > bbb_time_limit:
            break
        
        # Try: remove 1 element, then greedily fill
        if best_size > 0:
            in_elements = [i for i in range(m) if best_S & bits_list[i]]
            if in_elements:
                to_remove = rng_sa.choice(in_elements)
                new_S = best_S & ~bits_list[to_remove]
                new_S = local_search_improve(new_S)
                sz = count_bits(new_S)
                if sz > best_size:
                    best_S = new_S
                    best_size = sz
        
        # Try: remove 2 elements, then greedily fill
        if best_size > 1 and rng_sa.random() < 0.3:
            in_elements = [i for i in range(m) if best_S & bits_list[i]]
            if len(in_elements) >= 2:
                i1, i2 = rng_sa.sample(in_elements, 2)
                new_S = best_S & ~(bits_list[i1] | bits_list[i2])
                new_S = local_search_improve(new_S)
                sz = count_bits(new_S)
                if sz > best_size:
                    best_S = new_S
                    best_size = sz
    
    # Final greedy fill
    best_S = local_search_improve(best_S)
    best_size = count_bits(best_S)

    # Convert bitmask to list
    final_list = []
    for i in range(m):
        if best_S & (1 << i):
            final_list.append(i)
    
    # Verify the final set
    for i in range(len(final_list)):
        for j in range(i+1, len(final_list)):
            a, b = final_list[i], final_list[j]
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden_set or d2 in forbidden_set:
                # Should not happen, but if it does, rebuild
                final_list = [0]
                break
    
    return np.array(final_list, dtype=np.int64)

# EVOLVE-BLOCK-END
