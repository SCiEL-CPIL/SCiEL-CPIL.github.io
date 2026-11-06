import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    max_sum = 2 * N
    
    # Precompute perfect squares
    is_square = [False] * (max_sum + 1)
    for k in range(1, int(math.isqrt(max_sum)) + 1):
        is_square[k * k] = True
    
    # Identify forbidden vertices (where 2*a is a square)
    # a ranges from 1 to 256
    # 2*a is a square => a = 2 * k^2
    forbidden = set()
    k = 1
    while 2 * k * k <= N:
        forbidden.add(2 * k * k)
        k += 1
    
    # Allowed vertices
    allowed = [i for i in range(1, N + 1) if i not in forbidden]
    m = len(allowed)
    
    # Map original values to indices 0..m-1
    # val_to_idx[value] = index
    val_to_idx = {v: i for i, v in enumerate(allowed)}
    
    # Build adjacency bitsets
    # adj_bits[i] has bit j set if allowed[i] and allowed[j] conflict (sum is square)
    adj_bits = [0] * m
    for i in range(m):
        val_i = allowed[i]
        for j in range(i + 1, m):
            val_j = allowed[j]
            if is_square[val_i + val_j]:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Helper to convert bitset to list of indices
    def bits_to_list(bits):
        res = []
        while bits:
            lsb = bits & -bits
            idx = lsb.bit_length() - 1
            res.append(idx)
            bits ^= lsb
        return res
    
    # Helper to convert list of indices to bitset
    def list_to_bits(lst):
        bits = 0
        for i in lst:
            bits |= (1 << i)
        return bits
    
    best_set_bits = 0
    best_size = 0
    best_solution = []
    
    # Greedy initialization using Minimum Degree Heuristic
    def greedy_min_degree():
        # Start with all vertices available
        available_bits = (1 << m) - 1
        current_bits = 0
        
        # We need to pick vertices one by one.
        # To be efficient, we can just iterate and pick the one with min degree in the remaining subgraph.
        # However, calculating degree dynamically is expensive.
        # Alternative: Randomized greedy or simple greedy by index.
        # Let's use a slightly smarter greedy: pick vertex with min conflicts in remaining.
        
        remaining = set(range(m))
        sol = set()
        
        # Precompute initial degrees
        deg = [len(bin(adj_bits[i]).count('1')) for i in range(m)]
        
        while remaining:
            # Find vertex with min degree in 'remaining'
            min_deg = float('inf')
            best_v = None
            # Optimization: if remaining is large, this loop is O(M^2) total. M~230, so ~50k ops. Fine.
            for v in remaining:
                # Count neighbors in remaining
                # neighbors = adj_bits[v] & list_to_bits(remaining) -- too slow to convert every time
                # Instead, just iterate neighbors and check if in remaining
                d = 0
                neighbors_bits = adj_bits[v]
                # Iterate set bits in neighbors_bits
                temp = neighbors_bits
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    if u in remaining:
                        d += 1
                    temp ^= lsb
                
                if d < min_deg:
                    min_deg = d
                    best_v = v
            
            if best_v is None:
                break
            
            sol.add(best_v)
            remaining.remove(best_v)
            # Remove neighbors of best_v from remaining
            neighbors_bits = adj_bits[best_v]
            temp = neighbors_bits
            to_remove = []
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                if u in remaining:
                    to_remove.append(u)
                temp ^= lsb
            
            for u in to_remove:
                remaining.remove(u)
                
        return list_to_bits(list(sol))
    
    # Local Search with Swaps
    def local_search(initial_bits, time_limit):
        nonlocal best_set_bits, best_size, best_solution
        
        current_bits = initial_bits
        current_size = bin(current_bits).count('1')
        
        if current_size > best_size:
            best_size = current_size
            best_set_bits = current_bits
            best_solution = bits_to_list(current_bits)
            
        improved = True
        last_improvement_time = time.time()
        
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add any vertex that doesn't conflict
            # Iterate through all vertices not in set
            not_in_set = ((1 << m) - 1) ^ current_bits
            temp = not_in_set
            added = False
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Check if v conflicts with current set
                if (adj_bits[v] & current_bits) == 0:
                    current_bits |= lsb
                    current_size += 1
                    improved = True
                    added = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set_bits = current_bits
                        best_solution = bits_to_list(current_bits)
                    break # Restart loop to check new state
                temp ^= lsb
            
            if not improved:
                # 2. Try 1-for-2 swap
                # Remove u, add v1, v2
                # v1, v2 must not conflict with (current - {u}) and must not conflict with each other.
                # Also v1, v2 must conflict with u (otherwise they could have been added in step 1).
                
                # Iterate through vertices u in current set
                temp_u = current_bits
                swap_done = False
                while temp_u and not swap_done:
                    lsb_u = temp_u & -temp_u
                    u = lsb_u.bit_length() - 1
                    temp_u ^= lsb_u
                    
                    # Find candidates v that are not in set, and (adj[v] & current) == {u}
                    # This means v conflicts only with u in the current set.
                    candidates = []
                    not_in_set = ((1 << m) - 1) ^ current_bits
                    temp_v = not_in_set
                    while temp_v:
                        lsb_v = temp_v & -temp_v
                        v = lsb_v.bit_length() - 1
                        # Check if v conflicts only with u
                        conflict_mask = adj_bits[v] & current_bits
                        if conflict_mask == lsb_u:
                            candidates.append((v, lsb_v))
                        temp_v ^= lsb_v
                    
                    if len(candidates) >= 2:
                        # Find two non-conflicting candidates
                        # Sort candidates by something? Or just check pairs.
                        # To be efficient, limit checks if list is huge.
                        found_swap = False
                        # Optimization: if candidates is large, we might just pick the first two non-conflicting
                        for i in range(len(candidates)):
                            if found_swap: break
                            v1, bit1 = candidates[i]
                            for j in range(i + 1, len(candidates)):
                                v2, bit2 = candidates[j]
                                if (adj_bits[v1] & bit2) == 0:
                                    # Perform swap
                                    current_bits ^= lsb_u
                                    current_bits |= bit1
                                    current_bits |= bit2
                                    current_size += 1
                                    improved = True
                                    swap_done = True
                                    found_swap = True
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_set_bits = current_bits
                                        best_solution = bits_to_list(current_bits)
                                    break
                        if found_swap:
                            break
                            
                if not swap_done and time.time() > last_improvement_time + 0.5:
                    # 3. Perturbation: Remove 1-3 random vertices and refill
                    # Only do this if we are stuck for a while
                    if time.time() < time_limit - 2:
                        k_remove = random.randint(1, 3)
                        to_remove = random.sample(bits_to_list(current_bits), min(k_remove, current_size))
                        for v in to_remove:
                            current_bits &= ~(1 << v)
                            current_size -= 1
                        
                        # Refill greedily
                        refill_improved = True
                        while refill_improved:
                            refill_improved = False
                            not_in_set = ((1 << m) - 1) ^ current_bits
                            temp = not_in_set
                            while temp:
                                lsb = temp & -temp
                                v = lsb.bit_length() - 1
                                if (adj_bits[v] & current_bits) == 0:
                                    current_bits |= lsb
                                    current_size += 1
                                    refill_improved = True
                                    break
                                temp ^= lsb
                                    
                        if current_size > best_size:
                            best_size = current_size
                            best_set_bits = current_bits
                            best_solution = bits_to_list(current_bits)
                            
                        last_improvement_time = time.time()
                        improved = True # Continue loop to try more moves
    
    # Main Loop: Multiple Restarts
    random.seed(42)
    
    # Initial Greedy
    initial_bits = greedy_min_degree()
    local_search(initial_bits, deadline)
    
    # Random Restarts
    while time.time() < deadline - 2:
        # Random Greedy Init
        random_order = list(range(m))
        random.shuffle(random_order)
        current_bits = 0
        for v in random_order:
            if (adj_bits[v] & current_bits) == 0:
                current_bits |= (1 << v)
        
        local_search(current_bits, deadline)
        
    # Return the best solution found
    if not best_solution:
        # Fallback
        return [1]
        
    # Convert indices back to original values
    result = [allowed[i] for i in best_solution]
    
    # Verify validity (optional but good for debugging, remove for speed in final run if slow)
    # Actually, let's just trust the bitset logic.
    
    return result

# EVOLVE-BLOCK-END
