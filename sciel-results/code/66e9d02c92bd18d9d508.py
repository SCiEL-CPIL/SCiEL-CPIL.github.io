import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute conflict graph using bitsets for speed
    # conflict_mask[v] is a bitset where bit u is set if v conflicts with u
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = 0
            a, b = i, j
            while b:
                a, b = b, a % b
            g = a
            u_prime = i // g
            v_prime = j // g
            if g % (u_prime + v_prime) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Bitset for all vertices
    all_mask = 0
    for v in vertices:
        all_mask |= (1 << v)
    
    def can_add(v, s_mask):
        """Check if vertex v can be added to set represented by s_mask."""
        # v can be added if it doesn't conflict with any vertex in s_mask
        return (conflict_mask[v] & s_mask) == 0
    
    def greedy_by_order(order, s_mask=0):
        result_mask = s_mask
        for v in order:
            if (result_mask >> v) & 1:
                continue
            if can_add(v, result_mask):
                result_mask |= (1 << v)
        return result_mask
    
    def mask_to_set(mask):
        return {v for v in vertices if (mask >> v) & 1}
    
    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << v)
        return m
    
    # Greedy construction
    best_mask = 0
    
    # Try multiple orderings
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N + 1)]
    order_asc = sorted(vertices, key=lambda x: degrees[x])
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    order_odd = [v for v in vertices if v % 2 == 1]
    order_even = [v for v in vertices if v % 2 == 0]
    order_odd_first = sorted(vertices, key=lambda x: (x % 2 == 0, degrees[x]))
    order_value = list(range(1, N + 1))
    order_value_desc = list(range(N, 0, -1))
    
    candidates = [
        greedy_by_order(order_asc),
        greedy_by_order(order_desc),
        greedy_by_order(order_odd),
        greedy_by_order(order_even),
        greedy_by_order(order_odd_first),
        greedy_by_order(order_value),
        greedy_by_order(order_value_desc),
    ]
    
    # Multiple random shuffles
    for _ in range(50):
        r = vertices[:]
        random.shuffle(r)
        candidates.append(greedy_by_order(r))
    
    for c in candidates:
        if bin(c).count('1') > bin(best_mask).count('1'):
            best_mask = c
    
    # Local search: add vertices, then try 1-for-2 swaps
    def local_search(s_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Try adding each excluded vertex
            excluded_mask = all_mask & ~s_mask
            excluded = [v for v in vertices if (excluded_mask >> v) & 1]
            random.shuffle(excluded)
            for v in excluded:
                if time.time() >= time_limit:
                    break
                if can_add(v, s_mask):
                    s_mask |= (1 << v)
                    improved = True
            
            # Phase 2: Try 1-for-2 swaps (remove one, add two non-conflicting)
            if time.time() < time_limit:
                s_list = [v for v in vertices if (s_mask >> v) & 1]
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() >= time_limit:
                        break
                    temp_mask = s_mask & ~(1 << u)
                    # Find all vertices that can be added to temp
                    addable_mask = all_mask & ~s_mask
                    addable = []
                    for v in vertices:
                        if (addable_mask >> v) & 1 and v != u and can_add(v, temp_mask):
                            addable.append(v)
                    if len(addable) >= 2:
                        random.shuffle(addable)
                        found = False
                        limit = min(len(addable), 20)
                        for i in range(limit):
                            for j in range(i + 1, limit):
                                v1, v2 = addable[i], addable[j]
                                if (conflict_mask[v1] >> v2) & 1 == 0:
                                    s_mask = temp_mask | (1 << v1) | (1 << v2)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
        return s_mask
    
    # Run local search from best candidate
    time_limit = deadline - 10.0
    best_mask = local_search(best_mask, time_limit)
    
    # Multiple restarts with random perturbations
    while time.time() < deadline - 10.0:
        # Create perturbation: remove 1-3 random vertices
        perturbed_mask = best_mask
        s_list = [v for v in vertices if (perturbed_mask >> v) & 1]
        if s_list:
            n_remove = min(random.randint(1, 3), len(s_list))
            to_remove = random.sample(s_list, n_remove)
            for v in to_remove:
                perturbed_mask &= ~(1 << v)
        
        result_mask = local_search(perturbed_mask, deadline - 10.0)
        if bin(result_mask).count('1') > bin(best_mask).count('1'):
            best_mask = result_mask
    
    # Final validation
    result_list = sorted(mask_to_set(best_mask))
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            if (a * b) % (a + b) == 0:
                return list(range(1, 257, 2))
    
    return result_list
# EVOLVE-BLOCK-END
