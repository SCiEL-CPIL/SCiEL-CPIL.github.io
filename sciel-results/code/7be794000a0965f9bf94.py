import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: 55s out of 60s limit. 
    # If we exceed this, we return the best found so far.
    deadline = start_time + 55.0
    
    n = 256
    
    # Perfect squares up to 512 (max sum 256+256)
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
        
    # Diagonal constraint: 2a is a perfect square
    # a = 2 * m^2
    diagonal_bad = set()
    m = 1
    while 2 * m * m <= n:
        diagonal_bad.add(2 * m * m)
        m += 1
        
    # Available vertices: those not violating diagonal constraint
    available_vertices = [i for i in range(1, n + 1) if i not in diagonal_bad]
    available_set = set(available_vertices)
    
    # Build conflict graph
    # conflicts[v] = set of u such that u+v is square and u != v
    # We only care about conflicts between available vertices
    conflicts = {v: set() for v in available_vertices}
    
    for a in available_vertices:
        for b in available_vertices:
            if a < b and (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Precompute sorted available vertices for deterministic iteration if needed
    avail_list = sorted(available_vertices)

    def greedy_init_weighted(pool, seed=None):
        """
        Weighted Greedy Init:
        At each step, calculate degree in the remaining pool.
        Select a vertex with probability proportional to exp(-degree * temperature).
        Lower degree -> higher probability.
        This allows for diverse initial solutions compared to pure min-degree.
        """
        if seed is not None:
            random.seed(seed)
        
        current_pool = set(pool)
        selected = set()
        
        temperature = 1.0 # Tuning parameter
        
        while current_pool:
            # Calculate degrees in current_pool
            degrees = {}
            min_deg = float('inf')
            for v in current_pool:
                # Degree is number of neighbors in current_pool
                deg = len(conflicts[v] & current_pool)
                degrees[v] = deg
                if deg < min_deg:
                    min_deg = deg
            
            # If we have isolated nodes (deg 0), we can pick any of them randomly
            # or just pick one. Picking randomly adds diversity.
            if min_deg == 0:
                isolated = [v for v, d in degrees.items() if d == 0]
                v = random.choice(isolated)
            else:
                # Calculate weights
                # weight = exp(-(deg - min_deg) * temperature)
                # This ensures min_deg nodes have weight 1.0, higher degrees decay.
                weights = {}
                total_w = 0.0
                for v, deg in degrees.items():
                    w = math.exp(-(deg - min_deg) * temperature)
                    weights[v] = w
                    total_w += w
                
                # Random selection based on weights
                r = random.random() * total_w
                cumulative = 0.0
                v = None
                for node in current_pool:
                    cumulative += weights[node]
                    if r <= cumulative:
                        v = node
                        break
                if v is None: # Fallback due to float precision
                    v = random.choice(list(current_pool))
            
            selected.add(v)
            # Remove v and its neighbors from pool
            to_remove = {v} | (conflicts[v] & current_pool)
            current_pool -= to_remove
            
        return selected

    def local_search(s, time_limit):
        """
        Local Search with:
        1. Add any compatible vertex.
        2. 1-for-2 Swap: Remove u, add v1, v2 if v1, v2 are only blocked by u and not each other.
        3. Perturbation: Randomly remove k vertices, then greedy refill.
        """
        current = set(s)
        in_set = set(current)
        
        improved = True
        # Track best found in this local search to avoid getting stuck in bad perturbations
        best_in_ls = set(current)
        best_size = len(current)
        
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try adding any vertex that is not in set and has no conflicts with set
            # Optimization: Iterate through vertices not in set
            # We can keep a list of "candidates" but set intersection is fast enough for N=256
            for v in avail_list:
                if v in in_set:
                    continue
                # Check if v has any neighbor in current
                if not (conflicts[v] & in_set):
                    current.add(v)
                    in_set.add(v)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_in_ls = set(current)
                    break # Restart loop to check new state
            
            if not improved:
                # 2. Try 1-for-2 swap
                # For each u in current, find candidates v not in current such that 
                # conflicts[v] & current == {u}
                # If we find two such v1, v2 that are not adjacent, swap u -> {v1, v2}
                
                swap_done = False
                # Iterate over a copy of current as we might modify it
                for u in list(current):
                    candidates = []
                    for v in avail_list:
                        if v in in_set:
                            continue
                        # v is blocked by current if (conflicts[v] & in_set) is non-empty
                        # We want (conflicts[v] & in_set) to be exactly {u}
                        
                        # Fast check:
                        # 1. u must be in conflicts[v]
                        if u not in conflicts[v]:
                            continue
                        # 2. No other neighbor of v should be in in_set
                        # i.e., (conflicts[v] & in_set) - {u} is empty
                        # This is equivalent to len(conflicts[v] & in_set) == 1
                        
                        # Optimization: Check if any other neighbor is in set
                        # We can iterate neighbors of v and check if they are in in_set (excluding u)
                        # But set intersection is implemented in C, so it's fast.
                        common = conflicts[v] & in_set
                        if len(common) == 1: # Must be {u} since u is in common
                            candidates.append(v)
                    
                    if len(candidates) >= 2:
                        # Find two non-adjacent candidates
                        found_swap = False
                        # Optimization: We don't need to check all pairs if we find one quickly
                        # But candidates list is usually small.
                        for i in range(len(candidates)):
                            if found_swap: break
                            for j in range(i + 1, len(candidates)):
                                if candidates[i] not in conflicts[candidates[j]]:
                                    # Swap!
                                    current.remove(u)
                                    in_set.remove(u)
                                    current.add(candidates[i])
                                    in_set.add(candidates[i])
                                    current.add(candidates[j])
                                    in_set.add(candidates[j])
                                    improved = True
                                    swap_done = True
                                    if len(current) > best_size:
                                        best_size = len(current)
                                        best_in_ls = set(current)
                                    found_swap = True
                                    break
                        if swap_done:
                            break # Restart outer loop
                
                if not swap_done:
                    # 3. Perturbation: Remove 1-3 random vertices and refill
                    # This helps escape local minima
                    if time.time() > time_limit - 2:
                        break
                        
                    # Random removal
                    # Remove vertices that are "critical" or just random?
                    # Random is good for diversity.
                    k_remove = random.randint(1, 3)
                    to_remove = random.sample(list(current), min(k_remove, len(current)))
                    for v in to_remove:
                        current.remove(v)
                        in_set.remove(v)
                    
                    # Refill greedily
                    refill_improved = True
                    while refill_improved:
                        refill_improved = False
                        for v in avail_list:
                            if v in in_set:
                                continue
                            if not (conflicts[v] & in_set):
                                current.add(v)
                                in_set.add(v)
                                refill_improved = True
                                break
                    
                    # We accept the perturbation state if it leads to improvement later,
                    # but for now, we just continue the loop.
                    # To avoid infinite loops in bad states, we rely on the time limit.
                    # We don't mark 'improved' here because size might have decreased.
                    # However, if we don't mark improved, we might loop on the same state if 
                    # refill doesn't change anything and swap fails.
                    # So, if size increased, mark improved. If size same, maybe mark improved to try swaps again?
                    # Actually, if size decreased, we are in a new state.
                    # Let's mark improved if size >= previous best in this LS iteration?
                    # No, standard approach: if we changed the set, we continue.
                    # But if we just removed and refilled to the same set, we loop.
                    # Check if set changed
                    # For simplicity, we assume perturbation changes state.
                    pass

        # Return the best set found during this local search
        if len(best_in_ls) > len(current):
            return best_in_ls
        return current

    best_set = set()
    best_size = 0
    
    # Initial greedy (Deterministic Min-Degree for a strong baseline)
    # We'll use a simple min-degree greedy for the first one
    def greedy_min_degree(pool):
        current_pool = set(pool)
        selected = set()
        while current_pool:
            min_deg = float('inf')
            best_v = None
            for v in current_pool:
                deg = len(conflicts[v] & current_pool)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if min_deg == 0:
                        break
            if best_v is None:
                break
            selected.add(best_v)
            to_remove = {best_v} | (conflicts[best_v] & current_pool)
            current_pool -= to_remove
        return selected

    s0 = greedy_min_degree(available_vertices)
    best_set = set(s0)
    best_size = len(s0)
    
    # Local search on initial
    s_ls = local_search(s0, deadline - 10) # Leave 10s for restarts
    if len(s_ls) > best_size:
        best_size = len(s_ls)
        best_set = set(s_ls)
        
    # Multiple restarts with Weighted Greedy
    random.seed(42)
    
    restart_count = 0
    while time.time() < deadline - 2:
        restart_count += 1
        # Weighted Greedy Init with random seed
        s_init = greedy_init_weighted(available_vertices, seed=restart_count)
        
        # Local search
        s_res = local_search(s_init, deadline - 2)
        
        if len(s_res) > best_size:
            best_size = len(s_res)
            best_set = set(s_res)
            
    return list(best_set)
