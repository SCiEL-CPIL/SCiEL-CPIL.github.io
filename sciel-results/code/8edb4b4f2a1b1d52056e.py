import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Precompute forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency as bitmasks for fast conflict checking
    # adj[u] is a bitmask where bit v is 1 if u and v conflict
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask
        
    # Helper: Check if a vertex can be added to a set (bitmask)
    def can_add(v, set_mask):
        return (set_mask & adj[v]) == 0

    best_size = 0
    best_set_mask = 0
    best_set_list = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    # Strategy: 
    # 1. Greedy Construction with Random Restarts
    # 2. Local Search: 
    #    a. Simple Hill Climbing (Add if possible)
    #    b. Swap-based Local Search (Remove 1, Add 1 if it allows adding more later? No, just 1-for-1 swap to maintain size, then try to add? 
    #       Actually, standard MIS local search is: 
    #       - Try to add any vertex.
    #       - If none, try to swap: remove v, add u. If the new set has a vertex w that could be added (which wasn't possible before), 
    #         then the swap is beneficial. 
    #       - A simpler, effective heuristic: "2-opt" style. Remove 2 vertices, add 3 vertices.
    #       - Or just: Randomly remove k vertices, then greedily fill.
    
    def greedy_fill(base_mask, available_mask=None):
        """
        Greedily add vertices to base_mask from available_mask (or all remaining) 
        in a randomized order.
        """
        if available_mask is None:
            available_mask = ((1 << m) - 1) & ~base_mask
        
        current = base_mask
        # Get list of candidates
        candidates = [i for i in range(m) if (available_mask >> i) & 1]
        random.shuffle(candidates)
        
        for v in candidates:
            if can_add(v, current):
                current |= (1 << v)
                
        return current

    def local_search_improve(current_mask):
        """
        Try to improve the current independent set using local moves.
        Returns the improved mask.
        """
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any vertex (Greedily fill)
            # This is equivalent to checking if we are a maximal independent set.
            # If not, add one.
            # We can just run greedy_fill on the complement.
            # But greedy_fill might add multiple. That's fine.
            new_mask = greedy_fill(current_mask)
            if new_mask != current_mask:
                current_mask = new_mask
                improved = True
                continue
            
            # 2. Try swaps (1-for-1)
            # For each v in set, remove it.
            # For each u not in set, if u is not adjacent to (set - {v}), 
            #    then we can swap v with u.
            #    This maintains size. 
            #    Does it help? Only if it allows a subsequent addition.
            #    So, after swap, try greedy_fill.
            
            # Optimization: 
            # Instead of trying all swaps blindly, we can look for "promising" swaps.
            # A swap (v, u) is promising if u is adjacent to v, but not to others? 
            # Actually, if u is not adjacent to set\{v}, the swap is valid.
            # 
            # Let's try a limited number of random swaps to avoid O(N^2) per step.
            
            verts_in_set = [i for i in range(m) if (current_mask >> i) & 1]
            verts_not_in_set = [i for i in range(m) if not (current_mask >> i) & 1]
            
            # Try a few random swaps
            swaps_tried = 0
            max_swaps = 50
            
            while swaps_tried < max_swaps and not improved:
                if not verts_in_set or not verts_not_in_set:
                    break
                v = random.choice(verts_in_set)
                u = random.choice(verts_not_in_set)
                swaps_tried += 1
                
                # Check if swap is valid: u must not conflict with current_mask without v
                temp_mask = current_mask & ~(1 << v)
                if can_add(u, temp_mask):
                    # Perform swap
                    new_mask = temp_mask | (1 << u)
                    
                    # Try to extend from this new state
                    extended = greedy_fill(new_mask)
                    
                    # If extended is larger, accept
                    if extended.bit_count() > current_mask.bit_count():
                        current_mask = extended
                        improved = True
                    # Even if not immediately larger, the swap might be neutral but lead to better future?
                    # For simplicity, only accept if it leads to immediate improvement (via greedy fill).
                    # This is a form of simulated annealing with T=0 (greedy).
        
        return current_mask

    # Initial Greedy Runs
    for _ in range(200):
        if time.time() > deadline:
            break
        sol = greedy_fill(0)
        size = sol.bit_count()
        if size > best_size:
            best_size = size
            best_set_mask = sol
            
        # Immediate local search on the result
        improved_sol = local_search_improve(sol)
        if improved_sol.bit_count() > best_size:
            best_size = improved_sol.bit_count()
            best_set_mask = improved_sol

    # Main Loop: Kick-based Local Search
    # We perturb the best solution and try to improve.
    
    while time.time() < deadline:
        # Perturb best solution
        # Remove 1 to 5 random vertices
        num_removals = random.randint(1, 5)
        current_mask = best_set_mask
        
        verts_in_set = [i for i in range(m) if (current_mask >> i) & 1]
        
        if len(verts_in_set) < num_removals:
            # Restart if set is too small
            new_mask = greedy_fill(0)
        else:
            to_remove = random.sample(verts_in_set, num_removals)
            for v in to_remove:
                current_mask &= ~(1 << v)
            
            # Greedy fill
            new_mask = greedy_fill(current_mask)
            
            # Local search on the result
            new_mask = local_search_improve(new_mask)
        
        if new_mask.bit_count() > best_size:
            best_size = new_mask.bit_count()
            best_set_mask = new_mask
            
    # Extract result
    result = []
    for i in range(m):
        if (best_set_mask >> i) & 1:
            result.append(i)
            
    return np.array(result, dtype=np.int64)
