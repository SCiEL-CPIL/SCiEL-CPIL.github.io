import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit

    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build Graph using Bitmasks for Speed
    # neighbors_mask[i] is an integer where bit j is 1 if (i-j)%m is in forbidden
    neighbors_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        neighbors_mask[i] = mask

    # Helper: Convert set to mask
    def set_to_mask(s):
        mask = 0
        for v in s:
            mask |= (1 << v)
        return mask

    # Helper: Check if vertex v is compatible with mask M
    def is_compatible(v, M):
        return (neighbors_mask[v] & M) == 0

    # Helper: Greedy Min-Degree Construction
    def greedy_min_degree(seed_val):
        rng = random.Random(seed_val)
        candidates = set(range(m))
        selected = set()
        
        while candidates:
            min_deg = m
            min_nodes = []
            # To speed up, we can approximate degree or just compute exactly
            # m=205 is small, exact is fine.
            # We compute degree in the induced subgraph of candidates
            for v in candidates:
                # Count neighbors in candidates
                # neighbors_mask[v] & candidates_mask
                # But we don't have candidates_mask updated dynamically in O(1) easily without rebuilding
                # Let's just iterate candidates for each v? O(m^2) per step -> O(m^3) total.
                # 205^3 ~ 8.6M operations. In Python, this might be ~1-2 seconds per run.
                # Too slow for many restarts?
                
                # Optimization: Maintain a list of candidates and a mask.
                # Actually, let's use a simpler heuristic for speed: 
                # Randomized Greedy is faster. 
                # Let's try to optimize Min-Degree by maintaining a candidate mask.
                
                pass # Placeholder for logic below

        # Re-implementing Greedy Min-Degree with Mask for Candidates
        candidates_mask = (1 << m) - 1
        selected = set()
        
        while candidates_mask:
            min_deg = m
            min_nodes = []
            # Iterate through bits in candidates_mask
            temp_mask = candidates_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                temp_mask ^= lsb
                
                # Count degree in candidates
                deg = bin(neighbors_mask[v] & candidates_mask).count('1')
                
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
            
            # Pick random from min_nodes
            v = rng.choice(min_nodes)
            selected.add(v)
            
            # Remove v and its neighbors from candidates
            # New candidates = candidates - {v} - neighbors(v)
            remove_mask = (1 << v) | neighbors_mask[v]
            candidates_mask &= ~remove_mask
            
        return selected

    # Helper: Random Sequential Greedy
    def random_sequential_greedy(seed_val):
        rng = random.Random(seed_val)
        nodes = list(range(m))
        rng.shuffle(nodes)
        selected = set()
        selected_mask = 0
        for v in nodes:
            if is_compatible(v, selected_mask):
                selected.add(v)
                selected_mask |= (1 << v)
        return selected

    # Helper: Local Search Improvement
    def local_search(initial_set, seed_val):
        rng = random.Random(seed_val)
        current = set(initial_set)
        current_mask = set_to_mask(current)
        best = set(current)
        best_mask = current_mask
        best_size = len(current)
        
        # Phase 1: Maximal Extension (Add any possible vertex)
        def extend(s, mask):
            improved = True
            while improved:
                improved = False
                for v in range(m):
                    if v in s:
                        continue
                    if is_compatible(v, mask):
                        s.add(v)
                        mask |= (1 << v)
                        improved = True
                        if len(s) > len(best):
                            best = set(s)
                            best_mask = mask
                            best_size = len(s)
                        break # Restart to find more
            return s, mask

        current, current_mask = extend(current, current_mask)
        
        # Phase 2: Swap Search
        # Try 1-swaps: remove u, add v
        # If valid, accept if it allows extension or randomly to explore
        for _ in range(200): # Limit iterations per local search call
            if time.time() > deadline:
                break
            if not current:
                break
            
            u = rng.choice(list(current))
            v = rng.randint(0, m - 1)
            if v in current:
                continue
            
            # Check if v is compatible with current \ {u}
            temp_mask = current_mask & ~(1 << u)
            if is_compatible(v, temp_mask):
                # Valid swap
                # New state: current - {u} + {v}
                new_set = set(current)
                new_set.discard(u)
                new_set.add(v)
                new_mask = current_mask & ~(1 << u) | (1 << v)
                
                # Try to extend this new state
                ext_set, ext_mask = extend(new_set, new_mask)
                
                if len(ext_set) > best_size:
                    best = ext_set
                    best_mask = ext_mask
                    best_size = len(ext_set)
                    current = ext_set
                    current_mask = ext_mask
                else:
                    # Accept with some probability to escape local minima
                    # Heuristic: Accept if new state has "more potential"
                    # Potential = number of vertices compatible with current set
                    # Expensive to compute fully, so just random accept
                    if rng.random() < 0.3:
                        current = ext_set
                        current_mask = ext_mask

        return best

    best_set = set()
    best_size = 0
    
    # Run multiple strategies
    # 1. Greedy Min-Degree with various seeds
    attempts = 0
    while time.time() < deadline - 10: # Keep 10s for final polish
        seed = attempts
        if seed % 2 == 0:
            s = greedy_min_degree(seed)
        else:
            s = random_sequential_greedy(seed)
            
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
        # Local search on the result
        # Only run local search if we have time
        if time.time() < deadline - 20:
            improved_s = local_search(s, seed)
            if len(improved_s) > best_size:
                best_set = improved_s
                best_size = len(improved_s)
                
        attempts += 1
        
        # If we've been stuck for a while, maybe increase swap aggressiveness?
        # For now, just keep restarting.
        
    # Final Polish on the best set found
    if time.time() < deadline - 5:
        final_s = local_search(best_set, 99999)
        if len(final_s) > best_size:
            best_set = final_s
            best_size = len(final_s)
            
    return np.array(list(best_set), dtype=np.int64)
