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
    # adj[u] is a bitmask where bit v is 1 if u and v conflict (i.e., v is in the forbidden neighborhood of u)
    # Note: If (u-v) % m is in forbidden, then u and v cannot be in the independent set together.
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask
        
    # Helper: Check if a set (bitmask) is valid independent set
    # For performance in the main loop, we will maintain validity incrementally or check conflicts directly.
    
    best_size = 0
    best_set_mask = 0
    best_set_list = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    # Strategy: Greedy Construction with Random Restarts (Kick)
    # We try to build a large independent set by greedily adding vertices.
    # To avoid getting stuck in local optima, we occasionally "kick" by removing a few vertices and re-greedy-ing.
    
    # A better heuristic for Greedy: 
    # At each step, choose a vertex from the remaining candidates that has the minimum degree 
    # within the remaining candidates. This is the "Minimum Degree" heuristic for MIS.
    
    def greedy_mis(P_mask, kick_count=0):
        """
        Greedy construction of an Independent Set from available vertices P_mask.
        If kick_count > 0, we simulate a 'kick' by randomly removing kick_count vertices 
        from the initial pool before starting, or rather, we start with a subset.
        Actually, a common kick is: take current best, remove k random elements, add back greedily.
        Here we implement a pure greedy from a random subset of P_mask.
        """
        current_set = 0
        available = P_mask
        
        # If kicking, we might start with a random subset of the available vertices
        # to break symmetry and avoid the same greedy path.
        if kick_count > 0 and available:
            # Randomly select a subset of available vertices to start with
            # This is a heuristic to explore different parts of the space
            # We pick a random subset of size roughly half? Or just randomize the order.
            # Let's stick to randomizing the order of selection.
            pass
            
        # To make it effective, we randomize the order in which we consider vertices.
        # However, the "Minimum Degree" heuristic requires dynamic recalculation.
        
        # Simple Greedy with Randomized Order:
        # 1. Get list of vertices in P_mask
        # 2. Shuffle them
        # 3. Iterate, adding vertex if no conflict with current_set
        
        # For better performance, let's use a slightly smarter greedy:
        # Maintain a list of candidates.
        # In each step, pick the candidate with the fewest neighbors in the *remaining* candidates.
        
        # Given N=205, a simpler randomized greedy is fast enough.
        vertices = [i for i in range(m) if (P_mask >> i) & 1]
        random.shuffle(vertices)
        
        for v in vertices:
            # Check if v conflicts with any vertex in current_set
            # Conflict if (current_set & adj[v]) != 0
            if (current_set & adj[v]) == 0:
                current_set |= (1 << v)
                
        return current_set

    # Strategy 2: Local Search (Simulated Annealing style or Simple Hill Climbing with kicks)
    # We will use a "Kick" strategy:
    # 1. Start with a greedy solution.
    # 2. While time remains:
    #    a. Remove k random vertices from the current best set.
    #    b. Try to add back vertices greedily from the remaining available vertices.
    #    c. If the new set is larger, accept it.
    
    # Initialize with a few greedy runs
    for _ in range(100):
        if time.time() > deadline:
            break
        sol = greedy_mis((1 << m) - 1)
        size = sol.bit_count()
        if size > best_size:
            best_size = size
            best_set_mask = sol
    
    # Kick-based local search
    # We keep the best set found so far.
    # In each iteration, we take the best set, remove a few random elements, 
    # and then greedily fill the gaps.
    
    # To make this more powerful, we can also try to swap vertices.
    # But the "remove and re-greedy" is often very effective.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Randomly perturb the best solution
        # Remove 1 to 5 random vertices from the current best set
        num_removals = random.randint(1, 5)
        current_mask = best_set_mask
        
        # Get list of vertices in current_mask
        verts_in_set = [i for i in range(m) if (current_mask >> i) & 1]
        
        if len(verts_in_set) < num_removals:
            # If set is too small, just restart greedy
            new_mask = greedy_mis((1 << m) - 1)
        else:
            # Remove random vertices
            to_remove = random.sample(verts_in_set, num_removals)
            for v in to_remove:
                current_mask &= ~(1 << v)
            
            # Now, try to greedily add vertices from the remaining available pool
            # The available pool is all vertices NOT in current_mask
            # But we should prioritize vertices that were just removed or neighbors?
            # Standard approach: Just run a greedy fill on the *complement* of current_mask,
            # but constrained to not conflict with current_mask.
            
            # Let's define the "available" vertices as those not in current_mask.
            # We want to add as many as possible without conflict.
            
            # We can use the same greedy_mis logic but starting with current_mask as the base.
            # Let's write a specific function for this: extend_mask
            def extend_mask(base_mask):
                available = ((1 << m) - 1) & ~base_mask
                # Shuffle available vertices to break symmetry
                verts = [i for i in range(m) if (available >> i) & 1]
                random.shuffle(verts)
                
                for v in verts:
                    if (base_mask & adj[v]) == 0:
                        base_mask |= (1 << v)
                return base_mask

            new_mask = extend_mask(current_mask)
        
        new_size = new_mask.bit_count()
        if new_size > best_size:
            best_size = new_size
            best_set_mask = new_mask
            
        # Optional: If we find a solution that is significantly better, we can restart the search from there
        # The loop naturally does this by always perturbing the current best.

    # Extract result
    result = []
    for i in range(m):
        if (best_set_mask >> i) & 1:
            result.append(i)
            
    # If no valid set found (should not happen, empty set is valid but size 0 is bad), 
    # return a single element if possible, or empty.
    # But greedy should always find at least 1.
    
    return np.array(result, dtype=np.int64)
