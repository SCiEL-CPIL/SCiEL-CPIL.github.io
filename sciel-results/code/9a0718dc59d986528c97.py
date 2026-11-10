import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]
        
    # 3. Build adjacency bitmasks
    # index i corresponds to valid_vertices[i]
    # adj[i] is a bitmask where bit j is 1 if valid_vertices[i] + valid_vertices[j] is a square
    adj = [0] * n_vars
    
    # Precompute sum check for speed? 
    # We just iterate pairs. n_vars is ~200. 200*200/2 = 20k checks. Very fast.
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Helper: Get degree from bitmask
    # Python int.bit_count() is fast (C implementation)
    degrees = [adj[i].bit_count() for i in range(n_vars)]
    
    # Helper: Check if a specific vertex can be added to a set represented by bitmask
    # selected_mask: bits set for vertices in current set
    # v_idx: index of vertex to add
    # A vertex v can be added if no neighbor of v is in selected_mask.
    # i.e., (adj[v_idx] & selected_mask) == 0
    
    def greedy_construct(order):
        """
        order: list of indices in the order to consider
        returns: bitmask of selected vertices
        """
        selected_mask = 0
        for idx in order:
            if (adj[idx] & selected_mask) == 0:
                selected_mask |= (1 << idx)
        return selected_mask

    def local_search(initial_mask):
        """
        Hill climbing on the independent set.
        Tries to add vertices. If stuck, tries to remove one and re-greedy.
        """
        current_mask = initial_mask
        current_size = current_mask.bit_count()
        
        # Precompute list of indices for iteration
        indices = list(range(n_vars))
        
        improved = True
        iter_count = 0
        max_iters = 100 # Limit local search steps to avoid long loops on one restart
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # 1. Try to add any vertex
            # We can find all vertices not in current_mask and not adjacent to current_mask
            # forbidden_mask = union of neighbors of all vertices in current_mask
            # This is expensive to compute from scratch every time.
            # Instead, just iterate through candidates.
            
            # Optimization: Keep a list of "blocked" vertices? 
            # Or just iterate all n_vars. n_vars is small (~200).
            
            added_any = False
            for idx in indices:
                if not (current_mask & (1 << idx)):
                    if (adj[idx] & current_mask) == 0:
                        current_mask |= (1 << idx)
                        current_size += 1
                        improved = True
                        added_any = True
                        break # Restart to check for more additions
                    
            if added_any:
                continue
            
            # 2. If no direct add, try "remove one, re-greedy"
            # This is a 1-move perturbation.
            # Find a vertex in current_mask to remove.
            # Heuristic: Remove a vertex that blocks the most other vertices?
            # Or just random?
            # Let's try a few random removals.
            
            # Get list of selected indices
            selected_indices = [i for i in range(n_vars) if (current_mask & (1 << i))]
            
            if not selected_indices:
                break
                
            # Try removing a few candidates
            # To save time, just try 1 random removal per iteration if stuck
            victim = random.choice(selected_indices)
            current_mask &= ~(1 << victim)
            current_size -= 1
            
            # Re-greedy from this state
            # We can use the same greedy logic but starting from current_mask
            # Actually, simpler: just run greedy on the remaining candidates
            # But greedy depends on order. 
            # Let's just try to add back vertices in random order.
            
            # Shuffle indices for re-greedy
            shuffled_indices = indices[:]
            random.shuffle(shuffled_indices)
            
            new_mask = current_mask
            for idx in shuffled_indices:
                if not (new_mask & (1 << idx)):
                    if (adj[idx] & new_mask) == 0:
                        new_mask |= (1 << idx)
                        
            if new_mask.bit_count() > current_size:
                current_mask = new_mask
                current_size = new_mask.bit_count()
                improved = True
            else:
                # If size didn't increase, this move was bad or neutral.
                # We might loop if we keep removing and not adding enough.
                # To prevent infinite loops on bad moves, we can break or accept neutral.
                # If neutral, we are stuck. Break local search.
                improved = False
                # Optionally revert? No, just break.
                
        return current_mask, current_size

    best_mask = 0
    best_size = 0
    
    # Multi-start Local Search
    # We can do many more restarts now due to speed.
    
    # Initial good ordering: sort by degree (ascending)
    base_order = sorted(range(n_vars), key=lambda x: degrees[x])
    
    max_restarts = 5000 # Can be high because each restart is fast
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # 1. Create a perturbed order
        # Mix base order with random noise
        order = base_order[:]
        # Randomly swap some elements or shuffle a subset
        # Or just shuffle completely for high diversity?
        # Let's try: 50% chance full shuffle, 50% chance slight perturbation of base
        if random.random() < 0.5:
            random.shuffle(order)
        else:
            # Perturb: swap a few random pairs
            for _ in range(5):
                i, j = random.sample(range(n_vars), 2)
                order[i], order[j] = order[j], order[i]
                
        # 2. Greedy
        mask = greedy_construct(order)
        size = mask.bit_count()
        
        if size > best_size:
            best_size = size
            best_mask = mask
            
        # 3. Local Search
        if size > best_size - 2: # Only refine if close to best? Or always?
             # Always refine if we have time
             mask, size = local_search(mask)
             if size > best_size:
                 best_size = size
                 best_mask = mask

    # Fallback: ensure validity and convert to list
    if best_mask == 0:
        # Greedy from scratch with sorted order
        best_mask = greedy_construct(base_order)
        
    result = []
    for i in range(n_vars):
        if best_mask & (1 << i):
            result.append(valid_vertices[i])
            
    return result
