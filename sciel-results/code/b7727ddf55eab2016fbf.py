import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: kill at 55s, limit is 60s
    deadline = start_time + 55.0
    
    N = 256
    
    # 1. Identify valid nodes (those that don't have self-conflicts)
    # A node 'a' is invalid if 2*a is a perfect square
    valid_nodes = []
    for a in range(1, N + 1):
        s = 2 * a
        r = int(s**0.5)
        if r * r != s:
            valid_nodes.append(a)
            
    # Map valid node values to indices 0..M-1
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    # 2. Precompute neighbor masks
    # neighbor_masks[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    # Note: We only care about conflicts between distinct valid nodes.
    # If a+b is a square, they conflict.
    squares = set(i*i for i in range(1, int((2*N)**0.5) + 2))
    
    neighbor_masks = [0] * M
    
    for i, a in enumerate(valid_nodes):
        mask = 0
        for j, b in enumerate(valid_nodes):
            if i == j:
                continue
            if a + b in squares:
                mask |= (1 << j)
        neighbor_masks[i] = mask
        
    # 3. Search for Maximum Independent Set using greedy + local search
    best_size = 0
    best_mask = 0
    
    def greedy_random():
        """Generate a random greedy independent set."""
        indices = list(range(M))
        random.shuffle(indices)
        
        current_mask = 0
        count = 0
        
        for idx in indices:
            if (neighbor_masks[idx] & current_mask) == 0:
                current_mask |= (1 << idx)
                count += 1
                
        return current_mask, count
    
    def local_search(mask, count, time_limit):
        """Try to improve by adding/removing/swapping nodes."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adding a node not in the set
            for idx in range(M):
                if (mask >> idx) & 1:
                    continue
                if (neighbor_masks[idx] & mask) == 0:
                    mask |= (1 << idx)
                    count += 1
                    improved = True
                    break
            
            if time.time() >= time_limit:
                break
            
            # Try swapping: remove one node, add two non-conflicting nodes
            for remove_idx in range(M):
                if not ((mask >> remove_idx) & 1):
                    continue
                
                # Temporarily remove
                mask_without = mask & ~(1 << remove_idx)
                
                # Find candidates that can be added
                candidates = []
                for add_idx in range(M):
                    if (mask_without >> add_idx) & 1:
                        continue
                    if (neighbor_masks[add_idx] & mask_without) == 0:
                        candidates.append(add_idx)
                
                # Try to find 2+ non-conflicting candidates from the list
                if len(candidates) >= 2:
                    # Try pairs
                    found_pair = False
                    for i in range(len(candidates)):
                        if found_pair or time.time() >= time_limit:
                            break
                        a = candidates[i]
                        for j in range(i + 1, len(candidates)):
                            if time.time() >= time_limit:
                                break
                            b = candidates[j]
                            # Check if a and b conflict with each other
                            if (neighbor_masks[a] >> b) & 1:
                                continue
                            # They don't conflict, so we can add both
                            new_mask = mask_without | (1 << a) | (1 << b)
                            if new_mask != mask:
                                mask = new_mask
                                count = count - 1 + 2
                                improved = True
                                found_pair = True
                                break
                    
                    if found_pair:
                        break
        
        return mask, count
    
    # Phase 1: Multiple random greedy initializations
    num_greedy = 200
    for _ in range(num_greedy):
        if time.time() >= deadline:
            break
        mask, count = greedy_random()
        if count > best_size:
            best_size = count
            best_mask = mask
    
    # Phase 2: Local search on best solution and random perturbations
    remaining_time = deadline - time.time()
    if remaining_time > 0.5:
        # Local search on current best
        best_mask, best_size = local_search(best_mask, best_size, deadline)
        
        # Try local search from different starting points
        perturb_count = 0
        while time.time() < deadline - 0.2:
            # Perturb the best solution
            mask = best_mask
            # Randomly remove 1-3 nodes
            nodes_in_set = [i for i in range(M) if (mask >> i) & 1]
            if len(nodes_in_set) > 3:
                remove_count = random.randint(1, min(3, len(nodes_in_set) - 1))
                to_remove = random.sample(nodes_in_set, remove_count)
                for r in to_remove:
                    mask &= ~(1 << r)
                count = best_size - remove_count
            else:
                mask, count = greedy_random()
            
            # Local search
            mask, count = local_search(mask, count, deadline)
            
            if count > best_size:
                best_size = count
                best_mask = mask
            
            perturb_count += 1
            
    # Convert best_mask back to list of values
    result = []
    for i, v in enumerate(valid_nodes):
        if (best_mask >> i) & 1:
            result.append(v)
            
    # If for some reason result is empty (shouldn't happen), return a single valid node
    if not result and valid_nodes:
        result = [valid_nodes[0]]
        
    return result
