import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    n = 256
    # 1. Precompute conflict bitsets
    # conflict[i] is an integer bitset where bit j is 1 if i and j conflict
    conflict = [0] * n
    
    for i in range(n):
        a = i + 1
        bits = 0
        for j in range(n):
            if i == j: continue
            b = j + 1
            s = a + b
            p = a * b
            if p % s == 0:
                bits |= (1 << j)
        conflict[i] = bits
        
    # 2. Initial Set: All Odds
    mask = 0
    for i in range(0, n, 2):
        mask |= (1 << i)
        
    best_mask = mask
    best_size = bin(mask).count('1')
    
    def get_size(m):
        return bin(m).count('1')

    def greedy_add(mask):
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if not (mask & (1 << v)):
                    if not (mask & conflict[v]):
                        mask |= (1 << v)
                        improved = True
                        break
        return mask

    def local_search(mask):
        improved = True
        iterations = 0
        while improved and (time.time() - start_time) < deadline:
            improved = False
            iterations += 1
            
            # Step A: Greedy Add
            new_mask = greedy_add(mask)
            if new_mask != mask:
                mask = new_mask
                improved = True
                continue
            
            # Step B: 1-Exchange
            # For each v not in S, if |N(v) \cap S| == 1, try swap
            found_swap = False
            for v in range(n):
                if mask & (1 << v):
                    continue
                confs = mask & conflict[v]
                if confs and (confs & (confs - 1)) == 0:
                    u_bit = confs
                    u_idx = u_bit.bit_length() - 1
                    new_mask = (mask & ~u_bit) | (1 << v)
                    # Check if we can add any w
                    # Optimization: Check if there is any w not in new_mask with (new_mask & conflict[w]) == 0
                    # We can just call greedy_add on new_mask and see if size increases
                    # But greedy_add is expensive if called for every candidate.
                    # Let's do a quick check:
                    # Is there any w?
                    # If not, we might still accept the swap if it reduces "potential"?
                    # For now, only accept if it allows immediate growth.
                    
                    # Quick check for existence of w
                    # We can iterate w.
                    # To speed up, we can precompute "free" nodes? No, mask changes.
                    
                    # Let's just try the swap and run greedy_add.
                    # If size(new) > size(old), accept.
                    # Since size(new_mask) == size(mask) initially, greedy_add must add at least 1.
                    
                    # To avoid expensive greedy_add calls for every single candidate,
                    # we can limit the number of candidates we fully evaluate per pass.
                    # Or we can just do it, as n=256 is small.
                    
                    test_mask = greedy_add(new_mask)
                    if get_size(test_mask) > get_size(mask):
                        mask = test_mask
                        improved = True
                        found_swap = True
                        break # Restart loop
            
            if found_swap:
                continue
            
            # Step C: 2-Exchange
            # Try removing 2 and adding 2
            # This is O(n^2) pairs of v1, v2.
            # For each pair, check if union of their conflicts in S is size <= 2.
            # If size 2, let u1, u2 be the blockers.
            # Remove u1, u2. Add v1, v2.
            # Then try to add more.
            
            found_2swap = False
            # To make this faster, we only try 2-swaps if we have time
            if (time.time() - start_time) < deadline - 5.0:
                for v1 in range(n):
                    if mask & (1 << v1):
                        continue
                    confs1 = mask & conflict[v1]
                    if get_size(confs1) > 2:
                        continue
                    for v2 in range(v1 + 1, n):
                        if mask & (1 << v2):
                            continue
                        confs2 = mask & conflict[v2]
                        if get_size(confs2) > 2:
                            continue
                        
                        union_confs = confs1 | confs2
                        if get_size(union_confs) > 2:
                            continue
                        
                        # Extract u1, u2
                        # If union_confs has 1 bit, it means v1 and v2 share the same blocker?
                        # Or one has 0?
                        # If v1 has 1 blocker u, v2 has 1 blocker u. Union is {u}. Size 1.
                        # We remove u, add v1, v2. Size change: -1 + 2 = +1.
                        # This is effectively a 1-2 exchange.
                        
                        # If union has 2 bits, u1, u2.
                        # Remove u1, u2, add v1, v2. Size change: 0.
                        # Then try to add more.
                        
                        new_mask = mask
                        # Remove all bits in union_confs
                        new_mask &= ~union_confs
                        # Add v1, v2
                        new_mask |= (1 << v1)
                        new_mask |= (1 << v2)
                        
                        test_mask = greedy_add(new_mask)
                        if get_size(test_mask) > get_size(mask):
                            mask = test_mask
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
            
            if found_2swap:
                continue

            # Step D: Random Perturbation (if stuck)
            # If we have been iterating without improvement, perturb.
            # We don't track "stuck" count easily here, but if improved is False, we break the while loop.
            # To allow restarts, we should not break immediately.
            # Instead, if no improvement, we can do a random kick.
            
            # Let's add a random kick if we have time
            if (time.time() - start_time) < deadline - 10.0:
                # Remove 2 random nodes from S
                # Add 2 random nodes not in S (if compatible)
                # This is a heuristic kick.
                
                # Simple kick: Remove a random node from S, then greedy add.
                # This might decrease size, but allows escape.
                # We only keep it if it eventually leads to a better size?
                # That's hard to track.
                # Better: Just break and rely on multiple starts?
                # The current function is a single start.
                
                # Let's just break here. The multi-start will be handled by calling construct?
                # No, construct is called once.
                # We should implement multiple starts inside construct.
                
                break
        
        return mask

    # Multi-start strategy
    # We have ~50s.
    # Run local search a few times with different initializations or perturbations.
    
    # Start 1: Odds
    best_mask = local_search(mask)
    best_size = get_size(best_mask)
    
    # Start 2: Greedy from scratch?
    # Or perturb best_mask?
    
    if (time.time() - start_time) < deadline - 10.0:
        # Try to improve best_mask with a different strategy?
        # Let's try a greedy construction with a different order.
        # Or just run local_search on a perturbed version of best_mask.
        
        # Perturb: Remove 5 random nodes from best_mask
        perturbed_mask = best_mask
        indices = [i for i in range(n) if perturbed_mask & (1 << i)]
        if len(indices) > 5:
            random.shuffle(indices)
            for i in indices[:5]:
                perturbed_mask &= ~(1 << i)
        
        new_mask = local_search(perturbed_mask)
        if get_size(new_mask) > best_size:
            best_mask = new_mask
            best_size = get_size(best_mask)

    # Convert mask to list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return result
