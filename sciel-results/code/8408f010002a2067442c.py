import time
import random

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 55
    
    N = 256
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    def popcount(x):
        return x.bit_count()
    
    def bits(x):
        res = []
        while x:
            lsb = x & -x
            res.append(lsb.bit_length() - 1)
            x ^= lsb
        return res

    all_mask = (1 << (N + 1)) - 2
    
    def greedy_construct():
        selected_mask = 0
        remaining_mask = all_mask
        while remaining_mask:
            # Pick node with min degree in remaining
            best_node = -1
            best_deg = N + 1
            r = remaining_mask
            while r:
                lsb = r & -r
                node = lsb.bit_length() - 1
                deg = popcount(conflict_mask[node] & remaining_mask)
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
                r ^= lsb
            
            if best_node == -1:
                break
                
            selected_mask |= (1 << best_node)
            remaining_mask &= ~(1 << best_node)
            remaining_mask &= ~conflict_mask[best_node]
            
        return selected_mask

    def local_search(start_mask):
        cur_mask = start_mask
        cur_size = popcount(cur_mask)
        improved = True
        
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Maximize (Add any non-conflicting node)
            # We iterate through all non-selected nodes. If it doesn't conflict with cur_mask, add it.
            # To be efficient, we can just check all 1..N.
            for i in range(1, N + 1):
                bit = 1 << i
                if (cur_mask & bit) == 0:
                    if (conflict_mask[i] & cur_mask) == 0:
                        cur_mask |= bit
                        cur_size += 1
                        improved = True
            
            # Phase 2: 1-for-1 Swap / Remove-Refill
            # Try removing each node in cur_mask and see if we can add more.
            # We do this in a single pass to avoid order bias, or random order.
            # Let's try a random order of selected nodes to escape local optima.
            selected_nodes = bits(cur_mask)
            random.shuffle(selected_nodes)
            
            for u in selected_nodes:
                if time.time() > deadline - 2:
                    break
                u_bit = 1 << u
                if (cur_mask & u_bit) == 0:
                    continue # Already removed? shouldn't happen in this loop unless we modified cur_mask, which we don't yet.
                
                # Tentatively remove u
                new_mask = cur_mask ^ u_bit
                new_size = cur_size - 1
                
                # Try to add any node that is not in new_mask and doesn't conflict with new_mask
                # Optimization: Only check nodes that were blocked by u? 
                # No, removing u might unblock nodes that didn't conflict with u but were just not picked?
                # No, if a node didn't conflict with u, it would have been added in Phase 1 if it didn't conflict with the rest.
                # So only nodes that conflict with u are candidates for addition.
                # Let's check all non-selected nodes for simplicity, N=256 is small.
                for i in range(1, N + 1):
                    bit = 1 << i
                    if (new_mask & bit) == 0:
                        if (conflict_mask[i] & new_mask) == 0:
                            new_mask |= bit
                            new_size += 1
                
                if new_size > cur_size:
                    cur_mask = new_mask
                    cur_size = new_size
                    improved = True
        
        return cur_mask, cur_size

    # Initial greedy
    best_mask = greedy_construct()
    best_size = popcount(best_mask)
    
    # Refine
    best_mask, best_size = local_search(best_mask)
    
    # Random restarts with perturbation
    for i in range(50):
        if time.time() > deadline - 2:
            break
        
        # Perturb best solution
        cur_mask = best_mask
        nodes = bits(cur_mask)
        if len(nodes) > 5:
            # Remove a few random nodes to escape local optimum
            k = random.randint(3, 8)
            to_remove = random.sample(nodes, k)
            for n in to_remove:
                cur_mask &= ~(1 << n)
        
        refined_mask, refined_size = local_search(cur_mask)
        
        if refined_size > best_size:
            best_size = refined_size
            best_mask = refined_mask
            
    return bits(best_mask)

# EVOLVE-BLOCK-END
