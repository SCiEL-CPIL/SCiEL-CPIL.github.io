import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s/90s limits
    
    N = 256
    # Precompute conflicts
    # conflicts[i] is a bitmask of j such that (i*j) % (i+j) == 0
    # We only care about distinct pairs, but for bitmask logic, 
    # if i conflicts with j, j conflicts with i.
    
    conflicts = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
                
    # Function to check if x is compatible with set represented by mask
    # mask has bits set for elements in the set
    # x is compatible if (conflicts[x] & mask) == 0
    def is_compatible(x, mask):
        return (conflicts[x] & mask) == 0

    def mask_to_set(mask):
        return [i for i in range(1, N + 1) if (mask >> i) & 1]

    def set_to_mask(s):
        m = 0
        for i in s:
            m |= (1 << i)
        return m

    best_size = 0
    best_mask = 0

    # Heuristic 1: Greedy Minimum Degree
    # We maintain a set of available nodes.
    # At each step, pick the node with the minimum degree within the available set.
    # Add it to the solution, and remove it and its neighbors from available.
    
    def greedy_min_degree():
        available = (1 << (N + 1)) - 1
        # Clear bit 0
        available &= ~(1 << 0)
        
        solution_mask = 0
        current_available = available
        
        while current_available:
            # Find node with min degree in current_available
            # Degree of node x in current_available is popcount(conflicts[x] & current_available)
            best_node = -1
            min_deg = N + 1
            
            # Iterate through set bits in current_available
            temp = current_available
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                x = lsb.bit_length() - 1
                deg = bin(conflicts[x] & current_available).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_node = x
                temp -= lsb
                
            if best_node == -1:
                break
                
            # Add best_node to solution
            solution_mask |= (1 << best_node)
            
            # Remove best_node and its neighbors from current_available
            # Neighbors are conflicts[best_node]
            # We want to keep nodes that are NOT best_node and NOT neighbors
            # So we mask out (1 << best_node) | conflicts[best_node]
            remove_mask = (1 << best_node) | conflicts[best_node]
            current_available &= ~remove_mask
            
        return solution_mask

    # Heuristic 2: Greedy Random with Compatibility Check
    # Start with empty set, add nodes in random order if compatible.
    def greedy_random():
        solution_mask = 0
        nodes = list(range(1, N + 1))
        random.shuffle(nodes)
        
        for x in nodes:
            if is_compatible(x, solution_mask):
                solution_mask |= (1 << x)
                
        return solution_mask

    # Local Search: 1-Remove, 2-Add (or more)
    # Try to improve a given mask
    def local_search(mask):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
                
            nodes_in_set = mask_to_set(mask)
            # Shuffle to avoid bias
            random.shuffle(nodes_in_set)
            
            for x in nodes_in_set:
                if time.time() > deadline - 2:
                    break
                    
                # Remove x
                new_mask = mask & ~(1 << x)
                
                # Find all candidates y not in new_mask that are compatible with new_mask
                # candidates = ~new_mask & available_all
                # But we need to check compatibility.
                
                # Let's just iterate through all nodes not in new_mask
                # This is O(N^2) per swap attempt, which might be slow if done naively for many steps.
                # Optimization: Precompute "compatible_with_mask" is hard because mask changes.
                
                # However, N=256 is small.
                # Let's collect candidates first.
                candidates = []
                for y in range(1, N + 1):
                    if not (new_mask >> y) & 1:
                        if is_compatible(y, new_mask):
                            candidates.append(y)
                            
                # Now, can we add 2 or more from candidates?
                # Greedily add as many as possible from candidates
                added = 0
                # Sort candidates by degree? Or just random?
                # To maximize chance of adding 2, we should pick ones that don't conflict with each other.
                # Let's try to add them one by one.
                # Since we want to increase size, we need at least 1 added if we removed 1 to stay same,
                # but we want to increase, so we need >= 2 added.
                
                # Let's try to build a maximal independent set from candidates
                # using a simple greedy approach on the subgraph induced by candidates.
                
                # Actually, simpler:
                # If len(candidates) < 2, we can't increase size (we removed 1, can add at most 1).
                # If len(candidates) >= 2, check if there are two compatible candidates.
                
                if len(candidates) >= 2:
                    # Try to find a pair (u, v) in candidates such that u, v compatible
                    # And then check if we can add more?
                    # Let's just greedily add from candidates.
                    # We want to maximize the number added.
                    
                    # To ensure we find a good pair, we can just greedily add.
                    # But greedy on candidates might only add 1 if they all conflict with each other.
                    # If we add 1, size is same. We only accept if size increases.
                    # So we need to add >= 2.
                    
                    # Let's try to add candidates in random order and see how many we get.
                    # If we get >= 2, we have an improvement.
                    
                    random.shuffle(candidates)
                    trial_mask = new_mask
                    added_count = 0
                    for y in candidates:
                        if is_compatible(y, trial_mask):
                            trial_mask |= (1 << y)
                            added_count += 1
                            
                    if added_count > 1:
                        # Improvement found
                        mask = trial_mask
                        improved = True
                        break # Restart the while loop with new mask
                    
        return mask

    # Run multiple restarts with different heuristics
    num_restarts = 200
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if i % 2 == 0:
            mask = greedy_min_degree()
        else:
            mask = greedy_random()
            
        # Polish with local search
        mask = local_search(mask)
        
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            
    return mask_to_set(best_mask)
