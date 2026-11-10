"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    N = 256
    # Precompute conflict masks
    # conflicts[i] is a bitmask where bit j is 1 if i and j conflict
    conflicts = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    all_nodes_mask = (1 << (N + 1)) - 2  # Bits 1 to N set
    
    def get_size(mask):
        return bin(mask).count("1")
        
    # Initial solution: Greedy by degree (fewest conflicts first) or Random
    # Let's use a few random restarts with Local Search to ensure robustness
    
    best_mask = 0
    best_size = 0
    
    # Time limit check
    start_time = time.time()
    time_limit = 55.0
    
    # We will run multiple rounds of local search
    # Each round starts with a random/greedy seed
    
    nodes = list(range(1, N + 1))
    
    for restart in range(200): # Limit restarts, local search is heavier
        if time.time() - start_time > time_limit:
            break
            
        # Initialize current mask
        # Strategy: Random permutation greedy
        random.shuffle(nodes)
        current_mask = 0
        for num in nodes:
            if (conflicts[num] & current_mask) == 0:
                current_mask |= (1 << num)
        
        current_size = get_size(current_mask)
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
        # Local Search Phase
        # Try to improve current_mask
        # 1. Add all possible non-conflicting nodes
        # 2. If stuck, try swaps
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # Step 1: Add any node that can be added
            # Find candidates: not in current_mask, no conflict with current_mask
            # candidates = all_nodes_mask & ~current_mask
            # We need to filter candidates: (conflicts[c] & current_mask) == 0
            
            # Efficient way:
            # Iterate through bits not in current_mask?
            # Or maintain a list of candidates?
            # For N=256, iterating 1..256 is fast.
            
            added_any = False
            for num in range(1, N + 1):
                if not (current_mask & (1 << num)):
                    if (conflicts[num] & current_mask) == 0:
                        current_mask |= (1 << num)
                        added_any = True
                        # Don't break, add all possible in this pass?
                        # Adding all possible is a greedy completion.
            
            if added_any:
                improved = True
                current_size = get_size(current_mask)
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            
            # Step 2: If no node could be added, try a swap
            if not added_any:
                # Pick a random node to remove
                # Get list of nodes in current_mask
                in_set = []
                for num in range(1, N + 1):
                    if current_mask & (1 << num):
                        in_set.append(num)
                
                if not in_set:
                    break
                    
                # Try removing a random node and see if we can add more
                # To be efficient, we can try removing a node and then greedily adding all possible
                # We do this for a few random removals or all if small?
                # Let's try 10 random swaps
                
                for _ in range(10):
                    if time.time() - start_time > time_limit:
                        break
                    if not in_set:
                        break
                    rem_idx = random.randint(0, len(in_set) - 1)
                    rem_node = in_set[rem_idx]
                    
                    # Create new mask without rem_node
                    new_mask = current_mask & ~(1 << rem_node)
                    
                    # Try to add all possible nodes to new_mask
                    for num in range(1, N + 1):
                        if not (new_mask & (1 << num)):
                            if (conflicts[num] & new_mask) == 0:
                                new_mask |= (1 << num)
                    
                    new_size = get_size(new_mask)
                    if new_size > current_size:
                        current_mask = new_mask
                        current_size = new_size
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        # Update in_set for next iteration? 
                        # It's easier to just restart the inner while loop logic 
                        # or recalculate in_set.
                        # Since we found an improvement, we break out of swap loop 
                        # and go back to "Add all possible" phase which will 
                        # stabilize the set.
                        break
                    else:
                        # If no improvement, we could keep trying other swaps 
                        # but usually we accept the best swap or just one.
                        # Let's just try the next random swap in the for loop.
                        pass
                
                # If we tried swaps and didn't find improvement, 
                # the `improved` flag remains False, and the while loop exits.
                # This means we are at a local optimum.
                
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result
# EVOLVE-BLOCK-END
