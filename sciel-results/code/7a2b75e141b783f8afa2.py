import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # 1. Identify forbidden nodes (2*a is a perfect square)
    # These nodes cannot be in the set because a+a = 2a must not be a square.
    forbidden = set()
    for a in range(1, n + 1):
        s = 2 * a
        r = math.isqrt(s)
        if r * r == s:
            forbidden.add(a)
            
    valid_nodes = [i for i in range(1, n + 1) if i not in forbidden]
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    num_nodes = len(valid_nodes)
    
    # 2. Build conflict masks
    # conflict_masks[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    # We exclude self-loops in the mask logic for simplicity, but they are already filtered by 'forbidden'.
    # However, if i is not forbidden, 2*valid_nodes[i] is not a square, so no self-loop.
    
    conflict_masks = [0] * num_nodes
    
    # Precompute squares up to 2*256 = 512
    max_sum = 2 * n
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
        
    for i in range(num_nodes):
        a = valid_nodes[i]
        mask = 0
        for s in squares:
            b = s - a
            if b in node_to_idx:
                j = node_to_idx[b]
                if j != i:
                    mask |= (1 << j)
        conflict_masks[i] = mask
        
    # Helper: Check if a mask is a valid independent set
    # Returns True if no two nodes in mask conflict
    def is_valid_mask(mask):
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            # Check if any neighbor of idx is also in mask
            if (mask & conflict_masks[idx]) != 0:
                return False
            temp ^= lsb
        return True

    # Helper: Greedy initialization
    # Returns a mask of a valid independent set
    def greedy_init(order=None):
        if order is None:
            # Sort by degree (number of conflicts)
            # Lower degree first is usually better for MIS
            degrees = [bin(conflict_masks[i]).count('1') for i in range(num_nodes)]
            order = sorted(range(num_nodes), key=lambda x: degrees[x])
            
        mask = 0
        # We need to simulate removing nodes.
        # Instead of a full graph copy, we can just check compatibility with current mask.
        # But greedy usually removes neighbors.
        # Simple greedy: Iterate order, if compatible with current mask, add.
        # This is O(N^2) per restart, which is fine.
        
        for idx in order:
            # Check if idx is compatible with current mask
            if (mask & conflict_masks[idx]) == 0:
                mask |= (1 << idx)
        return mask

    best_mask = 0
    best_size = 0
    
    # Run multiple greedy restarts with different orderings
    random.seed(42)
    
    # 1. Deterministic Greedy (Min Degree)
    m = greedy_init()
    size = bin(m).count('1')
    if size > best_size:
        best_size = size
        best_mask = m

    # 2. Random Greedy
    for _ in range(20):
        order = list(range(num_nodes))
        random.shuffle(order)
        m = greedy_init(order)
        size = bin(m).count('1')
        if size > best_size:
            best_size = size
            best_mask = m
            
    # 3. Local Search: 1-for-2 Swap
    # Try to remove 1 node and add 2 nodes.
    
    def local_search_1_for_2(init_mask):
        current_mask = init_mask
        current_size = bin(current_mask).count('1')
        improved = True
        
        while improved:
            improved = False
            # Try removing each node u in current_mask
            # To make it faster, iterate over a copy of the indices
            indices_in_set = []
            temp = current_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                indices_in_set.append(idx)
                temp ^= lsb
                
            random.shuffle(indices_in_set)
            
            for u in indices_in_set:
                # Check if u is still in current_mask (it might have been removed by a previous improvement in this loop? 
                # No, we break on improvement and restart the while loop)
                # But we are iterating over a static list. If we improve, we restart the outer while.
                # So u is in current_mask at start of iteration.
                
                # Remove u
                candidate_mask = current_mask & ~(1 << u)
                
                # Find candidates that can be added to candidate_mask
                # A node v is compatible if (candidate_mask & conflict_masks[v]) == 0
                # And v is not in candidate_mask.
                
                # We need to find TWO such nodes v, w that are also compatible with each other.
                
                # Get list of potential adders
                adders = []
                for v in range(num_nodes):
                    if candidate_mask & (1 << v):
                        continue
                    if (candidate_mask & conflict_masks[v]) == 0:
                        adders.append(v)
                        
                # Try to find a pair in adders
                # If len(adders) < 2, skip
                if len(adders) < 2:
                    continue
                    
                # Optimization: If we just need ANY pair, we can iterate.
                # To avoid O(N^2) check for every u, we can be smart.
                # But N=256, adders size is small on average?
                # Let's just iterate.
                
                found_pair = False
                # Shuffle adders to randomize which pair we find first
                random.shuffle(adders)
                
                for i in range(len(adders)):
                    v = adders[i]
                    for j in range(i + 1, len(adders)):
                        w = adders[j]
                        # Check if v and w are compatible
                        if (conflict_masks[v] & (1 << w)) == 0:
                            # Valid pair found
                            new_mask = candidate_mask | (1 << v) | (1 << w)
                            new_size = current_size + 1
                            
                            # Accept and restart search
                            current_mask = new_mask
                            current_size = new_size
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                            
        return current_mask, current_size

    # Run Local Search on best_mask
    current_mask, current_size = local_search_1_for_2(best_mask)
    if current_size > best_size:
        best_size = current_size
        best_mask = current_mask

    # 4. Iterative Improvement with Kicks
    # If stuck, kick: remove k random nodes, then greedily add back.
    
    def kick_and_recover(mask, k=3):
        # Remove k random nodes from mask
        indices_in_set = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            indices_in_set.append(idx)
            temp ^= lsb
            
        if len(indices_in_set) < k:
            return mask
            
        to_remove = random.sample(indices_in_set, k)
        kicked_mask = mask
        for idx in to_remove:
            kicked_mask &= ~(1 << idx)
            
        # Greedy recovery: Add nodes that fit
        # Sort candidates by degree or randomly
        # Let's try adding any compatible node
        # We can use a simple greedy pass over all nodes
        # But to be fast, we just do one pass.
        
        # Actually, let's just do a few greedy restarts from the kicked state?
        # Or just one greedy pass.
        
        # Simple greedy pass:
        # We want to maximize size.
        # Iterate nodes, if compatible, add.
        # To make it better, iterate in order of "potential" (low degree).
        
        # For speed, let's just do a single random greedy pass.
        order = list(range(num_nodes))
        random.shuffle(order)
        
        # Ensure we keep the existing kicked_mask nodes?
        # The greedy function I wrote starts from 0.
        # Let's modify greedy to start from an initial mask.
        
        def greedy_from(start_mask, order):
            mask = start_mask
            for idx in order:
                if mask & (1 << idx):
                    continue
                if (mask & conflict_masks[idx]) == 0:
                    mask |= (1 << idx)
            return mask

        # Try a few random orders for recovery
        best_kicked = kicked_mask
        best_kicked_size = bin(kicked_mask).count('1')
        
        for _ in range(5):
            order = list(range(num_nodes))
            random.shuffle(order)
            m = greedy_from(kicked_mask, order)
            s = bin(m).count('1')
            if s > best_kicked_size:
                best_kicked_size = s
                best_kicked = m
                
        return best_kicked

    # Main Loop: Local Search + Kicks
    current_mask = best_mask
    current_size = best_size
    
    while time.time() - start_time < deadline:
        # Try local search 1-for-2
        new_mask, new_size = local_search_1_for_2(current_mask)
        
        if new_size > current_size:
            current_size = new_size
            current_mask = new_mask
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        else:
            # Stuck. Try a kick.
            # Probability of kick increases as we get stuck?
            # Just do a kick.
            kicked_mask = kick_and_recover(current_mask, k=random.randint(2, 5))
            kicked_size = bin(kicked_mask).count('1')
            
            if kicked_size > current_size:
                current_mask = kicked_mask
                current_size = kicked_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # If kick didn't help immediately, maybe try local search on the kicked version?
                # Or just try another kick.
                # Let's try local search on the kicked version if it's not too small.
                if kicked_size > best_size - 5:
                    new_mask, new_size = local_search_1_for_2(kicked_mask)
                    if new_size > current_size:
                        current_mask = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                    # If still stuck, loop continues and might kick again

    # Convert best_mask to list
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(valid_nodes[idx])
        temp ^= lsb
        
    if not result:
        result = [1]
        
    return result
