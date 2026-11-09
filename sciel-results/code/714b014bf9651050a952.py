import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline to be safe
    
    # 1. Identify valid nodes
    # 2a is not a perfect square
    def is_square(n):
        if n < 0: return False
        r = int(math.isqrt(n))
        return r * r == n
    
    valid_nodes = []
    for i in range(1, 257):
        if not is_square(2 * i):
            valid_nodes.append(i)
    
    n_valid = len(valid_nodes)
    # Map valid node value to index 0..n_valid-1
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build conflict masks
    # conflict_mask[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a perfect square
    # Note: The problem states a+b is never a perfect square for a<=b.
    # Since the set is a subset, if i and j are both in the set, their sum must not be a square.
    # The condition applies to all pairs.
    
    conflict_masks = [0] * n_valid
    
    # Squares to check: min sum 1+2=3 (but 1+1=2 not square, 1+2=3 not square, 1+3=4 square)
    # Max sum 256+255 = 511. Squares: 4, 9, ..., 484.
    squares = [k*k for k in range(2, 23) if k*k <= 511]
    
    for i in range(n_valid):
        vi = valid_nodes[i]
        mask = 0
        for j in range(i + 1, n_valid):
            vj = valid_nodes[j]
            s = vi + vj
            if s in squares: # Optimization: use set for squares? 22 squares, list is fine.
                mask |= (1 << j)
                conflict_masks[j] |= (1 << i)
        conflict_masks[i] |= mask
        
    # 3. Greedy Heuristic for initial best
    def greedy_iset():
        available = (1 << n_valid) - 1
        set_nodes = []
        while available:
            # Pick node with min degree in available? Or just first?
            # Let's try picking node with min degree to leave more options
            best_node = -1
            min_deg = n_valid
            temp_avail = available
            while temp_avail:
                # Get lowest set bit
                lsb = temp_avail & -temp_avail
                idx = lsb.bit_length() - 1
                temp_avail ^= lsb
                deg = (available & conflict_masks[idx]).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
            if best_node == -1:
                break
            set_nodes.append(best_node)
            # Remove best_node and its neighbors
            available &= ~(1 << best_node)
            available &= ~conflict_masks[best_node]
        return set_nodes
    
    initial_set = greedy_iset()
    best_size = len(initial_set)
    best_set = initial_set[:]
    
    # 4. Branch and Bound
    # State: available_mask, current_set_list
    # We need to track current set to reconstruct the solution
    
    # Greedy Coloring Bound
    def coloring_bound(available):
        if available == 0:
            return 0
        colors = 0
        remaining = available
        while remaining:
            colors += 1
            # Pick a node from remaining
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            # Remove idx and its neighbors from 'remaining' for this color class
            # The nodes that can share this color are those not adjacent to idx and not idx itself
            # Actually, greedy coloring assigns color to a maximal independent set.
            # We pick idx, assign color. Then we remove idx and all its neighbors from the pool of nodes that can have this color.
            # The remaining nodes in 'remaining' that are NOT neighbors of idx and not idx can potentially have this color?
            # No, standard greedy coloring:
            # 1. Pick a vertex v.
            # 2. Assign current color to v.
            # 3. Remove v and its neighbors from the graph.
            # 4. Repeat on the remaining graph with next color.
            # This produces a valid coloring. The number of colors is an upper bound on the independence number.
            
            # Nodes to keep for next iteration of coloring loop (i.e., nodes NOT colored with current color)
            # Wait, the algorithm is:
            # While graph not empty:
            #   Pick v
            #   Color v with c
            #   Remove v and N(v)
            #   c++
            
            # So 'remaining' becomes 'remaining' minus (v and N(v)).
            remove_mask = (1 << idx) | conflict_masks[idx]
            remaining &= ~remove_mask
            
        return colors

    def search(available, current_set):
        nonlocal best_size, best_set
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if available == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        # Upper bound
        ub = coloring_bound(available)
        if len(current_set) + ub <= best_size:
            return
            
        # Select pivot node
        # Heuristic: Highest degree in the subgraph induced by 'available'
        pivot = -1
        max_deg = -1
        temp_avail = available
        while temp_avail:
            lsb = temp_avail & -temp_avail
            idx = lsb.bit_length() - 1
            temp_avail ^= lsb
            deg = (available & conflict_masks[idx]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = idx
        
        if pivot == -1:
            return
            
        # Branch 1: Include pivot
        # New available: remove pivot and its neighbors
        new_avail_include = available & ~(1 << pivot) & ~conflict_masks[pivot]
        current_set.append(pivot)
        search(new_avail_include, current_set)
        current_set.pop()
        
        # Branch 2: Exclude pivot
        # New available: remove pivot
        new_avail_exclude = available & ~(1 << pivot)
        search(new_avail_exclude, current_set)

    # Initial full mask
    full_mask = (1 << n_valid) - 1
    
    # To improve performance, we can sort nodes or use a better pivot strategy, 
    # but let's run the B&B.
    # Note: Python recursion depth. Max depth is n_valid (256). Default is 1000. OK.
    
    # Run search
    search(full_mask, [])
    
    # Convert best_set indices back to values
    result = [valid_nodes[i] for i in best_set]
    
    # Final verification
    def is_valid(subset):
        vals = sorted(subset)
        for i in range(len(vals)):
            a = vals[i]
            if is_square(2 * a):
                return False
            for j in range(i + 1, len(vals)):
                b = vals[j]
                if is_square(a + b):
                    return False
        return True
        
    if not is_valid(result):
        # Fallback to seed or simple greedy
        # Seed was range(1, 257, 3). Let's check if it's valid.
        seed = list(range(1, 257, 3))
        if is_valid(seed):
            return seed
        # If seed invalid, return valid_nodes filtered by greedy
        return [valid_nodes[i] for i in initial_set]
        
    return result
