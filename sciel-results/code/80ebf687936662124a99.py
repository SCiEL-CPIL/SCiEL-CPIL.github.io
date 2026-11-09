import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 256
    
    # 1. Build Conflict Graph using Bitmasks
    # conflict_mask[i] is a bitmask where bit j is set if i and j conflict
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                
    # Precompute bit_count for fast popcount if needed, but Python 3.10+ has int.bit_count()
    # We assume modern Python.
    
    def popcount(x):
        return x.bit_count()
        
    def bits(x):
        res = []
        while x:
            lsb = x & -x
            res.append(lsb.bit_length() - 1)
            x ^= lsb
        return res

    # 2. Branch and Bound for Maximum Independent Set
    # We use a recursive search with bitsets.
    # State: current_set (bitmask), candidates (bitmask of nodes that can be added)
    # Upper Bound: Greedy coloring of the candidate subgraph.
    
    best_size = 0
    best_mask = 0
    
    # Initial greedy to get a good lower bound
    def greedy_mis(candidates):
        mask = 0
        rem = candidates
        while rem:
            # Pick node with min degree in rem? Or just first?
            # Let's pick the one with min degree to keep options open
            best_node = -1
            min_deg = N + 1
            r = rem
            while r:
                lsb = r & -r
                node = lsb.bit_length() - 1
                deg = popcount(conflict_mask[node] & rem)
                if deg < min_deg:
                    min_deg = deg
                    best_node = node
                r ^= lsb
            if best_node == -1: break
            mask |= (1 << best_node)
            rem &= ~(1 << best_node)
            rem &= ~conflict_mask[best_node]
        return mask

    initial_mask = greedy_mis((1 << (N + 1)) - 2)
    best_size = popcount(initial_mask)
    best_mask = initial_mask
    
    # Greedy Coloring for Upper Bound
    # Returns the number of colors needed to color the subgraph induced by 'candidates'
    # This number is an upper bound on the size of the MIS in that subgraph.
    def greedy_color_bound(candidates):
        if candidates == 0:
            return 0
        colors = 0
        remaining = candidates
        while remaining:
            colors += 1
            # Select a maximal independent set from 'remaining'
            # This is one color class
            current_color_class = 0
            rem_scan = remaining
            while rem_scan:
                lsb = rem_scan & -rem_scan
                node = lsb.bit_length() - 1
                # Check if node conflicts with any node already in current_color_class
                # If it doesn't, we can add it to this color class
                if (conflict_mask[node] & current_color_class) == 0:
                    current_color_class |= lsb
                    # Remove from remaining
                    remaining &= ~lsb
                rem_scan ^= lsb
        return colors

    # Recursive Branch and Bound
    # To avoid recursion depth issues and overhead, we can use a stack or limit depth.
    # However, for N=256, the depth is at most 256, which is fine for Python if we manage it.
    # But function call overhead is high. Let's try a standard recursive approach with strict pruning.
    
    def branch_and_bound(candidates, current_size, current_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Upper bound check
        # A simple bound is current_size + popcount(candidates)
        if current_size + popcount(candidates) <= best_size:
            return
            
        # Stronger bound: Greedy Coloring
        # This is expensive, so we only do it if the simple bound is close or if depth is small?
        # Actually, greedy coloring is the standard effective bound for MIS.
        # Let's optimize: only compute if popcount(candidates) is not too small?
        # Or just always compute? For N=256, it might be slow.
        # Let's try a hybrid: simple bound first, then coloring bound.
        
        # Optimization: If the number of candidates is small, we can brute force or use a faster bound.
        # But let's stick to the coloring bound as it's strong.
        
        # To speed up coloring, we can cache it? No, candidates change.
        
        # Let's implement a faster coloring bound specific to this context.
        # We pick the node with highest degree in the remaining graph to color?
        # Standard greedy coloring for bound:
        # 1. While candidates not empty:
        #    a. Find a maximal independent set (MIS) in the subgraph.
        #    b. Decrement candidates by MIS.
        #    c. Increment color count.
        
        # This is effectively the same as the greedy_mis function but returning count.
        
        # Let's use a heuristic for node selection in B&B:
        # Pick a node v from candidates.
        # Branch 1: Include v. New candidates = candidates & ~conflict_mask[v] & ~(1<<v).
        # Branch 2: Exclude v. New candidates = candidates & ~(1<<v).
        
        # Which node to pick?
        # Heuristic: Pick the node with the highest degree in the candidate subgraph.
        # This tends to prune the "include" branch faster (removes many candidates).
        
        # Find node with max degree in candidates
        max_deg = -1
        pivot_node = -1
        r = candidates
        while r:
            lsb = r & -r
            node = lsb.bit_length() - 1
            deg = popcount(conflict_mask[node] & candidates)
            if deg > max_deg:
                max_deg = deg
                pivot_node = node
            r ^= lsb
            
        if pivot_node == -1:
            return
            
        pivot_bit = 1 << pivot_node
        
        # Branch 1: Include pivot_node
        new_candidates_1 = candidates & ~conflict_mask[pivot_node] & ~pivot_bit
        branch_and_bound(new_candidates_1, current_size + 1, current_mask | pivot_bit)
        
        # Check time
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude pivot_node
        new_candidates_2 = candidates & ~pivot_bit
        branch_and_bound(new_candidates_2, current_size, current_mask)

    # Run B&B
    # Start with all nodes
    all_nodes = (1 << (N + 1)) - 2
    branch_and_bound(all_nodes, 0, 0)
    
    # If B&B didn't finish or found a smaller set than greedy (unlikely if B&B is correct), 
    # we should keep the best found. 
    # Note: The initial greedy set is a valid lower bound. B&B explores from scratch.
    # If B&B completes, best_mask is the exact MIS.
    # If B&B times out, best_mask is the best found so far.
    
    # We can also try to improve the best found by local search if there is time left.
    if time.time() < deadline - 5:
        # Local search on the best mask found
        cur_mask = best_mask
        cur_size = best_size
        removed_mask = all_nodes & ~cur_mask
        
        # Simple 1-swap improvement
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            r = removed_mask
            while r:
                lsb = r & -r
                node = lsb.bit_length() - 1
                # Check if node can be added directly
                if (conflict_mask[node] & cur_mask) == 0:
                    cur_mask |= lsb
                    removed_mask &= ~lsb
                    cur_size += 1
                    improved = True
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur_mask
                # Check if 1-swap is possible
                conflicts = conflict_mask[node] & cur_mask
                if conflicts and (conflicts & (conflicts - 1)) == 0:
                    s_bit = conflicts
                    s_node = s_bit.bit_length() - 1
                    # Check if node is compatible with rest of cur_mask
                    if (conflict_mask[node] & (cur_mask ^ s_bit)) == 0:
                        # Perform swap
                        cur_mask ^= s_bit
                        cur_mask |= lsb
                        removed_mask ^= s_bit
                        removed_mask ^= lsb
                        # Size unchanged, but structure changed.
                        # We don't update best_size here as it's same size, 
                        # but it might enable future adds.
                        improved = True
                r ^= lsb
                
    return bits(best_mask)

# EVOLVE-BLOCK-END
