import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # 1. Determine valid candidates (diagonal check)
    # 2a must not be a perfect square
    squares = set(k*k for k in range(1, 23)) # max sum 512, sqrt(512) ~ 22.6
    
    candidates = []
    for i in range(1, 257):
        if 2 * i not in squares:
            candidates.append(i)
            
    N = len(candidates)
    
    # 2. Build conflict masks
    # mask[i] is a bitset where bit j is 1 if candidates[i] + candidates[j] is a square
    masks = [0] * N
    for i in range(N):
        ci = candidates[i]
        m = 0
        for j in range(N):
            if i == j:
                continue
            cj = candidates[j]
            if ci + cj in squares:
                m |= (1 << j)
        masks[i] = m
        
    # 3. Greedy initialization
    def greedy_solve():
        remaining = (1 << N) - 1
        selected = 0
        while remaining:
            # Pick node with min degree in remaining
            best_node = -1
            min_deg = float('inf')
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                r ^= lsb
                
            if best_node == -1:
                break
                
            selected |= (1 << best_node)
            # Remove best_node and its neighbors
            remaining &= ~(masks[best_node] | (1 << best_node))
            
        return selected, selected.bit_count()

    best_set, best_size = greedy_solve()
    
    # 4. Branch and Bound
    # To speed up, we can order nodes by degree initially?
    # The masks are fixed. The 'remaining' set changes.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(remaining, current_size, current_set):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        # Pruning
        if current_size + remaining.bit_count() <= best_size:
            return
            
        if remaining == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Heuristic: Pick a node to branch on
        # Choose node with max degree in remaining subgraph? 
        # Or just pick the lowest index?
        # Let's pick the node with the highest degree in the remaining graph to prune faster?
        # Actually, for MIS, picking a high-degree node removes many options, which might be bad if we want a large set.
        # But for B&B, we just need to split.
        # A common strategy is to pick the node with the maximum degree in the remaining graph.
        
        best_node = -1
        max_deg = -1
        r = remaining
        while r:
            lsb = r & (-r)
            idx = lsb.bit_length() - 1
            deg = (masks[idx] & remaining).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_node = idx
            r ^= lsb
            
        if best_node == -1:
            return
            
        bit = 1 << best_node
        
        # Branch 1: Include best_node
        # New remaining: remove best_node and its neighbors
        new_remaining_1 = remaining & ~(masks[best_node] | bit)
        branch_and_bound(new_remaining_1, current_size + 1, current_set | bit)
        
        # Branch 2: Exclude best_node
        new_remaining_2 = remaining & ~bit
        branch_and_bound(new_remaining_2, current_size, current_set)
        
    # Initial call
    initial_remaining = (1 << N) - 1
    branch_and_bound(initial_remaining, 0, 0)
    
    # Convert best_set bitset back to list of values
    result = []
    for i in range(N):
        if best_set & (1 << i):
            result.append(candidates[i])
            
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# To match the interface exactly, I should wrap it or just define it here.
# The prompt asks to keep function names. The current program has `construct`.
# I will place the code inside the block or define it globally.
# The seed program defines `construct` globally. I will do the same.
# EVOLVE-BLOCK-END
