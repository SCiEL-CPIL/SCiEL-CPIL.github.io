import time
import random
import math

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # Safety margin before 60s kill
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # Build adjacency masks
    # mask[i] is a bitmask where bit j is 1 if i+1 and j+1 form a square sum
    # Note: We exclude self-loops (i==j) from the mask for the "add" check logic 
    # but we must ensure we don't pick self-conflicting vertices at all.
    
    # First, identify valid vertices (no self-loop)
    valid_vertices = []
    for i in range(N):
        v = i + 1
        if 2 * v not in squares:
            valid_vertices.append(i)
    
    # Map original index to new index in valid_vertices for compactness? 
    # No, just use original indices in bitmasks. N=256 fits in Python int easily.
    
    masks = [0] * N
    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            continue # Skip self-conflicting
        m = 0
        for sq in squares:
            u = sq - v
            if 1 <= u <= N and u != v:
                u_idx = u - 1
                m |= (1 << u_idx)
        masks[i] = m
    
    best_size = 0
    best_set = set()
    
    # Initial Greedy Solution (Random Order)
    def run_greedy_and_improve():
        nonlocal best_size, best_set
        
        # Shuffle valid vertices
        candidates = valid_vertices[:]
        random.shuffle(candidates)
        
        current_mask = 0
        current_set = set()
        
        # 1. Greedy Phase
        for v in candidates:
            if masks[v] & current_mask == 0:
                current_mask |= (1 << v)
                current_set.add(v)
                
        # 2. Local Search Phase
        # Try to add any remaining valid vertex
        # Then try swaps to potentially enable additions
        improved = True
        while improved:
            improved = False
            
            # Try direct additions
            added = False
            for v in valid_vertices:
                if v not in current_set:
                    if masks[v] & current_mask == 0:
                        current_mask |= (1 << v)
                        current_set.add(v)
                        added = True
                        improved = True
            if added:
                continue # Restart loop to check for more additions
            
            # Try swaps: remove one, add one
            # This keeps size same but might break deadlocks
            swapped = False
            for v_out in list(current_set):
                # Temporarily remove v_out
                temp_mask = current_mask & ~(1 << v_out)
                
                # Look for a vertex v_in that can be added
                for v_in in valid_vertices:
                    if v_in in current_set:
                        continue
                    if masks[v_in] & temp_mask == 0:
                        # Perform swap
                        current_mask = temp_mask | (1 << v_in)
                        current_set.remove(v_out)
                        current_set.add(v_in)
                        swapped = True
                        improved = True
                        break # Restart swap loop with new state
                if swapped:
                    break
                    
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    # Run multiple iterations
    # Estimate iterations: 
    # Greedy is O(N^2) bit ops ~ very fast.
    # Local search is O(N^3) worst case per restart, but usually much less.
    # With N=256, even 1000 restarts should be fast in Python with bitmasks.
    
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT:
            break
        run_greedy_and_improve()
        
    # Final check: ensure best_set is not empty
    if not best_set:
        # Fallback: just return any valid vertex
        for v in valid_vertices:
            best_set.add(v)
            break
            
    return sorted(list(best_set))
