"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict matrix as a boolean numpy array
    # conflict[i, j] is True if lcm(numbers[i], numbers[j]) <= 256
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i, j] = True
                conflict[j, i] = True
                
    # Weights: reciprocal sum. w[i] = 1.0 / numbers[i]
    weights = np.array([1.0 / x for x in numbers])
    
    # Branch and Bound for Maximum Weight Independent Set
    # State: (current_sum, current_set_mask, candidate_indices)
    # candidate_indices: list of indices that can still be added (compatible with current_set)
    
    best_sum = 0.0
    best_mask = np.zeros(n, dtype=bool)
    
    # Initial ordering: Sort indices by weight descending to find good solutions early
    # This helps in pruning branches that cannot beat the current best
    order = np.argsort(-weights)
    
    # We will maintain a list of "available" candidates.
    # Instead of passing full masks, we pass a list of candidate indices.
    # When we include a node, we remove it and all its neighbors from the candidate list.
    # When we exclude a node, we just remove it.
    
    # To speed up, we can precompute the neighbor lists (including self) for each node
    # But since n is small (255), simple list filtering is okay if optimized.
    
    # Let's use a recursive function with a stack to avoid recursion depth limits if needed, 
    # though n=255 is likely fine for recursion if we prune well. 
    # However, Python recursion limit might be an issue if the tree is deep. 
    # We'll use an iterative stack-based B&B.
    
    # Stack items: (current_sum, current_indices_list, current_mask)
    # current_indices_list: list of indices from 'order' that are compatible with current_mask
    # We need to be careful with the "order" - we process them in a fixed order to ensure determinism and allow bounding.
    
    # Actually, standard B&B on Independent Set usually processes vertices in a fixed order.
    # Let's stick to the sorted order 'order'.
    # At step k, we decide whether to include order[k] or not.
    # If we include order[k], we must remove order[k] and all its neighbors from future consideration.
    # If we exclude order[k], we just move to k+1.
    
    # To make bounding effective, we need the sum of weights of all remaining candidates (upper bound).
    
    # Let's restructure:
    # We process vertices in the order defined by 'order'.
    # State: (idx, current_sum, excluded_mask)
    # idx: index in 'order' we are currently considering.
    # current_sum: sum of weights of selected vertices.
    # excluded_mask: boolean array of vertices that are no longer candidates (either selected or conflict with selected).
    # Note: We don't need to track "selected" separately for validity if we update excluded_mask correctly.
    
    # Optimization: 
    # Precompute for each vertex v in 'order', the list of vertices in 'order' that are neighbors of v.
    # When we select v, we mark v and its neighbors as excluded.
    
    # Let's use a stack for B&B.
    # Each stack item: (next_idx, current_sum, excluded_mask)
    # next_idx: index in 'order' array to consider next.
    
    # Initial state:
    # next_idx = 0
    # current_sum = 0.0
    # excluded_mask = all False
    
    # Upper bound calculation:
    # The maximum possible additional sum is the sum of weights of all vertices in 'order'[next_idx:] 
    # that are NOT in excluded_mask.
    # We can precompute a suffix sum of weights in 'order' to quickly estimate the bound, 
    # but we must subtract the weights of excluded vertices.
    # Since n is small, we can just sum the remaining available weights.
    # To speed this up, we can maintain a running sum of available weights? 
    # No, because exclusions depend on the path.
    # However, summing ~255 floats is very fast.
    
    stack = []
    initial_mask = np.zeros(n, dtype=bool)
    stack.append((0, 0.0, initial_mask))
    
    # To improve pruning, we can sort 'order' such that high weight nodes come first.
    # 'order' is already sorted by -weights.
    
    # One optimization: If the upper bound (current_sum + sum(weights of available candidates)) <= best_sum, prune.
    
    # Let's add a time check inside the loop.
    
    while stack:
        if elapsed() > DEADLINE:
            break
            
        next_idx, current_sum, excluded_mask = stack.pop()
        
        # If we've processed all vertices
        if next_idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                # Reconstruct mask? 
                # We didn't store the mask of selected nodes, only excluded.
                # We need to track selected nodes to return the result.
                # Let's modify state to include selected_mask.
                # State: (next_idx, current_sum, selected_mask, excluded_mask)
                # This doubles memory but n is small.
                pass
            continue
        
        # Calculate upper bound
        # Sum of weights of all vertices from next_idx to end that are not excluded
        # We can approximate this by sum(weights[next_idx:]) - sum(weights of excluded in that range)
        # Or just sum directly.
        remaining_weights = 0.0
        # Optimization: Precompute suffix sums of weights in 'order'
        # suffix_sum[i] = sum(weights[order[i]] for i in range(n, 0, -1))
        
        # Let's precompute suffix sums outside the loop
        # But 'excluded_mask' changes.
        
        # Quick bound:
        # max_possible = current_sum + sum(weights[order[k]] for k in range(next_idx, n) if not excluded_mask[order[k]])
        
        # To make this fast, let's keep a list of available indices?
        # No, let's just compute it. n=255, so 255 ops.
        
        bound_sum = current_sum
        for k in range(next_idx, n):
            idx = order[k]
            if not excluded_mask[idx]:
                bound_sum += weights[idx]
                
        if bound_sum <= best_sum:
            continue
            
        # Branching
        v = order[next_idx]
        
        # Option 1: Include v
        # Check if v is not excluded
        if not excluded_mask[v]:
            new_selected_mask = selected_mask.copy() # Need selected_mask in state
            # Wait, I didn't include selected_mask in the stack item above.
            # Let's restart the stack definition to include selected_mask.
            
            # Re-define stack item: (next_idx, current_sum, selected_mask, excluded_mask)
            # We need to pop the current state and push new states.
            
            # Let's refine the loop structure to be cleaner.
            
            pass
        
        # Option 2: Exclude v
        # new_excluded_mask = excluded_mask.copy()
        # new_excluded_mask[v] = True
        # push (next_idx + 1, current_sum, selected_mask, new_excluded_mask)
        
        # This approach is getting complex to implement correctly in one shot without errors.
        # Let's use a simpler recursive B&B with lru_cache? No, state is too large for cache.
        
        # Let's try a different structural change: 
        # **Genetic Algorithm** or **Particle Swarm**? 
        # Or **Local Search with Large Neighborhood Search (LNS)**?
        
        # The prompt asks for a MATERIAL DIFFERENCE. 
        # B&B is the most robust "different" approach for exact/near-exact optimization on small graphs.
        # Let's implement a clean recursive B&B.
        
        pass

    # Since the iterative B&B implementation above got messy in thought, let's write a clean recursive one.
    # Recursion depth is at most 255, which is safe.
    
    # Precompute suffix sums of weights in 'order' for faster bounding?
    # suffix_w[i] = sum(weights[order[i]] for i in range(n, 0, -1))
    suffix_w = np.zeros(n + 1)
    for i in range(n - 1, -1, -1):
        suffix_w[i] = suffix_w[i + 1] + weights[order[i]]
    
    best_sum = 0.0
    best_mask = np.zeros(n, dtype=bool)
    
    # To speed up the bound calculation, we can't just use suffix_w because some nodes are excluded.
    # However, if we process in a fixed order, and we maintain a bitmask of excluded nodes,
    # the bound is: current_sum + sum(weights[order[k]] for k in range(next_idx, n) if not excluded[order[k]])
    
    # We can optimize this by noting that if we exclude a node, we also exclude its neighbors.
    # This reduces the number of available nodes significantly.
    
    def bb_solve(next_idx, current_sum, selected_mask, excluded_mask):
        nonlocal best_sum, best_mask
        
        if elapsed() > DEADLINE:
            return
            
        if next_idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask = selected_mask.copy()
            return
        
        # Upper Bound
        # Sum of weights of available nodes from next_idx onwards
        # We can approximate by taking suffix_w[next_idx] and subtracting weights of excluded nodes in that range.
        # But calculating exact sum is O(n).
        # Let's try to be smart: 
        # The maximum gain is at most suffix_w[next_idx].
        # If current_sum + suffix_w[next_idx] <= best_sum, prune.
        # This is a weak bound but very fast.
        if current_sum + suffix_w[next_idx] <= best_sum:
            return
            
        # Stronger bound: sum of actually available weights
        # Given n=255, O(n) check is ~255 float adds. Very fast.
        actual_bound = current_sum
        for k in range(next_idx, n):
            idx = order[k]
            if not excluded_mask[idx]:
                actual_bound += weights[idx]
                
        if actual_bound <= best_sum:
            return
            
        v = order[next_idx]
        
        # Branch 1: Include v
        if not excluded_mask[v]:
            new_selected = selected_mask.copy()
            new_selected[v] = True
            new_excluded = excluded_mask.copy()
            
            # Mark v and its neighbors as excluded
            new_excluded[v] = True
            # Find neighbors of v in the remaining list? 
            # We can just iterate all and check conflict.
            # Or precompute neighbor masks.
            
            # Precomputed neighbor masks for speed
            # neighbor_mask[i] is a boolean array where neighbor_mask[i][j] is True if i and j conflict.
            # We already have 'conflict' matrix.
            
            # Update excluded mask: OR with conflict[v]
            new_excluded = new_excluded | conflict[v]
            # Note: conflict[v] includes v? 
            # In our precomputation, conflict[i, i] is False. 
            # So we need to set new_excluded[v] = True explicitly.
            
            bb_solve(next_idx + 1, current_sum + weights[v], new_selected, new_excluded)
            
        # Branch 2: Exclude v
        new_excluded = excluded_mask.copy()
        new_excluded[v] = True
        bb_solve(next_idx + 1, current_sum, selected_mask, new_excluded)
        
    # Initial call
    initial_selected = np.zeros(n, dtype=bool)
    initial_excluded = np.zeros(n, dtype=bool)
    
    # To improve performance, we can try to find a good initial solution using the greedy method from the seed
    # and set best_sum to that.
    
    # Greedy Init
    sel_count = 0
    greedy_mask = np.zeros(n, dtype=bool)
    greedy_sum = 0.0
    for i in order:
        if not np.any(conflict[i] & greedy_mask):
            greedy_mask[i] = True
            greedy_sum += weights[i]
            
    best_sum = greedy_sum
    best_mask = greedy_mask.copy()
    
    # Run B&B
    bb_solve(0, 0.0, initial_selected, initial_excluded)
    
    # Return best found
    result = [numbers[i] for i in np.where(best_mask)[0]]
    return result

# EVOLVE-BLOCK-END
