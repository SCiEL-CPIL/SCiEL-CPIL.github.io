"""Optimized solver for large reciprocal weight under LCM separation."""
# EVOLVE-BLOCK-START
import math
from math import gcd
import random
import time

def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a,b) > 256 for all distinct pairs,
    maximizing the sum of reciprocals.
    """
    MAX_N = 256
    N = MAX_N - 1  # Number of candidates: 2..256
    
    # Map value to index 0..254
    # val = index + 2
    
    # Precompute incompatibility matrix using bitsets for speed
    # incompat[i] is a bitmask of nodes incompatible with i
    incompat = [0] * N
    for i in range(N):
        vi = i + 2
        for j in range(i + 1, N):
            vj = j + 2
            g = gcd(vi, vj)
            if vi // g * vj <= 256:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
                
    # Objective function
    def get_score(selected_indices):
        return sum(1.0 / (i + 2) for i in selected_indices)
        
    # Check if a node can be added to the selected set (represented as bitmask)
    def can_add(node, selected_mask):
        # If node is already selected, cannot add
        if (selected_mask >> node) & 1:
            return False
        # Check if node is incompatible with any in selected_mask
        # We need to check if incompat[node] intersects with selected_mask
        if incompat[node] & selected_mask:
            return False
        return True

    # Greedy Construction
    def greedy_construct(order):
        selected = []
        selected_mask = 0
        for node in order:
            if can_add(node, selected_mask):
                selected.append(node)
                selected_mask |= (1 << node)
        return selected, selected_mask

    # Local Search with Best Improvement
    # Returns improved selected list and mask
    def local_search(selected, selected_mask):
        improved = True
        while improved:
            improved = False
            
            # 1. Try adding any compatible node
            best_add_node = -1
            best_add_gain = 0
            for node in range(N):
                if can_add(node, selected_mask):
                    gain = 1.0 / (node + 2)
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_node = node
            
            if best_add_node != -1:
                selected.append(best_add_node)
                selected_mask |= (1 << best_add_node)
                improved = True
                continue # Restart loop to check for more adds or swaps

            # 2. Try swapping one node s in selected with one node c not in selected
            best_swap_s = -1
            best_swap_c = -1
            best_swap_gain = 0
            
            for s in selected:
                # Mask of selected nodes excluding s
                others_mask = selected_mask & ~(1 << s)
                recip_s = 1.0 / (s + 2)
                
                for c in range(N):
                    if (selected_mask >> c) & 1:
                        continue
                    # Check if c is compatible with others
                    if incompat[c] & others_mask:
                        continue
                    gain = 1.0 / (c + 2) - recip_s
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_s = s
                        best_swap_c = c
                        
            if best_swap_s != -1:
                # Perform swap
                selected.remove(best_swap_s)
                selected_mask &= ~(1 << best_swap_s)
                selected.append(best_swap_c)
                selected_mask |= (1 << best_swap_c)
                improved = True
            else:
                # No improving moves found
                break
                
        return selected, selected_mask

    # Iterated Local Search
    def iterated_local_search(initial_sel, initial_mask, num_iterations=1000):
        current_sel = initial_sel[:]
        current_mask = initial_mask
        best_sel = current_sel[:]
        best_mask = current_mask
        best_score = get_score(current_sel)
        
        for _ in range(num_iterations):
            # Perturb: Randomly remove 5-10 nodes
            k = random.randint(5, 10)
            if len(current_sel) > k:
                nodes_to_remove = random.sample(current_sel, k)
                for node in nodes_to_remove:
                    current_sel.remove(node)
                    current_mask &= ~(1 << node)
            
            # Local Search
            current_sel, current_mask = local_search(current_sel, current_mask)
            current_score = get_score(current_sel)
            
            if current_score > best_score:
                best_score = current_score
                best_sel = current_sel[:]
                best_mask = current_mask
                
        return best_sel, best_mask, best_score

    # Generate initial solutions and run ILS
    random.seed(42)
    best_global_score = 0
    best_global_sel = []
    
    # Start with a few heuristic starts
    starts = []
    # Ascending
    order_asc = list(range(N))
    sel1, mask1 = greedy_construct(order_asc)
    starts.append((sel1, mask1))
    
    # Descending
    order_desc = list(range(N-1, -1, -1))
    sel2, mask2 = greedy_construct(order_desc)
    starts.append((sel2, mask2))
    
    # Random starts
    for _ in range(10):
        order_rand = list(range(N))
        random.shuffle(order_rand)
        sel_r, mask_r = greedy_construct(order_rand)
        starts.append((sel_r, mask_r))
    
    # Run ILS on each start
    for sel, mask in starts:
        # Run local search first to get a local optimum
        sel_ls, mask_ls = local_search(sel, mask)
        score_ls = get_score(sel_ls)
        
        if score_ls > best_global_score:
            best_global_score = score_ls
            best_global_sel = sel_ls[:]
            
        # Run ILS from this local optimum
        # Limit iterations based on time
        t0 = time.time()
        time_limit = 5.0 # seconds per start
        max_iter = 500
        # Adjust iterations dynamically if we want, but fixed is safer for reproducibility
        # We have ~12 starts. 12 * 5s = 60s. Too tight.
        # Let's reduce starts or iterations.
        # 5 starts * 10s = 50s.
        
        # Let's just run a fixed number of ILS iterations for all starts
        # Total budget 55s.
        # 12 starts. ~4.5s each.
        
        sel_ils, mask_ils, score_ils = iterated_local_search(sel_ls, mask_ls, num_iterations=200)
        
        if score_ils > best_global_score:
            best_global_score = score_ils
            best_global_sel = sel_ils[:]
            
        # Check time
        if time.time() - t0 > 4.5:
            # We could break early if we were looping over starts, but we are in a loop
            # Just proceed to next start
            break

    # Convert indices back to values
    result = [i + 2 for i in best_global_sel]
    
    return result

# EVOLVE-BLOCK-END
