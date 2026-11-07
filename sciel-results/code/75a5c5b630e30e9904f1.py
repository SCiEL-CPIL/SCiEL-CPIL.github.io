"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

P = {'n': 256}

def construct():
    n = P['n']
    # Vertices are 2..n
    # 1 is always included
    base_sum = 1
    vertices = list(range(2, n + 1))
    num_vertices = len(vertices)
    
    # Precompute prime factors or just use gcd for conflict
    # Conflict matrix as bitmasks
    # index i corresponds to vertices[i]
    conflict = [0] * num_vertices
    for i in range(num_vertices):
        vi = vertices[i]
        mask = 0
        for j in range(num_vertices):
            if i != j:
                vj = vertices[j]
                if math.gcd(vi, vj) > 1:
                    mask |= (1 << j)
        conflict[i] = mask

    weights = vertices[:]  # weights are the values themselves

    # Sort vertices by weight descending to improve pruning
    # We need to remap indices if we sort
    # Let's create a list of (weight, index) and sort
    indexed_weights = [(w, i) for i, w in enumerate(weights)]
    indexed_weights.sort(reverse=True)
    
    # New order
    new_weights = [w for w, i in indexed_weights]
    old_to_new = [0] * num_vertices
    for new_idx, (w, old_idx) in enumerate(indexed_weights):
        old_to_new[old_idx] = new_idx
    
    # Remap conflict masks
    new_conflict = [0] * num_vertices
    for old_idx in range(num_vertices):
        new_idx = old_to_new[old_idx]
        mask = conflict[old_idx]
        new_mask = 0
        # Iterate through set bits in old mask
        m = mask
        while m:
            lsb = m & -m
            old_j = lsb.bit_length() - 1
            new_j = old_to_new[old_j]
            new_mask |= (1 << new_j)
            m ^= lsb
        new_conflict[new_idx] = new_mask

    # Initial greedy solution for lower bound
    selected_mask = 0
    greedy_sum = 0
    available = (1 << num_vertices) - 1
    for i in range(num_vertices):
        if (available >> i) & 1:
            # Check if i is compatible with all selected
            # Since we process in order, we just check against selected
            # Actually, greedy: if i is in available, it means it doesn't conflict with previously selected?
            # No, we need to maintain available set.
            pass
    
    # Proper Greedy
    available = (1 << num_vertices) - 1
    greedy_sum = 0
    for i in range(num_vertices):
        if (available >> i) & 1:
            greedy_sum += new_weights[i]
            # Remove i and its neighbors
            available &= ~(new_conflict[i] | (1 << i))

    best_sum = greedy_sum

    # Branch and Bound
    # To avoid recursion depth issues, we use a stack or recursion with sys.setrecursionlimit
    # But 255 depth is fine.
    
    def solve(available_mask, current_sum):
        nonlocal best_sum
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
            return
        
        # Upper bound: current_sum + sum of weights of available vertices
        # Calculate sum of available weights
        # This is O(N) per call.
        avail_sum = 0
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            avail_sum += new_weights[idx]
            m ^= lsb
            
        if current_sum + avail_sum <= best_sum:
            return

        # Pick a vertex to branch on
        # Choose the one with highest weight in available_mask
        # Or just the first one?
        # Highest weight usually prunes better.
        m = available_mask
        best_idx = -1
        best_w = -1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if new_weights[idx] > best_w:
                best_w = new_weights[idx]
                best_idx = idx
            m ^= lsb
            
        if best_idx == -1:
            return

        # Branch 1: Include best_idx
        new_avail = available_mask & ~(new_conflict[best_idx] | (1 << best_idx))
        solve(new_avail, current_sum + new_weights[best_idx])
        
        # Branch 2: Exclude best_idx
        new_avail = available_mask & ~(1 << best_idx)
        solve(new_avail, current_sum)

    initial_mask = (1 << num_vertices) - 1
    solve(initial_mask, 0)

    return [1] + [v for v in vertices]  # Wait, we need the actual values, not just sum.
    # The task asks for a list of ints.
    # We need to recover the set.
    # The B&B above only tracks the sum.
    # We need to track the selected vertices.
    
    # Re-implement to track selection or just return the sum?
    # "Return a list of ints. The raw value is their sum."
    # So we must return the list.
    
    # Let's modify the solver to store the best set.
    
    best_set_mask = 0
    
    def solve2(available_mask, current_sum, current_mask):
        nonlocal best_sum, best_set_mask
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set_mask = current_mask
            return
        
        # Upper bound
        avail_sum = 0
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            avail_sum += new_weights[idx]
            m ^= lsb
            
        if current_sum + avail_sum <= best_sum:
            return

        # Pick vertex
        m = available_mask
        best_idx = -1
        best_w = -1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if new_weights[idx] > best_w:
                best_w = new_weights[idx]
                best_idx = idx
            m ^= lsb
            
        if best_idx == -1:
            return

        # Include
        new_avail = available_mask & ~(new_conflict[best_idx] | (1 << best_idx))
        solve2(new_avail, current_sum + new_weights[best_idx], current_mask | (1 << best_idx))
        
        # Exclude
        new_avail = available_mask & ~(1 << best_idx)
        solve2(new_avail, current_sum, current_mask)

    # Reset best
    best_sum = greedy_sum
    # Recover greedy mask
    available = (1 << num_vertices) - 1
    greedy_mask = 0
    for i in range(num_vertices):
        if (available >> i) & 1:
            greedy_mask |= (1 << i)
            available &= ~(new_conflict[i] | (1 << i))
    best_set_mask = greedy_mask

    solve2(initial_mask, 0, 0)

    # Construct result list
    result = [1]
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(new_weights[idx])
        m ^= lsb
        
    return result

# EVOLVE-BLOCK-END
