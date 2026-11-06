"""Rewritten program: High-Performance Greedy + Local Search for Max Weight Independent Set."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 256
    vals = list(range(2, 257))
    num_nodes = len(vals)
    
    # Precompute conflict bitmasks
    # conflict[i] has bit j set if lcm(vals[i], vals[j]) <= 256
    conflict = [0] * num_nodes
    for i in range(num_nodes):
        a = vals[i]
        for j in range(i+1, num_nodes):
            b = vals[j]
            if a * b // gcd(a, b) <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                
    # Weights: weight[i] = 1/vals[i]
    weights = [1.0 / v for v in vals]
    
    # 1. Greedy Initialization
    # Sort indices by weight descending (value ascending)
    order = sorted(range(num_nodes), key=lambda i: -weights[i])
    selected_mask = 0
    for i in order:
        if (conflict[i] & selected_mask) == 0:
            selected_mask |= (1 << i)
            
    def get_score(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    best_lb = get_score(selected_mask)
    
    # 2. Local Search (Add/Swap)
    # We perform multiple passes until no improvement or time limit.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding single vertices
        for i in range(num_nodes):
            if not (selected_mask & (1 << i)):
                if (conflict[i] & selected_mask) == 0:
                    new_mask = selected_mask | (1 << i)
                    new_score = best_lb + weights[i]
                    if new_score > best_lb + 1e-12:
                        selected_mask = new_mask
                        best_lb = new_score
                        improved = True
        
        # Try swapping one out, one in
        # To speed up, we can iterate over selected items and potential additions
        # Optimization: Only consider swaps if the gain is positive
        for i in range(num_nodes):
            if selected_mask & (1 << i):
                # Potential removal
                for j in range(num_nodes):
                    if not (selected_mask & (1 << j)):
                        # Check if j can be added if i is removed
                        # new_mask = (selected_mask & ~(1 << i)) | (1 << j)
                        # Condition: (conflict[j] & (selected_mask & ~(1 << i))) == 0
                        temp_mask = selected_mask & ~(1 << i)
                        if (conflict[j] & temp_mask) == 0:
                            new_score = best_lb - weights[i] + weights[j]
                            if new_score > best_lb + 1e-12:
                                selected_mask = temp_mask | (1 << j)
                                best_lb = new_score
                                improved = True
                                break # Restart swap loop for this i? Or just continue?
                                # Breaking inner loop is safer to re-evaluate state
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
        # Try swapping one out, two in (2-1 swap)
        # This is more expensive but can escape local minima.
        # We limit this to a few attempts per pass to ensure speed.
        if improved:
            continue # If we improved, restart the main loop to re-evaluate adds/swaps
            
        # If no simple add/swap found, try 2-1 swaps
        # This is O(N^3) worst case, so we must be careful.
        # We'll try a limited number of random or ordered 2-1 swaps.
        found_21 = False
        # Iterate over pairs of selected vertices to remove? 
        # Actually, removing 2 and adding 1 is less likely to improve weight than 1-1 if weights are positive.
        # Wait, objective is max sum. Removing 2 (small weights) and adding 1 (large weight) is bad.
        # Removing 1 (large weight) and adding 2 (medium weights) is good if sum(w2) > w1.
        # So we want to try: Remove i (selected), Add j, k (unselected).
        # Condition: j, k compatible with each other and with (selected \ {i}).
        
        # To keep it fast, we only do this if we have time.
        if time.time() < deadline - 5:
            for i in range(num_nodes):
                if not (selected_mask & (1 << i)):
                    continue
                # Candidate to remove
                # Candidates to add: j, k
                # We can optimize by precomputing compatible pairs for unselected nodes?
                # For now, simple nested loop with early exit.
                for j in range(num_nodes):
                    if selected_mask & (1 << j):
                        continue
                    # j must be compatible with selected \ {i}
                    temp_mask = selected_mask & ~(1 << i)
                    if (conflict[j] & temp_mask) != 0:
                        continue
                    # j is a candidate. Now find k compatible with j and temp_mask
                    for k in range(j+1, num_nodes):
                        if selected_mask & (1 << k):
                            continue
                        if (conflict[k] & temp_mask) != 0:
                            continue
                        if (conflict[j] & (1 << k)) != 0:
                            continue
                        # Check if j and k are compatible with each other
                        # Already checked via conflict[j] & (1<<k)
                        
                        new_score = best_lb - weights[i] + weights[j] + weights[k]
                        if new_score > best_lb + 1e-12:
                            new_mask = temp_mask | (1 << j) | (1 << k)
                            selected_mask = new_mask
                            best_lb = new_score
                            improved = True
                            found_21 = True
                            break
                    if found_21:
                        break
                if found_21:
                    break
            if found_21:
                continue # Restart main loop

    # Extract result
    result = []
    m = selected_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(vals[idx])
        m ^= lsb
        
    return result
# EVOLVE-BLOCK-END
