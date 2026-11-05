import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs

    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}

    # Convert rows to integer bitmasks for fast XOR operations
    # This is much faster than list-based representations for checking co-occurrence
    row_masks = []
    for i in range(n_rows):
        mask = 0
        # Optimization: Use numpy bit operations if possible, but Python int is fine for small n_cols
        # For n_cols up to 64, Python int is very fast.
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire representation: wire_mask[w] = bitmask of inputs that wire w represents
    # Initial wires: 0..n_inputs-1 are inputs, n_inputs is zero
    wire_mask = [0] * (n_inputs + 1)
    for i in range(n_inputs):
        wire_mask[i] = (1 << i)
    wire_mask[n_inputs] = 0  # zero wire

    # Map from mask to wire index for quick lookup and merging
    mask_to_wire = {}
    for i in range(n_inputs + 1):
        mask_to_wire[wire_mask[i]] = i

    # Each row is a set of wire indices whose XOR gives the row value
    # We use lists for easy removal, but we'll manage them carefully.
    # To speed up pair counting, we can maintain a global count of pairs.
    
    # State:
    # row_wires[i] is a list of wire indices
    row_wires = []
    for i in range(n_rows):
        wires = []
        mask = row_masks[i]
        for j in range(n_cols):
            if mask & (1 << j):
                wires.append(j)
        row_wires.append(wires)
        
    gates = []
    next_wire = n_inputs + 1
    
    # To speed up the search for the best pair, we maintain a dictionary of pair counts.
    # pair_counts[(a, b)] = number of rows where both a and b are present.
    # Note: We only care about pairs where a != b.
    pair_counts = defaultdict(int)
    
    # Initialize pair counts
    for wires in row_wires:
        n_w = len(wires)
        if n_w < 2:
            continue
        # Use a set to avoid double counting if duplicates exist (they shouldn't in a valid XOR sum)
        seen = set()
        for x in range(n_w):
            for y in range(x + 1, n_w):
                a, b = wires[x], wires[y]
                if a > b:
                    a, b = b, a
                pair = (a, b)
                if pair not in seen:
                    pair_counts[pair] += 1
                    seen.add(pair)

    max_iter = 5000 # Limit iterations to prevent infinite loops or excessive time
    iter_count = 0
    no_improvement = 0
    
    # Baseline cost calculation for comparison
    def calculate_cost():
        cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                cost += len(wires) - 1
        return cost

    initial_cost = calculate_cost()
    
    while iter_count < max_iter:
        if time.time() > deadline - 5.0:
            break
        iter_count += 1

        if not pair_counts:
            break

        best_score = -1e9
        best_pair = None
        best_new_mask = None
        best_existing = -1
        
        # Heuristic: Only consider pairs that appear in at least 2 rows
        # This is a strong filter that reduces the search space significantly.
        # We iterate through the pair_counts dict.
        # To be efficient, we could sort, but dict iteration is O(N).
        # For 16x16, number of wires is small, so pair count is manageable.
        
        for (a, b), count in pair_counts.items():
            if count < 2:
                continue
            
            new_mask = wire_mask[a] ^ wire_mask[b]
            if new_mask == 0:
                continue
            
            # Benefit:
            # 1. If we merge a and b in a row, we reduce the number of terms in that row by 1.
            #    Since this pair appears in 'count' rows, we save 'count' gates in the final assembly.
            # 2. If the new wire is NEW, we add 1 gate. If it EXISTS, we add 0 gates.
            
            if new_mask in mask_to_wire:
                # Reusing an existing wire saves the creation gate
                score = count
                existing = mask_to_wire[new_mask]
            else:
                # Creating a new wire costs 1 gate, but saves 'count' gates in assembly
                # Net change: count - 1
                # We also penalize high popcount slightly to prefer simpler wires? 
                # Actually, popcount doesn't directly affect gate count in this model, 
                # but simpler wires might be reused more later.
                score = count - 1
                existing = -1

            if score > best_score:
                best_score = score
                best_pair = (a, b)
                best_new_mask = new_mask
                best_existing = existing

        if best_pair is None or best_score <= 0:
            break

        a, b = best_pair
        
        # Determine the new wire
        if best_existing >= 0:
            new_wire = best_existing
        else:
            new_wire = next_wire
            next_wire += 1
            wire_mask.append(best_new_mask)
            mask_to_wire[best_new_mask] = new_wire
            gates.append([a, b])

        # Update pair counts and row_wires
        # This is the expensive part. We need to update all rows containing a or b.
        
        # Identify affected rows
        affected_rows = []
        for i in range(n_rows):
            if a in row_wires[i] or b in row_wires[i]:
                affected_rows.append(i)
                
        # We need to decrement counts for pairs involving a or b that are being removed/changed.
        # And increment counts for new pairs involving new_wire.
        
        # To avoid complex bookkeeping errors, we can just rebuild the pair counts for affected rows?
        # No, that's O(N^2). 
        # Better: 
        # 1. For each affected row, remove a and b from the row.
        # 2. Add new_wire to the row.
        # 3. Update pair_counts globally.
        
        # Let's track which pairs are invalidated.
        # Pairs involving (a, x) and (b, x) for any x in the row will change.
        
        # Simpler approach for correctness and speed on small N:
        # Just recalculate pair_counts for the affected rows?
        # If N is 16x16, max wires per row is 16. Pairs per row ~ 120.
        # Total pairs ~ 16*120 = 1920.
        # Rebuilding pair_counts from scratch every iteration is O(N * W^2).
        # With 5000 iterations, 5000 * 2000 = 10^7 operations. This is acceptable in Python.
        
        # Let's try the "Rebuild Pair Counts" strategy for simplicity and robustness.
        # It avoids subtle bugs in incremental updates.
        
        # 1. Update row_wires
        for i in affected_rows:
            wires = row_wires[i]
            # Remove a and b
            if a in wires:
                wires.remove(a)
            if b in wires:
                wires.remove(b)
            # Add new_wire
            # Check if new_wire is already in wires (it shouldn't be if a!=b and new=a^b, 
            # unless a or b was 0? No, 0 is handled separately. 
            # If a^b == a, then b=0. If a^b == b, then a=0.
            # If a or b is 0, we shouldn't have formed a pair with 0 in the first place 
            # because 0 doesn't contribute to XOR sum effectively (x^0=x).
            # Our initialization puts 0 in wire_mask but row_wires only contains inputs (1s).
            # So a and b are never 0 in the pairs we consider.
            # However, new_wire could be 0 if a==b, but we ensure a!=b.
            # So new_wire is non-zero.
            wires.append(new_wire)
            # Sort to maintain canonical order? Not strictly necessary for correctness, 
            # but good for pair counting consistency.
            wires.sort()

        # 2. Rebuild pair_counts
        pair_counts.clear()
        for wires in row_wires:
            n_w = len(wires)
            if n_w < 2:
                continue
            seen = set()
            for x in range(n_w):
                for y in range(x + 1, n_w):
                    a_r, b_r = wires[x], wires[y]
                    if a_r > b_r:
                        a_r, b_r = b_r, a_r
                    pair = (a_r, b_r)
                    if pair not in seen:
                        pair_counts[pair] += 1
                        seen.add(pair)
                        
        # Check improvement
        new_cost = calculate_cost()
        if new_cost < initial_cost: # Compare against the very first cost? Or previous?
             # We want to stop if we stop improving relative to the *best* found so far.
             pass
             
        # Let's track global best cost
        if not hasattr(synthesize, '_best_cost'):
            synthesize._best_cost = initial_cost
            
        if new_cost < synthesize._best_cost:
            synthesize._best_cost = new_cost
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > 50:
                break

    # Reset the helper attribute to avoid state leakage between calls if any
    if hasattr(synthesize, '_best_cost'):
        del synthesize._best_cost

    # Final output assembly with balanced binary tree to control depth
    outputs = []
    
    # We need to ensure depth <= 16.
    # The greedy phase creates gates. The final assembly creates more.
    # We should check the depth of the circuit generated so far.
    # For now, we assume the greedy phase keeps the graph shallow.
    # The final assembly adds log2(k) depth for a row with k wires.
    # If k is small (e.g., < 16), log2(k) < 4.
    # Total depth = max_depth_greedy + log2(max_wires_in_row).
    
    for i in range(n_rows):
        wires = row_wires[i]
        if len(wires) == 0:
            outputs.append(zero_wire)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            # Build balanced XOR tree
            current_level = list(wires)
            while len(current_level) > 1:
                next_level = []
                for k in range(0, len(current_level), 2):
                    if k + 1 < len(current_level):
                        w = next_wire
                        next_wire += 1
                        gates.append([current_level[k], current_level[k + 1]])
                        next_level.append(w)
                    else:
                        next_level.append(current_level[k])
                current_level = next_level
            outputs.append(current_level[0])

    return {'gates': gates, 'outputs': outputs}
