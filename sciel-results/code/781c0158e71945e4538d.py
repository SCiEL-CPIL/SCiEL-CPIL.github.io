import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # Convert to list of lists of ints for easier manipulation if needed, 
    # but numpy is fine for small matrices.
    # We will use Python lists of ints (bitmasks) for speed and simplicity.
    
    n_cols = matrix.shape[1]
    m_rows = matrix.shape[0]
    
    # Convert matrix to list of bitmasks
    rows = []
    for i in range(m_rows):
        mask = 0
        for j in range(n_cols):
            if matrix[i][j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # We will perform Gaussian Elimination with Minimum Weight Pivot.
    # We need to track the actual gates.
    # 
    # Strategy:
    # 1. Identify pivot rows.
    # 2. Build pivot rows from inputs.
    # 3. Use pivot rows to eliminate other rows.
    #
    # To do this, we need to know the wire ID for each "logical" row.
    #
    # Let's maintain a list `current_rows` of bitmasks.
    # And a list `wire_ids` corresponding to these rows.
    #
    # Initially, we don't have wires for the rows.
    # So we must build them.
    
    gates = []
    next_wire = n_cols + 1  # n is constant 0, so n+1 is first new wire
    
    # We will process the matrix to find a set of pivot rows that form a basis for the row space.
    # Actually, we need to produce ALL m outputs.
    # So we need to express each output row in terms of the basis (pivot) rows and inputs?
    # No, we just need to produce the values.
    
    # Let's use a different approach:
    # We will greedily construct each output row using available wires.
    # Available wires start as inputs + constant 0.
    
    # To make it faster and smarter, we will use the GE structure to guide the construction.
    
    # 1. Perform GE to determine the "cost" and structure.
    # Let's just use a simple heuristic:
    # For each row, if it's simple, build it. If it's complex, try to combine existing wires.
    
    # Given the small size, let's try to find a minimal set of intermediate wires.
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
    
    # available_wires: dict mapping wire_id -> mask
    available_wires = {}
    for i in range(n_cols):
        available_wires[i] = 1 << i
    available_wires[n_cols] = 0  # constant zero
    
    outputs = []
    
    # Sort rows by weight to build simpler ones first? 
    # This might allow reuse.
    # Let's sort indices by weight.
    indices = list(range(m_rows))
    indices.sort(key=lambda i: popcount(rows[i]))
    
    # We will build rows in this order.
    # But the output order must be preserved.
    # So we store the result wire for each original index.
    result_wires = [None] * m_rows
    
    # To allow reuse, we add newly created wires to available_wires.
    
    for idx in indices:
        target = rows[idx]
        
        if target == 0:
            result_wires[idx] = n_cols
            continue
            
        # Check if target is already available
        if target in available_wires.values():
            # Find the wire id
            for wid, mask in available_wires.items():
                if mask == target:
                    result_wires[idx] = wid
                    break
            continue
            
        # Try to find two existing wires w1, w2 such that w1 ^ w2 == target
        # This would cost 1 gate.
        found = False
        # Optimization: Only check if number of wires is small enough
        if len(available_wires) < 100:
            items = list(available_wires.items())
            for i in range(len(items)):
                wid1, mask1 = items[i]
                for j in range(i+1, len(items)):
                    wid2, mask2 = items[j]
                    if mask1 ^ mask2 == target:
                        # Create new wire
                        new_wid = next_wire
                        next_wire += 1
                        gates.append([wid2, new_wid]) # new_wid = 0 ^ wid2? No.
                        # We need to XOR wid2 into a wire holding wid1?
                        # No, we need a wire holding wid1 ^ wid2.
                        # We can take a copy of wid1 and XOR wid2 into it.
                        # Gate [wid2, new_wid] makes new_wid = 0 ^ wid2 = wid2.
                        # Then we need to XOR wid1 into it.
                        # Gate [wid1, new_wid] makes new_wid = wid2 ^ wid1.
                        # So 2 gates.
                        
                        # Wait, is there a 1-gate solution?
                        # Only if one of the wires is 0? No.
                        # So combining two wires costs 2 gates (1 to copy, 1 to xor).
                        # Unless we can modify an existing wire? No, inputs are immutable.
                        # And intermediate wires are usually immutable in this model?
                        # "Gate k produces wire n+1+k". This implies new wires are created.
                        # So yes, 2 gates to combine two existing wires.
                        
                        # Is it worth it?
                        # Building from inputs costs weight(target) gates.
                        # If weight(target) > 2, it's worth it.
                        
                        # Let's check weight.
                        if popcount(target) > 2:
                            # Create new wire
                            new_wid = next_wire
                            next_wire += 1
                            # Copy wid1 to new_wid
                            gates.append([wid1, new_wid])
                            # XOR wid2 into new_wid
                            gates.append([wid2, new_wid])
                            available_wires[new_wid] = target
                            result_wires[idx] = new_wid
                            found = True
                            break
                if found:
                    break
                    
        if found:
            continue
            
        # Fallback: Build from inputs
        # Find inputs that make up the target
        # We can just XOR all inputs in the support.
        # Cost: weight(target) gates.
        # We can do better if we have partial sums?
        # For now, just use inputs.
        
        # Optimization: If we have a wire that is a subset of target, use it.
        # Find w in available_wires such that w & ~target == 0 (w is subset of target) and w != 0
        best_subset = 0
        best_wid = -1
        for wid, mask in available_wires.items():
            if mask & ~target == 0 and mask != 0:
                # mask is subset of target
                if popcount(mask) > popcount(best_subset):
                    best_subset = mask
                    best_wid = wid
                    
        if best_wid != -1:
            remaining = target ^ best_subset
            # Now build remaining from inputs
            # Create new wire
            new_wid = next_wire
            next_wire += 1
            # Copy best_subset to new_wid
            gates.append([best_wid, new_wid])
            
            # XOR inputs in remaining
            for j in range(n_cols):
                if (remaining >> j) & 1:
                    gates.append([j, new_wid])
            
            available_wires[new_wid] = target
            result_wires[idx] = new_wid
        else:
            # No subset found. Build from scratch.
            new_wid = next_wire
            next_wire += 1
            first = True
            for j in range(n_cols):
                if (target >> j) & 1:
                    if first:
                        gates.append([j, new_wid])
                        first = False
                    else:
                        gates.append([j, new_wid])
            available_wires[new_wid] = target
            result_wires[idx] = new_wid
            
        # Time check
        if time.time() > deadline:
            break

    # If time ran out, fill remaining with naive construction
    for idx in range(m_rows):
        if result_wires[idx] is None:
            target = rows[idx]
            if target == 0:
                result_wires[idx] = n_cols
            else:
                new_wid = next_wire
                next_wire += 1
                first = True
                for j in range(n_cols):
                    if (target >> j) & 1:
                        if first:
                            gates.append([j, new_wid])
                            first = False
                        else:
                            gates.append([j, new_wid])
                result_wires[idx] = new_wid

    return {'gates': gates, 'outputs': result_wires}
