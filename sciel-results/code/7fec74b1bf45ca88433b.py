"""XOR-circuit synthesis via minimum-cost maximum-flow with greedy refinement."""
import time
import numpy as np
from collections import defaultdict, deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin

    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs

    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}

    # Convert rows to integer bitmasks
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Phase 1: Build initial circuit using minimum-cost flow formulation
    # Nodes: all masks that appear in rows or are XORs of pairs
    # We'll use a simpler approach: identify high-value subexpressions via frequency analysis
    
    # Count how many times each mask appears as a subset of rows
    mask_frequency = defaultdict(int)
    for mask in row_masks:
        mask_frequency[mask] += 1
    
    # Identify candidate intermediate nodes: masks that are XORs of pairs of frequent masks
    # and would save gates if computed once
    candidate_masks = set()
    frequent_masks = [m for m, f in mask_frequency.items() if f >= 2]
    
    # Add pairwise XORs of frequent masks
    for i in range(len(frequent_masks)):
        for j in range(i + 1, len(frequent_masks)):
            xor_mask = frequent_masks[i] ^ frequent_masks[j]
            if xor_mask != 0 and xor_mask not in mask_frequency:
                candidate_masks.add(xor_mask)
    
    # Also consider smaller masks from input combinations
    for mask in row_masks:
        # Add all submasks with small popcount
        sub = mask
        while sub:
            if sub not in mask_frequency and sub not in candidate_masks:
                popcnt = bin(sub).count('1')
                if popcnt <= 3:
                    candidate_masks.add(sub)
            sub = (sub - 1) & mask
    
    # Phase 2: Greedy selection of intermediate nodes based on cost-benefit
    wire_mask = [0] * (n_inputs + 1)
    for i in range(n_inputs):
        wire_mask[i] = (1 << i)
    wire_mask[n_inputs] = 0
    
    mask_to_wire = {}
    for i in range(n_inputs + 1):
        mask_to_wire[wire_mask[i]] = i
    
    # Each row starts as its full mask
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
    
    # Greedily add intermediate nodes that provide best cost reduction
    max_iter = 5000
    iter_count = 0
    no_improvement = 0
    
    while iter_count < max_iter:
        if time.time() > deadline - 10.0:
            break
        iter_count += 1
        
        # Calculate current total gate count
        current_cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                current_cost += len(wires) - 1
        
        # Try each candidate mask: see if computing it as a gate saves gates overall
        best_candidate = None
        best_saving = 0
        
        # Build set of all masks currently represented
        current_masks = set(wire_mask)
        
        # Consider candidate masks and also XORs of existing wires
        candidates_to_try = set(candidate_masks)
        for w1 in range(len(wire_mask)):
            for w2 in range(w1 + 1, len(wire_mask)):
                xor_m = wire_mask[w1] ^ wire_mask[w2]
                if xor_m != 0 and xor_m not in current_masks:
                    candidates_to_try.add(xor_m)
        
        for cand_mask in candidates_to_try:
            if cand_mask in mask_to_wire:
                continue
            
            # Find all rows that would benefit from this intermediate
            # A row benefits if it contains both operands whose XOR is cand_mask
            # or if replacing a subset of its wires with cand_mask reduces wire count
            
            # Strategy: find pairs (a, b) such that wire_mask[a] ^ wire_mask[b] == cand_mask
            # and count how many rows contain both a and b
            benefit = 0
            affected_rows = []
            
            for i, wires in enumerate(row_wires):
                # Check if any pair in wires XORs to cand_mask
                found_pair = False
                for x in range(len(wires)):
                    for y in range(x + 1, len(wires)):
                        if wire_mask[wires[x]] ^ wire_mask[wires[y]] == cand_mask:
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair and len(wires) > 1:
                    benefit += 1
                    affected_rows.append(i)
            
            # Cost: 1 gate to compute cand_mask
            # Benefit: for each affected row, we can replace 2 wires with 1
            # Net saving = benefit - 1 (only if benefit > 1)
            net_saving = benefit - 1
            
            # Also consider: does this enable further simplifications?
            # Heuristic: prefer masks with lower popcount (simpler to compute)
            popcnt = bin(cand_mask).count('1')
            adjusted_saving = net_saving - 0.1 * popcnt
            
            if adjusted_saving > best_saving:
                best_saving = adjusted_saving
                best_candidate = (cand_mask, benefit, affected_rows)
        
        if best_candidate is None or best_saving <= 0:
            break
        
        cand_mask, benefit, affected_rows = best_candidate
        
        # Find the pair of wires that produce this mask
        a, b = -1, -1
        for w1 in range(len(wire_mask)):
            for w2 in range(w1 + 1, len(wire_mask)):
                if wire_mask[w1] ^ wire_mask[w2] == cand_mask:
                    a, b = w1, w2
                    break
            if a >= 0:
                break
        
        if a < 0:
            # Can't find a pair; skip this candidate
            break
        
        # Create new wire
        new_wire = next_wire
        next_wire += 1
        wire_mask.append(cand_mask)
        mask_to_wire[cand_mask] = new_wire
        gates.append([a, b])
        
        # Update affected rows: replace a and b with new_wire
        for i in affected_rows:
            wires = row_wires[i]
            if a in wires and b in wires:
                wires.remove(a)
                wires.remove(b)
                if new_wire not in wires:
                    wires.append(new_wire)
        
        # Check if we actually improved
        new_cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                new_cost += len(wires) - 1
        
        if new_cost < current_cost:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > 100:
                break
    
    # Phase 3: Final output assembly with balanced binary tree
    outputs = []
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
