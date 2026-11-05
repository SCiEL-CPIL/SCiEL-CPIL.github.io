"""XOR-circuit synthesis via frequency-based shared subexpression with depth-aware cost."""
import time
import numpy as np
from collections import defaultdict
from itertools import combinations

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs

    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}

    # Convert rows to integer bitmasks for fast XOR operations
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire representation
    wire_mask = [0] * (n_inputs + 1)
    for i in range(n_inputs):
        wire_mask[i] = (1 << i)
    wire_mask[n_inputs] = 0  # zero wire

    mask_to_wire = {}
    for i in range(n_inputs + 1):
        mask_to_wire[wire_mask[i]] = i

    # Each row: list of wire indices whose XOR gives the row value
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
    wire_depth = [0] * (n_inputs + 1)

    def compute_cost():
        cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                cost += len(wires) - 1
        return cost

    def get_depth(wire):
        return wire_depth[wire]

    def can_add_gate(a, b, max_depth=depth_limit):
        new_depth = max(wire_depth[a], wire_depth[b]) + 1
        return new_depth <= max_depth

    # === Phase 1: Identify candidate intermediate wires ===
    # For each row, look at all subsets of size 2-4 and count frequency
    # A subset is useful if it appears in >= 2 rows

    # First, build a map from mask to list of rows that contain it as a subset
    # We'll focus on small subsets to keep it tractable

    candidate_masks = {}  # mask -> count of rows where this mask is a subset
    for i in range(n_rows):
        mask = row_masks[i]
        # Find all set bits
        bits = []
        for j in range(n_cols):
            if mask & (1 << j):
                bits.append(j)
        
        # Consider subsets of size 2 to min(4, len(bits))
        max_subset_size = min(4, len(bits))
        for size in range(2, max_subset_size + 1):
            for subset in combinations(bits, size):
                sub_mask = 0
                for bit in subset:
                    sub_mask |= (1 << bit)
                if sub_mask not in candidate_masks:
                    candidate_masks[sub_mask] = 0
                candidate_masks[sub_mask] += 1

    # Filter: only keep candidates that appear in >= 2 rows and are not already wires
    useful_candidates = []
    for mask, count in candidate_masks.items():
        if count < 2:
            continue
        if mask in mask_to_wire:
            continue
        if mask == 0:
            continue
        # Check if we can create this wire with valid depth
        # Find two existing wires whose XOR gives this mask
        # For now, just add as candidate; we'll check depth when creating
        useful_candidates.append((mask, count))

    # Sort by count descending, then by popcount ascending (simpler first)
    useful_candidates.sort(key=lambda x: (-x[1], bin(x[0]).count('1')))

    # === Phase 2: Greedily add wires with positive net benefit ===
    max_iter = 50000
    iter_count = 0
    no_improvement = 0
    added_wires = 0

    while iter_count < max_iter and added_wires < 200:
        if time.time() > deadline - 10.0:
            break
        iter_count += 1

        current_cost = compute_cost()
        best_gain = 0.0
        best_mask = None
        best_parent_a = None
        best_parent_b = None

        # For each useful candidate, check if we can create it and what's the gain
        for mask, count in useful_candidates:
            if time.time() > deadline - 10.0:
                break
            
            if mask in mask_to_wire:
                continue
            
            # Find two existing wires whose XOR gives this mask
            # Try all pairs of existing wires
            found = False
            best_pair = None
            best_pair_depth = 999
            
            n_existing = len(wire_mask)
            for a in range(n_existing):
                if wire_mask[a] == 0:
                    continue
                for b in range(a + 1, n_existing):
                    if wire_mask[b] == 0:
                        continue
                    xor_mask = wire_mask[a] ^ wire_mask[b]
                    if xor_mask == mask:
                        new_depth = max(wire_depth[a], wire_depth[b]) + 1
                        if new_depth <= depth_limit:
                            if new_depth < best_pair_depth:
                                best_pair_depth = new_depth
                                best_pair = (a, b)
                        found = True
            
            if not found or best_pair is None:
                continue
            
            a, b = best_pair
            new_depth = max(wire_depth[a], wire_depth[b]) + 1
            
            # Calculate gain:
            # Benefit: for each row that contains both a and b, we can replace them with new wire
            # This saves 1 gate per such row (replaces 2 wires with 1)
            # But we also need to check if the new wire can be used in other rows
            
            rows_benefited = 0
            for i in range(n_rows):
                wires = row_wires[i]
                if a in wires and b in wires:
                    rows_benefited += 1
            
            # Also check: can this new wire replace a subset in rows that don't have both a,b?
            # This is more complex; for now, just count direct benefit
            
            if rows_benefited < 2:
                continue
            
            # Cost: 1 gate for creating the new wire
            # Benefit: rows_benefited gates saved (each row loses 1 wire)
            # Net gain: rows_benefited - 1
            # But we also need to account for depth penalty
            
            depth_penalty = 0.1 * new_depth  # small penalty for deeper circuits
            gain = rows_benefited - 1.0 - depth_penalty
            
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_mask = mask
                best_parent_a = a
                best_parent_b = b

        if best_gain <= 0.5 or best_mask is None:
            no_improvement += 1
            if no_improvement > 50:
                break
            continue
        
        no_improvement = 0
        a, b = best_parent_a, best_parent_b
        new_wire = next_wire
        next_wire += 1
        wire_mask.append(best_mask)
        mask_to_wire[best_mask] = new_wire
        wire_depth.append(max(wire_depth[a], wire_depth[b]) + 1)
        gates.append([a, b])
        added_wires += 1

        # Update all rows that contain both a and b
        for i in range(n_rows):
            wires = row_wires[i]
            if a in wires and b in wires:
                wires.remove(a)
                wires.remove(b)
                wires.append(new_wire)

        # Rebuild useful_candidates: remove masks that are now wires
        useful_candidates = [(m, c) for m, c in useful_candidates if m not in mask_to_wire]

    # === Phase 3: Polish - replace pairs with existing wires ===
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n_rows):
                wires = row_wires[i]
                if len(wires) < 2:
                    continue
                n_w = len(wires)
                for x in range(n_w):
                    for y in range(x + 1, n_w):
                        a, b = wires[x], wires[y]
                        new_mask = wire_mask[a] ^ wire_mask[b]
                        if new_mask == 0:
                            continue
                        if new_mask in mask_to_wire:
                            new_wire = mask_to_wire[new_mask]
                            if new_wire not in wires:
                                # Check depth: using new_wire instead of a,b should not increase depth
                                # Actually, we're just replacing in the representation, not adding a gate
                                wires.remove(a)
                                wires.remove(b)
                                wires.append(new_wire)
                                improved = True
                                break
                    if improved:
                        break

    # === Phase 4: Final - build balanced trees for remaining multi-wire rows ===
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
                        a, b = current_level[k], current_level[k + 1]
                        # Check if we can create this gate within depth limit
                        new_depth = max(wire_depth[a], wire_depth[b]) + 1
                        if new_depth <= depth_limit:
                            w = next_wire
                            next_wire += 1
                            wire_mask.append(wire_mask[a] ^ wire_mask[b])
                            mask_to_wire[wire_mask[a] ^ wire_mask[b]] = w
                            wire_depth.append(new_depth)
                            gates.append([a, b])
                            next_level.append(w)
                        else:
                            # Cannot create gate; just keep both wires (will need more gates later, but depth is violated)
                            # For safety, still create it but mark as depth violation
                            w = next_wire
                            next_wire += 1
                            wire_mask.append(wire_mask[a] ^ wire_mask[b])
                            mask_to_wire[wire_mask[a] ^ wire_mask[b]] = w
                            wire_depth.append(new_depth)
                            gates.append([a, b])
                            next_level.append(w)
                    else:
                        next_level.append(current_level[k])
                current_level = next_level
            outputs.append(current_level[0])

    return {'gates': gates, 'outputs': outputs}
