"""XOR-circuit synthesis via greedy shared-subexpression with balanced final tree."""
import time
import numpy as np
from collections import defaultdict

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

    # Convert rows to integer bitmasks for fast XOR operations
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire representation: wire_mask[w] = bitmask of inputs that wire w represents
    wire_mask = [0] * (n_inputs + 1)
    for i in range(n_inputs):
        wire_mask[i] = (1 << i)
    wire_mask[n_inputs] = 0  # zero wire

    mask_to_wire = {}
    for i in range(n_inputs + 1):
        mask_to_wire[wire_mask[i]] = i

    # Each row is a list of wire indices whose XOR gives the row value
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

    max_iter = 20000
    iter_count = 0
    no_improvement = 0

    while iter_count < max_iter:
        if time.time() > deadline - 5.0:
            break
        iter_count += 1

        # Current gate count (excluding final output assembly)
        current_cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                current_cost += len(wires) - 1

        # Find co-occurring pairs
        pair_rows = defaultdict(list)
        for i, wires in enumerate(row_wires):
            n_w = len(wires)
            if n_w < 2:
                continue
            for x in range(n_w):
                for y in range(x + 1, n_w):
                    a, b = wires[x], wires[y]
                    if a > b:
                        a, b = b, a
                    pair_rows[(a, b)].append(i)

        if not pair_rows:
            break

        best_score = -1e9
        best_pair = None
        best_new_mask = None
        best_rows = None
        best_existing = -1

        for (a, b), rows in pair_rows.items():
            if len(rows) < 2:
                continue

            new_mask = wire_mask[a] ^ wire_mask[b]
            if new_mask == 0:
                continue

            benefit = sum(1 for i in rows if len(row_wires[i]) > 1)

            if new_mask in mask_to_wire:
                score = benefit
                existing = mask_to_wire[new_mask]
            else:
                popcnt = bin(new_mask).count('1')
                score = benefit - 1 - 0.05 * popcnt
                existing = -1

            if score > best_score:
                best_score = score
                best_pair = (a, b)
                best_new_mask = new_mask
                best_rows = rows
                best_existing = existing

        if best_pair is None or best_score <= 0:
            break

        a, b = best_pair

        if best_existing >= 0:
            new_wire = best_existing
        else:
            new_wire = next_wire
            next_wire += 1
            wire_mask.append(best_new_mask)
            mask_to_wire[best_new_mask] = new_wire
            gates.append([a, b])

        # Update affected rows
        for i in best_rows:
            wires = row_wires[i]
            if a in wires and b in wires:
                wires.remove(a)
                wires.remove(b)
                if new_wire in wires:
                    wires.remove(new_wire)
                else:
                    wires.append(new_wire)

        # Check improvement
        new_cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                new_cost += len(wires) - 1

        if new_cost < current_cost:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > 200:
                break

    # Final output assembly with balanced binary tree to control depth
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
