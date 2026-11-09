"""XOR-circuit synthesis: minimum-cost basis completion with shared sub-expressions."""
import time
import numpy as np
from collections import defaultdict
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 80  # safety margin before 90s kill

    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n  # constant-zero wire

    # Convert rows to bitmasks (Python ints for GF(2) operations)
    row_masks = []
    for i in range(n_rows):
        mask = 0
        row = mat[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        row_masks.append(mask)

    # Gate management
    gates = []
    next_wire = n + 1
    # Wire -> mask (what XOR of inputs this wire represents)
    wire_mask = [0] * (n + 1)  # wires 0..n are inputs + zero
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0  # zero wire

    # Mask -> wire (cache of already-computed wires)
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[(1 << i)] = i
    mask_to_wire[0] = zero_wire

    def get_wire(mask):
        """Get wire index for given XOR mask, creating gates as needed."""
        nonlocal next_wire
        if mask in mask_to_wire:
            return mask_to_wire[mask]

        # Find best split: minimize number of new gates needed
        # Try all possible splits into two sub-masks
        # For efficiency, try splitting by removing one bit at a time
        # and also try balanced splits

        popcount = bin(mask).count('1')
        if popcount == 1:
            # Should have been caught above
            idx = mask.bit_length() - 1
            mask_to_wire[mask] = idx
            return idx

        best_m1 = None
        best_m2 = None
        best_cost = float('inf')

        # Strategy 1: Try removing each bit and see if the complement is cached
        for bit in range(n):
            if mask & (1 << bit):
                m1 = mask ^ (1 << bit)
                m2 = (1 << bit)
                c1 = m1 in mask_to_wire
                c2 = m2 in mask_to_wire  # always true for single bit
                cost = 0 if c1 else (bin(m1).count('1') - 1)
                if cost < best_cost:
                    best_cost = cost
                    best_m1 = m1
                    best_m2 = m2
                if c1:
                    break

        # Strategy 2: Try balanced splits (for larger popcounts)
        if popcount >= 4:
            bits = [i for i in range(n) if mask & (1 << i)]
            k = len(bits)
            # Try splitting into two halves
            half = k // 2
            m1 = 0
            m2 = 0
            for i in range(half):
                m1 |= (1 << bits[i])
            for i in range(half, k):
                m2 |= (1 << bits[i])

            c1 = m1 in mask_to_wire
            c2 = m2 in mask_to_wire
            cost1 = 0 if c1 else (bin(m1).count('1') - 1)
            cost2 = 0 if c2 else (bin(m2).count('1') - 1)
            total_cost = cost1 + cost2

            if total_cost < best_cost:
                best_cost = total_cost
                best_m1 = m1
                best_m2 = m2

        if best_m1 is None:
            # Fallback: split off one bit
            bit = mask.bit_length() - 1
            best_m1 = mask ^ (1 << bit)
            best_m2 = (1 << bit)

        w1 = get_wire(best_m1)
        w2 = get_wire(best_m2)

        if w1 == w2:
            result = zero_wire
        else:
            gates.append([w1, w2])
            result = next_wire
            next_wire += 1
            wire_mask.append(0)  # placeholder, we'll set it below
            wire_mask[result] = mask

        mask_to_wire[mask] = result
        return result

    # Process rows in order of increasing popcount to maximize sharing
    row_indices = list(range(n_rows))
    row_indices.sort(key=lambda i: bin(row_masks[i]).count('1'))

    for i in row_indices:
        mask = row_masks[i]
        if mask != 0 and mask not in mask_to_wire:
            get_wire(mask)

    # Build outputs
    outputs = []
    for i in range(n_rows):
        mask = row_masks[i]
        if mask == 0:
            outputs.append(zero_wire)
        else:
            outputs.append(mask_to_wire[mask])

    return {'gates': gates, 'outputs': outputs}
