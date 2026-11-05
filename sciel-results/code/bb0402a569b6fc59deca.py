"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    
    # Wire representation: 0..n-1 are inputs, n is zero
    # Gate k produces wire n+1+k
    # We'll track wires as bitmasks (n bits)
    
    # Available wires: dict from bitmask to wire_id
    # wire_id 0..n-1: inputs, wire_id n: zero
    wire_to_mask = {}
    mask_to_wires = {}
    
    for i in range(n):
        mask = 1 << i
        wire_to_mask[i] = mask
        mask_to_wires[mask] = i
    
    zero_mask = 0
    wire_to_mask[n] = zero_mask
    mask_to_wires[zero_mask] = n
    
    # Parse target rows as masks
    target_masks = []
    for row in matrix:
        mask = 0
        for j, val in enumerate(row):
            if val:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # Sort targets by popcount (easiest first)
    target_indices = sorted(range(m), key=lambda i: bin(target_masks[i]).count('1'))
    
    gates = []  # list of (wire_a, wire_b)
    wire_id_counter = n + 1  # next wire id
    wire_depth = {}
    for i in range(n + 1):
        wire_depth[i] = 0
    
    # For each target, find best way to compute it
    output_wire_for_target = [None] * m
    
    for idx in target_indices:
        target = target_masks[idx]
        if target == 0:
            output_wire_for_target[idx] = n
            continue
        
        # Find best decomposition: try all pairs of available wires
        # Cost = number of additional gates needed
        best_cost = float('inf')
        best_pair = None
        best_remaining = None
        
        avail_masks = list(mask_to_wires.keys())
        
        for i, mask_a in enumerate(avail_masks):
            wa = mask_to_wires[mask_a]
            for j, mask_b in enumerate(avail_masks):
                if i == j:
                    continue
                wb = mask_to_wires[mask_b]
                xor_mask = mask_a ^ mask_b
                remaining = target ^ xor_mask
                # Cost: 1 gate for this XOR + cost to compute remaining from inputs
                # remaining must be computable from inputs
                popcount_rem = bin(remaining).count('1')
                # remaining can be computed with popcount_rem - 1 gates if popcount_rem > 0
                cost = 1 + max(0, popcount_rem - 1)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (wa, wb)
                    best_remaining = remaining
        
        # If no good pair found, just compute from inputs directly
        if best_pair is None or best_cost > bin(target).count('1') - 1:
            # Direct computation from inputs
            bits = []
            for j in range(n):
                if target & (1 << j):
                    bits.append(j)
            if not bits:
                output_wire_for_target[idx] = n
                continue
            if len(bits) == 1:
                output_wire_for_target[idx] = bits[0]
                continue
            # Chain XOR
            current_wire = bits[0]
            current_depth = 0
            for b in bits[1:]:
                new_wire = wire_id_counter
                wire_id_counter += 1
                gates.append((current_wire, b))
                wire_to_mask[new_wire] = wire_to_mask[current_wire] ^ (1 << b)
                mask_to_wires[wire_to_mask[new_wire]] = new_wire
                current_depth += 1
                wire_depth[new_wire] = max(wire_depth[current_wire], wire_depth[b]) + 1
                current_wire = new_wire
            output_wire_for_target[idx] = current_wire
        else:
            wa, wb = best_pair
            new_wire = wire_id_counter
            wire_id_counter += 1
            gates.append((wa, wb))
            xor_mask = wire_to_mask[wa] ^ wire_to_mask[wb]
            wire_to_mask[new_wire] = xor_mask
            mask_to_wires[xor_mask] = new_wire
            wire_depth[new_wire] = max(wire_depth[wa], wire_depth[wb]) + 1
            
            remaining = best_remaining
            if remaining == 0:
                output_wire_for_target[idx] = new_wire
            else:
                # Compute remaining from inputs
                bits = []
                for j in range(n):
                    if remaining & (1 << j):
                        bits.append(j)
                if not bits:
                    output_wire_for_target[idx] = new_wire
                else:
                    current_wire = new_wire
                    for b in bits:
                        nw = wire_id_counter
                        wire_id_counter += 1
                        gates.append((current_wire, b))
                        wire_to_mask[nw] = wire_to_mask[current_wire] ^ (1 << b)
                        mask_to_wires[wire_to_mask[nw]] = nw
                        wire_depth[nw] = max(wire_depth[current_wire], wire_depth[b]) + 1
                        current_wire = nw
                    output_wire_for_target[idx] = current_wire
    
    outputs = output_wire_for_target
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
