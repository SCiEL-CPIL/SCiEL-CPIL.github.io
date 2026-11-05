"""XOR-circuit synthesis with wire merging."""
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    # Wire indices: 0..n-1 inputs, n = zero
    # Gate k produces wire n+1+k
    
    # Target mask for each output
    target_masks = []
    for j in range(m):
        mask = tuple(i for i in range(n) if matrix[j, i] == 1)
        target_masks.append(mask)
    
    # Available wires: wire -> mask
    wire_masks = {}
    for i in range(n):
        wire_masks[i] = (i,)
    wire_masks[n] = ()
    
    gates = []
    next_wire = n + 1
    
    def find_wire(mask):
        for w, m2 in wire_masks.items():
            if m2 == mask:
                return w
        return None
    
    def create_wire(mask_a, mask_b):
        """Create XOR of two wires, return new wire index"""
        nonlocal next_wire
        new_mask = tuple(sorted(set(mask_a) ^ set(mask_b)))
        existing = find_wire(new_mask)
        if existing is not None:
            return existing
        gates.append([find_wire(mask_a), find_wire(mask_b)])
        wire_masks[next_wire] = new_mask
        w = next_wire
        next_wire += 1
        return w
    
    # For each output, build its mask
    output_wires = []
    for j in range(m):
        tm = target_masks[j]
        if not tm:
            output_wires.append(n)
        elif len(tm) == 1:
            output_wires.append(tm[0])
        else:
            # Build by XORing inputs
            current = tm[0]
            for k in range(1, len(tm)):
                current = create_wire(wire_masks[current], (tm[k],))
            output_wires.append(current)
    
    return {'gates': gates, 'outputs': output_wires}
