"""XOR-circuit synthesis: Proactive CSE with frequency-based candidate selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 80  # Safety margin

    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n

    # Convert rows to bitmasks
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n):
            if mat[i][j]:
                mask |= (1 << j)
        row_masks.append(mask)

    # Identify candidates for synthesis
    # 1. All rows
    # 2. Sub-masks that appear in multiple rows
    sub_mask_freq = defaultdict(int)
    
    # Enumerate sub-masks for frequency counting
    # Optimization: Only enumerate if n is small enough or popcount is small
    # For n=16, 2^16 = 65536. 
    # If we have 16 rows, 16 * 65536 = 1M operations. Acceptable.
    
    for r in row_masks:
        # Enumerate all sub-masks of r
        # Standard trick: sub = r; while sub: ... sub = (sub-1) & r
        # But we also need to count the mask itself? 
        # Yes, the mask itself is a sub-mask.
        sub = r
        while sub:
            sub_mask_freq[sub] += 1
            sub = (sub - 1) & r
        # Don't count 0 as a useful candidate for gates usually, unless needed
        # But 0 is always available.

    candidates = set()
    for r in row_masks:
        candidates.add(r)
        
    for mask, freq in sub_mask_freq.items():
        if freq > 1:
            candidates.add(mask)
            
    # Filter out 0 and inputs (inputs are always available)
    candidates.discard(0)
    for i in range(n):
        candidates.discard(1 << i)

    # Sort candidates: 
    # Primary: Frequency (descending)
    # Secondary: Popcount (ascending) - simpler structures first
    # Tertiary: Mask value (for determinism)
    def sort_key(m):
        return (-sub_mask_freq[m], m.bit_count(), m)
    
    sorted_candidates = sorted(candidates, key=sort_key)

    # State
    available = set()
    for i in range(n):
        available.add(1 << i)
    available.add(0)
    
    wire_id = {}
    for i in range(n):
        wire_id[1 << i] = i
    wire_id[0] = zero_wire
    
    depth_map = {}
    for i in range(n):
        depth_map[1 << i] = 0
    depth_map[0] = 0
    
    gates = []
    next_wire_id = n + 1

    def get_wire_id(mask):
        return wire_id.get(mask, -1)

    # Process candidates
    for mask in sorted_candidates:
        if time.time() > DEADLINE:
            break
            
        if mask in available:
            continue
            
        # Find best split u, v in available such that u ^ v = mask
        best_u = None
        best_v = None
        best_depth = float('inf')
        
        # Special case: Popcount 2 can be formed by two inputs
        if mask.bit_count() == 2:
            # Find the two bits
            bits = []
            m = mask
            while m:
                lsb = m & -m
                bits.append(lsb)
                m ^= lsb
            if len(bits) == 2:
                u, v = bits[0], bits[1]
                if u in available and v in available:
                    # Depth 1
                    best_u = u
                    best_v = v
                    best_depth = 1
                    # No need to check further, depth 1 is optimal
                    pass
        
        if best_u is None:
            # General case: iterate available wires
            # Optimization: if we find depth 2, we can stop? 
            # Depth 1 is only possible if popcount 2 (handled above).
            # So if we find depth 2, it's the best possible (unless depth 1 was missed, which it wasn't).
            # Wait, if mask is an input, it's in available.
            # If mask is not an input, min depth is 1 (if popcount 2) or 2.
            # So if we find a depth 2 solution, we can break.
            
            for u in available:
                if u == 0:
                    v = mask
                    if v in available and v != 0: # v != mask? No, u^v=mask. If u=0, v=mask. But mask is not available. So this fails.
                        continue
                v = mask ^ u
                if v in available:
                    # Found a pair
                    d_u = depth_map[u]
                    d_v = depth_map[v]
                    d_new = 1 + max(d_u, d_v)
                    if d_new < best_depth:
                        best_depth = d_new
                        best_u = u
                        best_v = v
                        if best_depth <= 2:
                            # Can't get better than 2 (since 1 is impossible for non-popcount-2 non-inputs)
                            # Actually, if mask is popcount 3, min depth is 2 (e.g., e1^e2 ^ e3).
                            # So 2 is optimal.
                            break

        if best_u is not None and best_depth <= depth_limit:
            # Add gate
            id_u = wire_id[best_u]
            id_v = wire_id[best_v]
            
            # Avoid duplicate gates? 
            # If we already have a gate producing this mask, we skip.
            # But we check `if mask in available` at start.
            
            gates.append([id_u, id_v])
            wire_id[mask] = next_wire_id
            depth_map[mask] = best_depth
            available.add(mask)
            next_wire_id += 1

    # Check if all rows are available
    # If not, synthesize remaining rows using a recursive helper or fallback
    # The fallback should use the current `available` set to minimize new gates.
    
    def get_wire_recursive(mask):
        if mask in wire_id:
            return wire_id[mask]
        if mask == 0:
            return zero_wire
        
        # Find best split
        best_u = None
        best_v = None
        best_depth = float('inf')
        
        # Try splits from available wires
        for u in available:
            v = mask ^ u
            if v in available:
                d_new = 1 + max(depth_map[u], depth_map[v])
                if d_new < best_depth:
                    best_depth = d_new
                    best_u = u
                    best_v = v
                    if best_depth <= 2:
                        break
        
        if best_u is not None:
            id_u = wire_id[best_u]
            id_v = wire_id[best_v]
            gates.append([id_u, id_v])
            wire_id[mask] = next_wire_id
            depth_map[mask] = best_depth
            available.add(mask)
            return next_wire_id - 1
        else:
            # Fallback: split mask into two smaller parts
            # This should rarely happen if CSE was good
            # Pick a bit, split into that bit and rest
            lsb = mask & -mask
            rest = mask ^ lsb
            id_lsb = wire_id[lsb]
            id_rest = get_wire_recursive(rest)
            gates.append([id_lsb, id_rest])
            wire_id[mask] = next_wire_id
            depth_map[mask] = 1 + max(depth_map[lsb], depth_map[rest])
            available.add(mask)
            return next_wire_id - 1

    # Ensure all rows are resolved
    for r in row_masks:
        if r not in wire_id:
            get_wire_recursive(r)
            
    # Build outputs
    outputs = []
    for r in row_masks:
        if r in wire_id:
            outputs.append(wire_id[r])
        else:
            outputs.append(zero_wire)

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
