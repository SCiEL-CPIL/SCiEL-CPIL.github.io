import time
import random
import math

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 82.0  # 90s limit, leave 8s margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire management
    wire_masks = {}
    mask_to_wire = {}
    
    # Initialize inputs
    for j in range(n):
        wire_id = j
        mask = 1 << j
        wire_masks[wire_id] = mask
        mask_to_wire[mask] = wire_id
        
    # Initialize zero
    zero_wire = n
    wire_masks[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    next_wire_id = n + 1

    # row_bits[r] is an integer bitmask where bit w is set if wire w is in row r
    row_bits = [0] * m
    
    # Initialize row_bits based on input wires
    for i in range(m):
        mask = row_masks[i]
        if mask == 0:
            row_bits[i] = (1 << zero_wire)
        else:
            bits = 0
            for j in range(n):
                if (mask >> j) & 1:
                    bits |= (1 << j)
            row_bits[i] = bits

    gates = []
    
    def elapsed():
        return time.time() - start_time

    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')

    # Global Greedy Loop
    while elapsed() < DEADLINE - 5.0:
        best_gain = 0
        best_u = -1
        best_v = -1
        best_mask = -1
        best_w_exist = -1
        
        # Get list of current wire IDs
        wire_ids = list(wire_masks.keys())
        k = len(wire_ids)
        
        # Optimization: If k is large, sampling might be needed, but k is small.
        for i in range(k):
            u = wire_ids[i]
            if u == zero_wire:
                continue
            for j in range(i + 1, k):
                v = wire_ids[j]
                if v == zero_wire:
                    continue
                
                mask_u = wire_masks[u]
                mask_v = wire_masks[v]
                new_mask = mask_u ^ mask_v
                
                # Determine if we use existing wire or create new
                if new_mask in mask_to_wire:
                    w_exist = mask_to_wire[new_mask]
                    if w_exist == u or w_exist == v:
                        continue # Should not happen if u!=v and masks are distinct, but safe
                else:
                    w_exist = -1
                
                # Calculate gain
                # common_rows: rows containing both u and v
                common_rows = 0
                # We need to iterate rows to check bits? 
                # row_bits is a list of ints.
                # We can't easily AND row_bits across rows without iterating.
                # Let's iterate rows to find common rows.
                # This is O(M) per pair. O(K^2 * M).
                # For M=16, K~20, this is 400*16 = 6400 ops. Very fast.
                
                count = 0
                for r in range(m):
                    if (row_bits[r] >> u) & 1 and (row_bits[r] >> v) & 1:
                        count += 1
                        # If w_exist is in row r, gain is higher?
                        # If w_exist is in row r, we can cancel u, v, w_exist.
                        # Gain = 3 instead of 1?
                        # Let's just count rows for now, cancellation is a bonus.
                        # Actually, if w_exist is in row r, replacing {u,v,w_exist} with {} is great.
                        # If w_exist is NOT in row r, replacing {u,v} with {w_exist} is good.
                        # Both reduce size.
                        pass
                
                # Refined Gain:
                # If we create new wire:
                #   Gain = count (each row loses 1 wire: -2 + 1 = -1)
                # If we use existing wire:
                #   For each common row:
                #     If w_exist in row: remove u,v,w_exist. Size -3.
                #     Else: replace u,v with w_exist. Size -1.
                
                gain = 0
                if count > 0:
                    if w_exist == -1:
                        gain = count
                    else:
                        for r in range(m):
                            if (row_bits[r] >> u) & 1 and (row_bits[r] >> v) & 1:
                                if (row_bits[r] >> w_exist) & 1:
                                    gain += 3
                                else:
                                    gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_u = u
                    best_v = v
                    best_mask = new_mask
                    best_w_exist = w_exist
                    
                    # Early break if gain is very high?
                    if best_gain > 2 * m:
                        break
            if best_gain > 2 * m:
                break
        
        if best_gain <= 0:
            break
        
        # Apply the best move
        u = best_u
        v = best_v
        new_mask = best_mask
        
        if best_w_exist != -1:
            w = best_w_exist
            # No new gate
        else:
            w = next_wire_id
            next_wire_id += 1
            gates.append([u, v])
            wire_masks[w] = new_mask
            mask_to_wire[new_mask] = w
        
        # Update row_bits
        for r in range(m):
            if (row_bits[r] >> u) & 1 and (row_bits[r] >> v) & 1:
                # Clear u and v
                row_bits[r] &= ~(1 << u)
                row_bits[r] &= ~(1 << v)
                
                if best_w_exist != -1:
                    if (row_bits[r] >> w) & 1:
                        # Cancellation: w was already there, so u^v^w=0, remove w too
                        row_bits[r] &= ~(1 << w)
                    else:
                        # Add w
                        row_bits[r] |= (1 << w)
                else:
                    # Add new w
                    row_bits[r] |= (1 << w)

    # Final Output Construction
    outputs = [0] * m
    for i in range(m):
        bits = row_bits[i]
        if bits == 0:
            outputs[i] = zero_wire
        elif bits & (bits - 1) == 0:
            # Single bit set
            outputs[i] = bits.bit_length() - 1
        else:
            # Multiple wires: combine them
            wires = []
            temp_bits = bits
            while temp_bits:
                lsb = temp_bits & -temp_bits
                w_id = lsb.bit_length() - 1
                wires.append(w_id)
                temp_bits ^= lsb
            
            current_wires = wires
            while len(current_wires) > 1:
                new_current_wires = []
                for k in range(0, len(current_wires), 2):
                    if k + 1 < len(current_wires):
                        w1 = current_wires[k]
                        w2 = current_wires[k + 1]
                        m1 = wire_masks[w1]
                        m2 = wire_masks[w2]
                        nm = m1 ^ m2
                        if nm in mask_to_wire:
                            new_current_wires.append(mask_to_wire[nm])
                        else:
                            new_w = next_wire_id
                            next_wire_id += 1
                            gates.append([w1, w2])
                            wire_masks[new_w] = nm
                            mask_to_wire[nm] = new_w
                            new_current_wires.append(new_w)
                    else:
                        new_current_wires.append(current_wires[k])
                current_wires = new_current_wires
            outputs[i] = current_wires[0]

    return {'gates': gates, 'outputs': outputs}
