import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 87.0  # 90s limit, leave 3s margin

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

    def elapsed():
        return time.time() - start_time

    def popcount(x):
        return bin(x).count('1')

    best_result = None
    best_score = float('inf')

    # Run multiple restarts with adaptive count
    max_restarts = 500
    
    for restart in range(max_restarts):
        if elapsed() > DEADLINE - 3.0:
            break
        
        # State
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
        
        # Precompute wire usage counts for faster delta computation
        # wire_usage[w] = bitmask of rows that use wire w
        wire_usage = {}
        for w in wire_masks:
            usage = 0
            for r in range(m):
                if (row_bits[r] >> w) & 1:
                    usage |= (1 << r)
            wire_usage[w] = usage
        
        # Greedy Loop
        while elapsed() < DEADLINE - 3.0:
            best_delta = 0
            best_u = -1
            best_v = -1
            best_w_exist = -1
            best_new_mask = -1
            
            wire_ids = [w for w in wire_masks.keys() if w != zero_wire]
            k = len(wire_ids)
            
            for i in range(k):
                u = wire_ids[i]
                usage_u = wire_usage[u]
                for j in range(i + 1, k):
                    v = wire_ids[j]
                    usage_v = wire_usage[v]
                    
                    # Quick check: if no row uses both u and v, skip
                    common_rows = usage_u & usage_v
                    if common_rows == 0:
                        continue
                    
                    mask_u = wire_masks[u]
                    mask_v = wire_masks[v]
                    new_mask = mask_u ^ mask_v
                    
                    if new_mask == 0:
                        continue

                    if new_mask in mask_to_wire:
                        w_exist = mask_to_wire[new_mask]
                        if w_exist == u or w_exist == v:
                            continue
                    else:
                        w_exist = -1
                    
                    # Calculate delta cost using precomputed usage
                    delta_gates = 0 if w_exist != -1 else 1
                    delta_wires = 0
                    
                    if w_exist != -1:
                        usage_w = wire_usage[w_exist]
                        # For rows with both u and v:
                        #   if w also present: delta = -3
                        #   else: delta = -1
                        rows_with_uvw = common_rows & usage_w
                        rows_with_uv_not_w = common_rows & ~usage_w
                        delta_wires = -3 * popcount(rows_with_uvw) - 1 * popcount(rows_with_uv_not_w)
                    else:
                        # For rows with both u and v: delta = -1
                        delta_wires = -1 * popcount(common_rows)
                    
                    delta_cost = delta_gates + delta_wires
                    
                    if delta_cost < best_delta:
                        best_delta = delta_cost
                        best_u = u
                        best_v = v
                        best_w_exist = w_exist
                        best_new_mask = new_mask

            if best_delta >= 0:
                break
            
            # Apply best move
            u = best_u
            v = best_v
            new_mask = best_new_mask
            
            if best_w_exist != -1:
                w = best_w_exist
            else:
                w = next_wire_id
                next_wire_id += 1
                gates.append([u, v])
                wire_masks[w] = new_mask
                mask_to_wire[new_mask] = w
            
            # Update row_bits and wire_usage
            for r in range(m):
                if (row_bits[r] >> u) & 1 and (row_bits[r] >> v) & 1:
                    row_bits[r] &= ~(1 << u)
                    row_bits[r] &= ~(1 << v)
                    
                    if best_w_exist != -1:
                        if (row_bits[r] >> w) & 1:
                            row_bits[r] &= ~(1 << w)
                            wire_usage[w] &= ~(1 << r)
                        else:
                            row_bits[r] |= (1 << w)
                            wire_usage[w] |= (1 << r)
                    else:
                        row_bits[r] |= (1 << w)
            
            # Update wire_usage for u and v
            wire_usage[u] = 0
            wire_usage[v] = 0
            if best_w_exist == -1:
                wire_usage[w] = 0
                for r in range(m):
                    if (row_bits[r] >> w) & 1:
                        wire_usage[w] |= (1 << r)

        # Final Output Construction
        outputs = [0] * m
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs[i] = zero_wire
            elif bits & (bits - 1) == 0:
                outputs[i] = bits.bit_length() - 1
            else:
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

        # Calculate final score for this restart
        final_gates = len(gates)
        if best_score > final_gates:
            best_score = final_gates
            best_result = {'gates': gates, 'outputs': outputs}

    if best_result is None:
        # Fallback: return trivial circuit
        gates = []
        outputs = []
        for i in range(m):
            bits = row_masks[i]
            if bits == 0:
                outputs.append(n)
            elif bits & (bits - 1) == 0:
                outputs.append(bits.bit_length() - 1)
            else:
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
                            new_w = n + 1 + len(gates)
                            gates.append([w1, w2])
                            new_current_wires.append(new_w)
                        else:
                            new_current_wires.append(current_wires[k])
                    current_wires = new_current_wires
                outputs.append(current_wires[0])
        best_result = {'gates': gates, 'outputs': outputs}

    return best_result
