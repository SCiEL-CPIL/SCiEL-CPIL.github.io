"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0
    
    n_cols = len(matrix[0]) if matrix else 0
    n_rows = len(matrix)
    
    if n_rows == 0:
        return {'gates': [], 'outputs': []}
    
    rows = []
    for row in matrix:
        mask = 0
        for j, bit in enumerate(row):
            if bit:
                mask |= (1 << j)
        rows.append(mask)
    
    def w(m):
        return m.bit_count()
    
    zero_wire = n_cols
    
    # Available wires
    available = {}
    for j in range(n_cols):
        available[j] = (1 << j)
    available[zero_wire] = 0
    
    gates = []
    next_wire = n_cols + 1
    
    # Greedily add beneficial gates
    max_gates = 200
    for _ in range(max_gates):
        if time.time() > deadline - 5.0:
            break
        
        best_gain = 0
        best_pair = None
        best_c = None
        
        wire_ids = list(available.keys())
        
        for i in range(len(wire_ids)):
            for j in range(i + 1, len(wire_ids)):
                a = available[wire_ids[i]]
                b = available[wire_ids[j]]
                c = a ^ b
                if c == 0:
                    continue
                
                # Estimate benefit
                gain = 0
                for k in range(n_rows):
                    r = rows[k]
                    # Current cost to produce r from inputs: w(r) - 1 if r != 0, else 0
                    old_cost = max(0, w(r) - 1) if r != 0 else 0
                    # New cost using c: w(r ^ c) if we can produce (r^c) for free
                    # More accurately: 1 + max(0, w(r^c) - 1) if r^c != 0, else 1
                    if r ^ c == 0:
                        new_cost = 1  # r = c, use c directly (but c costs 1 gate)
                    else:
                        new_cost = 1 + max(0, w(r ^ c) - 1)
                    
                    if new_cost < old_cost:
                        gain += old_cost - new_cost
                
                net_gain = gain - 1  # subtract cost of the gate
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_pair = (wire_ids[i], wire_ids[j])
                    best_c = c
        
        if best_gain <= 0:
            break
        
        a_wire, b_wire = best_pair
        gates.append([a_wire, b_wire])
        available[next_wire] = best_c
        next_wire += 1
    
    # Determine outputs
    outputs = []
    for k in range(n_rows):
        r = rows[k]
        
        # Find best wire to use
        best_wid = None
        best_cost = w(r)  # cost from inputs
        
        for wid, mask in available.items():
            if mask == r:
                best_wid = wid
                best_cost = 0
                break
            else:
                diff = r ^ mask
                if diff == 0:
                    candidate = 0
                else:
                    candidate = 1 + max(0, w(diff) - 1)
                if candidate < best_cost:
                    best_cost = candidate
                    best_wid = wid
        
        if best_wid is not None and best_cost == 0:
            outputs.append(best_wid)
        elif best_wid is not None and best_cost > 0:
            # Need to add gates to produce r using best_wid
            diff = r ^ available[best_wid]
            if diff == 0:
                outputs.append(best_wid)
            else:
                # Produce diff from inputs, then XOR with best_wid
                diff_bits = [j for j in range(n_cols) if (diff >> j) & 1]
                if len(diff_bits) == 1:
                    gates.append([best_wid, diff_bits[0]])
                    outputs.append(next_wire)
                    available[next_wire] = r
                    next_wire += 1
                else:
                    current = diff_bits[0]
                    for b in diff_bits[1:]:
                        gates.append([current, b])
                        current = next_wire
                        next_wire += 1
                    gates.append([best_wid, current])
                    outputs.append(next_wire)
                    available[next_wire] = r
                    next_wire += 1
        else:
            # Produce r directly
            bits = [j for j in range(n_cols) if (r >> j) & 1]
            if len(bits) == 0:
                outputs.append(zero_wire)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                current = bits[0]
                for b in bits[1:]:
                    gates.append([current, b])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
