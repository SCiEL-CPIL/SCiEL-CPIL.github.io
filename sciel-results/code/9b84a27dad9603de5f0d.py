import numpy as np
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = 85
    
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    # Independent row baseline
    baseline = 0
    for i in range(m):
        w = int(np.sum(matrix[i]))
        if w > 0:
            baseline += (w - 1)
    
    if baseline == 0:
        return {'gates': [], 'outputs': [n] * m}
    
    # Greedy subexpression elimination
    # For each row, maintain a list of wire IDs
    row_wires = []
    for i in range(m):
        bits = np.nonzero(matrix[i])[0]
        if len(bits) == 0:
            row_wires.append([n])
        else:
            row_wires.append([int(b) for b in bits])
    
    gates = []
    next_wire = n + 1
    
    # Wire vectors: wire_id -> frozenset of input bits
    wire_vec = {}
    for i in range(n):
        wire_vec[i] = frozenset([i])
    wire_vec[n] = frozenset()
    
    def xor_fs(s1, s2):
        return frozenset(s1 ^ s2)
    
    def total_cost():
        return sum(len(rw) - 1 for rw in row_wires)
    
    # Greedy: try adding new wires from pairs of existing wires
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        best_saving = 0
        best_w1 = None
        best_w2 = None
        best_new_vec = None
        
        wire_ids = list(wire_vec.keys())
        for i in range(len(wire_ids)):
            for j in range(i+1, len(wire_ids)):
                w1 = wire_ids[i]
                w2 = wire_ids[j]
                new_vec = xor_fs(wire_vec[w1], wire_vec[w2])
                
                # Compute new total cost
                new_cost = 0
                for rw in row_wires:
                    rw_fs = frozenset(rw)
                    cost_keep = len(rw) - 1
                    rw_xor = xor_fs(rw_fs, new_vec)
                    cost_use = 1 + (len(rw_xor) - 1)
                    new_cost += min(cost_keep, cost_use)
                
                saving = total_cost() - new_cost
                if saving > best_saving:
                    best_saving = saving
                    best_w1 = w1
                    best_w2 = w2
                    best_new_vec = new_vec
        
        if best_saving > 0:
            gates.append([best_w1, best_w2])
            wire_vec[next_wire] = best_new_vec
            new_wire_id = next_wire
            next_wire += 1
            
            for k, rw in enumerate(row_wires):
                rw_fs = frozenset(rw)
                cost_keep = len(rw) - 1
                rw_xor = xor_fs(rw_fs, best_new_vec)
                cost_use = 1 + (len(rw_xor) - 1)
                if cost_use < cost_keep:
                    row_wires[k] = [new_wire_id] + list(rw_xor)
            improved = True
    
    # Build outputs
    outputs = []
    for rw in row_wires:
        if len(rw) == 0:
            outputs.append(n)
        elif len(rw) == 1:
            outputs.append(rw[0])
        else:
            prev = rw[0]
            for w in rw[1:]:
                gates.append([prev, w])
                prev = next_wire
                next_wire += 1
            outputs.append(prev)
    
    return {'gates': gates, 'outputs': outputs}
