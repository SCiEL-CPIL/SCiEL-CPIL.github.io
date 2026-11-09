import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()
    
    if not matrix:
        return {'gates': [], 'outputs': []}
        
    n = len(matrix[0])
    m = len(matrix)
    
    # Convert rows to integer representations
    targets = []
    for row in matrix:
        val = 0
        for j in range(n):
            if row[j]:
                val |= (1 << j)
        targets.append(val)
        
    # Data structures
    wire_val_to_id = {}
    wire_id_to_val = {}
    depths = {}
    parents = {}
    
    # Initialize inputs (0..n-1) and zero wire (n)
    next_wire = 0
    for i in range(n):
        val = 1 << i
        wire_val_to_id[val] = next_wire
        wire_id_to_val[next_wire] = val
        depths[next_wire] = 0
        parents[next_wire] = None
        next_wire += 1
        
    zero_val = 0
    wire_val_to_id[zero_val] = next_wire
    wire_id_to_val[next_wire] = zero_val
    depths[next_wire] = 0
    parents[next_wire] = None
    next_wire += 1
    
    # Track unresolved targets
    target_counts = defaultdict(int)
    for t in targets:
        target_counts[t] += 1
        
    resolved_wires = {} # target_value -> wire_id
    
    # Check initial resolution
    for t in list(target_counts.keys()):
        if t in wire_val_to_id:
            resolved_wires[t] = wire_val_to_id[t]
            del target_counts[t]
            
    gates = []
    
    # Helper to add a gate
    def add_gate(u, v):
        nonlocal next_wire
        w_val = wire_id_to_val[u] ^ wire_id_to_val[v]
        if w_val in wire_val_to_id:
            return wire_val_to_id[w_val]
        
        new_id = next_wire
        next_wire += 1
        
        d_u = depths[u]
        d_v = depths[v]
        d_new = max(d_u, d_v) + 1
        
        if d_new > depth_limit:
            return None # Invalid depth
            
        wire_val_to_id[w_val] = new_id
        wire_id_to_val[new_id] = w_val
        depths[new_id] = d_new
        parents[new_id] = (u, v)
        gates.append([u, v])
        return new_id

    # Greedy Loop
    while target_counts:
        if time.time() - start_time > deadline:
            break
            
        wire_ids = list(wire_id_to_val.keys())
        best_score = -1
        best_u = None
        best_v = None
        best_w_val = None
        
        L = len(wire_ids)
        for i in range(L):
            u = wire_ids[i]
            val_u = wire_id_to_val[u]
            for j in range(i + 1, L):
                v = wire_ids[j]
                val_v = wire_id_to_val[v]
                
                w_val = val_u ^ val_v
                
                if w_val in wire_val_to_id:
                    continue
                    
                d_new = max(depths[u], depths[v]) + 1
                if d_new > depth_limit:
                    continue
                    
                score = 0
                
                if w_val in target_counts:
                    score += 100 * target_counts[w_val]
                    
                for t, cnt in target_counts.items():
                    if t == w_val:
                        continue
                        
                    t_xor_w = t ^ w_val
                    
                    if t_xor_w in wire_val_to_id:
                        score += 10 * cnt
                    else:
                        wt_t = t.bit_count()
                        wt_new = t_xor_w.bit_count()
                        if wt_new < wt_t:
                            score += (wt_t - wt_new) * cnt
                            
                if score > best_score:
                    best_score = score
                    best_u = u
                    best_v = v
                    best_w_val = w_val
                    
        if best_score <= 0:
            # Fallback
            if target_counts:
                max_t = max(target_counts.keys(), key=lambda x: x.bit_count())
                best_score = -1
                for i in range(L):
                    u = wire_ids[i]
                    val_u = wire_id_to_val[u]
                    for j in range(i + 1, L):
                        v = wire_ids[j]
                        val_v = wire_id_to_val[v]
                        w_val = val_u ^ val_v
                        if w_val in wire_val_to_id:
                            continue
                        d_new = max(depths[u], depths[v]) + 1
                        if d_new > depth_limit:
                            continue
                        
                        t_xor_w = max_t ^ w_val
                        wt_new = t_xor_w.bit_count()
                        score = -wt_new
                        if score > best_score:
                            best_score = score
                            best_u = u
                            best_v = v
                            best_w_val = w_val
                            
                if best_u is None:
                    break
                    
        new_id = add_gate(best_u, best_v)
        if new_id is None:
            break
            
        if best_w_val in target_counts:
            resolved_wires[best_w_val] = new_id
            del target_counts[best_w_val]
            
        for t in list(target_counts.keys()):
            t_xor_new = t ^ best_w_val
            if t_xor_new in wire_val_to_id:
                existing_id = wire_val_to_id[t_xor_new]
                final_id = add_gate(new_id, existing_id)
                if final_id is not None:
                    resolved_wires[t] = final_id
                    del target_counts[t]
                    
    outputs = []
    for t in targets:
        if t in resolved_wires:
            outputs.append(resolved_wires[t])
        else:
            outputs.append(0)
            
    return {'gates': gates, 'outputs': outputs}
