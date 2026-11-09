import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = 55.0
    
    if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
        pass # already list
    # Assuming input is compatible with numpy or list of lists
    
    # Flatten to list of ints
    targets = []
    if len(matrix) == 0:
        return {'gates': [], 'outputs': []}
        
    n = len(matrix[0])
    m = len(matrix)
    
    for row in matrix:
        val = 0
        for j in range(n):
            if row[j]:
                val |= (1 << j)
        targets.append(val)
        
    wires = {}
    depths = {}
    val_to_wire = {}
    
    for i in range(n):
        wires[i] = 1 << i
        depths[i] = 0
        val_to_wire[1 << i] = i
    wires[n] = 0
    depths[n] = 0
    val_to_wire[0] = n
    
    next_wire = n + 1
    gates = []
    
    active_targets = {}
    for i, t in enumerate(targets):
        if t not in active_targets:
            active_targets[t] = []
        active_targets[t].append(i)
        
    resolved_indices = {}
    
    # Initial resolution
    for t, idxs in list(active_targets.items()):
        if t in val_to_wire:
            wid = val_to_wire[t]
            for idx in idxs:
                resolved_indices[idx] = wid
            del active_targets[t]
            
    while active_targets:
        if time.time() - start_time > deadline:
            break
            
        best_score = -1
        best_u = None
        best_v = None
        best_w = None
        
        wire_ids = list(wires.keys())
        
        for i in range(len(wire_ids)):
            u = wire_ids[i]
            for j in range(i + 1, len(wire_ids)):
                v = wire_ids[j]
                w_val = wires[u] ^ wires[v]
                
                if w_val in val_to_wire:
                    continue
                    
                d_u = depths[u]
                d_v = depths[v]
                new_depth = max(d_u, d_v) + 1
                if new_depth > depth_limit:
                    continue
                
                score = 0
                for t, idxs in active_targets.items():
                    if t == w_val:
                        score += 10 * len(idxs)
                    else:
                        t_xor_w = t ^ w_val
                        if t_xor_w in val_to_wire:
                            score += 5 * len(idxs)
                        else:
                            wt_t = t.bit_count()
                            wt_new = t_xor_w.bit_count()
                            if wt_new < wt_t:
                                score += (wt_t - wt_new)
                                
                if score > best_score:
                    best_score = score
                    best_u = u
                    best_v = v
                    best_w = w_val
                    
        if best_score <= 0:
            # Fallback
            max_t = max(active_targets.keys(), key=lambda x: x.bit_count())
            best_score = -1
            for i in range(len(wire_ids)):
                u = wire_ids[i]
                for j in range(i + 1, len(wire_ids)):
                    v = wire_ids[j]
                    w_val = wires[u] ^ wires[v]
                    if w_val in val_to_wire:
                        continue
                    d_u = depths[u]
                    d_v = depths[v]
                    new_depth = max(d_u, d_v) + 1
                    if new_depth > depth_limit:
                        continue
                    
                    t_xor_w = max_t ^ w_val
                    wt_new = t_xor_w.bit_count()
                    score = -wt_new
                    if score > best_score:
                        best_score = score
                        best_u = u
                        best_v = v
                        best_w = w_val
                        
            if best_u is None:
                break

        gates.append([best_u, best_v])
        wires[next_wire] = best_w
        depths[next_wire] = max(depths[best_u], depths[best_v]) + 1
        val_to_wire[best_w] = next_wire
        
        if best_w in active_targets:
            idxs = active_targets.pop(best_w)
            for idx in idxs:
                resolved_indices[idx] = next_wire
                
        for t in list(active_targets.keys()):
            t_xor_w = t ^ best_w
            if t_xor_w in val_to_wire:
                wid = val_to_wire[t_xor_w]
                gates.append([next_wire, wid])
                new_wire_for_t = next_wire + 1
                wires[new_wire_for_t] = t
                depths[new_wire_for_t] = max(depths[next_wire], depths[wid]) + 1
                val_to_wire[t] = new_wire_for_t
                next_wire = new_wire_for_t + 1
                
                idxs = active_targets.pop(t)
                for idx in idxs:
                    resolved_indices[idx] = new_wire_for_t

    outputs = [0] * m
    for i in range(m):
        if i in resolved_indices:
            outputs[i] = resolved_indices[i]
            
    return {'gates': gates, 'outputs': outputs}
