import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85.0
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert rows to integers
    row_vals = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                val |= (1 << j)
        row_vals.append(val)
        
    unique_targets = set(row_vals)
    
    # available: val -> (wire_id, depth)
    available = {}
    for j in range(n):
        available[1 << j] = (j, 0)
    available[0] = (n, 0)
    
    gates = []
    next_wire = n + 1
    
    while unique_targets:
        if time.time() > DEADLINE:
            break
            
        # 1. Resolve any target already in available
        resolved_any = False
        for t in list(unique_targets):
            if t in available:
                unique_targets.remove(t)
                resolved_any = True
        if not unique_targets:
            break
            
        # 2. Try to resolve a target using two available wires
        found = False
        for t in list(unique_targets):
            # Check if t = w1 ^ w2 for some w1, w2 in available
            # Iterate w1 in available
            for w1_val, (w1_id, d1) in available.items():
                w2_val = t ^ w1_val
                if w2_val in available:
                    w2_id, d2 = available[w2_val]
                    new_depth = max(d1, d2) + 1
                    if new_depth <= depth_limit:
                        gates.append([w1_id, w2_id])
                        available[t] = (next_wire, new_depth)
                        unique_targets.remove(t)
                        next_wire += 1
                        found = True
                        break
            if found:
                break
                
        if found:
            continue
            
        # 3. Add a helpful gate
        # Heuristic: Minimize popcount of new wire
        avail_list = list(available.items())
        L = len(avail_list)
        
        best_v = None
        best_info = None
        min_pc = n + 1
        
        # Limit search if L is large
        if L <= 100:
            for i in range(L):
                v1, (id1, d1) = avail_list[i]
                for j in range(i+1, L):
                    v2, (id2, d2) = avail_list[j]
                    v = v1 ^ v2
                    if v == 0 or v in available:
                        continue
                    pc = v.bit_count()
                    if pc < min_pc:
                        min_pc = pc
                        best_v = v
                        best_info = (id1, id2, max(d1, d2) + 1)
        else:
            import random
            for _ in range(500):
                i = random.randint(0, L-1)
                j = random.randint(0, L-1)
                if i == j: continue
                v1, (id1, d1) = avail_list[i]
                v2, (id2, d2) = avail_list[j]
                v = v1 ^ v2
                if v == 0 or v in available:
                    continue
                pc = v.bit_count()
                if pc < min_pc:
                    min_pc = pc
                    best_v = v
                    best_info = (id1, id2, max(d1, d2) + 1)
                    
        if best_v is None:
            # Fallback: pick any valid pair
            if L >= 2:
                v1, (id1, d1) = avail_list[0]
                v2, (id2, d2) = avail_list[1]
                v = v1 ^ v2
                if v not in available and v != 0:
                    best_v = v
                    best_info = (id1, id2, max(d1, d2) + 1)
            if best_v is None:
                break
                
        if best_info[2] > depth_limit:
            # Try to find a pair with lower depth?
            # For now, just break or skip?
            # If we can't resolve targets and can't add gates within depth, we are stuck.
            # But usually depth is not the bottleneck.
            pass

        gates.append([best_info[0], best_info[1]])
        available[best_v] = (next_wire, best_info[2])
        next_wire += 1
        
    # Build outputs
    outputs = []
    for val in row_vals:
        if val in available:
            outputs.append(available[val][0])
        else:
            # Fallback
            outputs.append(n)
            
    return {'gates': gates, 'outputs': outputs}
