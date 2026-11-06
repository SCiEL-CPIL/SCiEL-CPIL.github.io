import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = matrix.shape[0]
    m = matrix.shape[1]
    
    rows = []
    for i in range(n):
        val = 0
        for j in range(m):
            if matrix[i, j] == 1:
                val |= (1 << j)
        rows.append(val)
        
    # wires list
    wires = [1 << i for i in range(n)]
    val_to_idx = {w: i for i, w in enumerate(wires)}
    gates = []
    
    # Targets are the rows
    # We need to ensure all rows are in val_to_idx
    
    while True:
        if time.time() > deadline:
            break
            
        # Check if all rows are in val_to_idx
        missing = [t for t in rows if t not in val_to_idx]
        if not missing:
            break
            
        # Find best gate
        best_g = None
        best_score = -1
        best_u_idx = -1
        best_v_idx = -1
        
        l = len(wires)
        for i in range(l):
            for j in range(i + 1, l):
                g = wires[i] ^ wires[j]
                if g in val_to_idx:
                    continue
                
                # Score
                sub_count = 0
                for t in missing:
                    if (t & g) == g:
                        sub_count += 1
                
                w = bin(g).count('1')
                
                if sub_count > 0:
                    final_score = 1000 * sub_count - w
                else:
                    final_score = -w
                
                if final_score > best_score:
                    best_score = final_score
                    best_g = g
                    best_u_idx = i
                    best_v_idx = j

        if best_g is None:
            # No improvement possible? Should not happen if missing is not empty
            # unless we are stuck in a cycle, but we only add new vectors.
            break
            
        new_idx = len(wires)
        wires.append(best_g)
        val_to_idx[best_g] = new_idx
        
        u_wire = best_u_idx if best_u_idx < n else best_u_idx + 1
        v_wire = best_v_idx if best_v_idx < n else best_v_idx + 1
        
        gates.append([u_wire, v_wire])

    # Map outputs
    outputs = []
    for t in rows:
        if t in val_to_idx:
            idx = val_to_idx[t]
            outputs.append(idx if idx < n else idx + 1)
        else:
            # Fallback: should not happen
            outputs.append(0)

    return {'gates': gates, 'outputs': outputs}
