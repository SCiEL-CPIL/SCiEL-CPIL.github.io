import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Parse input
    if isinstance(matrix, np.ndarray):
        mat = matrix
    else:
        mat = np.array(matrix)
        
    if mat.size == 0:
        return {'gates': [], 'outputs': []}
        
    m, n = mat.shape
    
    # Input wires: 0..n-1
    # Zero wire: n
    # Gate outputs: n+1, n+2, ...
    
    # Initialize wires
    wires_val = [1 << i for i in range(n)]
    wires_val.append(0) # Wire n is 0
    
    wire_map = {val: idx for idx, val in enumerate(wires_val)}
    
    # Initialize outputs as sets of wire indices
    out_sets = []
    for i in range(m):
        mask = 0
        row = mat[i]
        for j in range(n):
            if row[j] == 1:
                mask |= (1 << j)
        s = set()
        for j in range(n):
            if mask & (1 << j):
                s.add(j)
        out_sets.append(s)
        
    gates = []
    next_wire_idx = n + 1
    
    # Track co-occurrence: pair_count[(i,j)] = number of outputs containing both i and j
    # Use a dict with frozenset of 2 elements as key
    pair_count = {}
    for s in out_sets:
        items = list(s)
        for idx_a in range(len(items)):
            for idx_b in range(idx_a + 1, len(items)):
                a, b = items[idx_a], items[idx_b]
                key = (a, b) if a < b else (b, a)
                pair_count[key] = pair_count.get(key, 0) + 1
    
    def update_pair_count_for_set(s, old_i, old_j, new_k):
        """Update pair counts when replacing {i,j} with k in set s"""
        items = list(s)
        # Remove pairs involving old_i or old_j that were in s
        for idx_a in range(len(items)):
            for idx_b in range(idx_a + 1, len(items)):
                a, b = items[idx_a], items[idx_b]
                key = (a, b) if a < b else (b, a)
                if key in pair_count:
                    pair_count[key] -= 1
                    if pair_count[key] <= 0:
                        del pair_count[key]
        # Add pairs involving new_k with other elements in s
        for item in items:
            if item == new_k:
                continue
            key = (item, new_k) if item < new_k else (new_k, item)
            pair_count[key] = pair_count.get(key, 0) + 1
    
    while True:
        if time.time() > deadline:
            break
            
        if all(len(s) == 1 for s in out_sets):
            break
            
        # Find best pair to XOR using pair_count
        best_benefit = 0
        best_i = -1
        best_j = -1
        best_secondary = 0  # Secondary: sum of set sizes where pair appears
        
        for key, count in pair_count.items():
            if count <= 0:
                continue
            i, j = key
            if i >= len(wires_val) or j >= len(wires_val):
                continue
            # Primary: benefit (how many outputs contain both)
            # Secondary: prefer pairs in larger sets (reduces more total complexity)
            secondary = 0
            for s in out_sets:
                if i in s and j in s:
                    secondary += len(s)
            
            if count > best_benefit or (count == best_benefit and secondary > best_secondary):
                best_benefit = count
                best_i = i
                best_j = j
                best_secondary = secondary
                
        if best_benefit > 0:
            i, j = best_i, best_j
            w_val = wires_val[i] ^ wires_val[j]
            
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wires_val.append(w_val)
                wire_map[w_val] = k
                gates.append([i, j])
                next_wire_idx += 1
                
            # Update pair counts and output sets
            for s in out_sets:
                if i in s and j in s:
                    # Save old state for pair count update
                    old_items = list(s)
                    s.remove(i)
                    s.remove(j)
                    s.add(k)
                    update_pair_count_for_set(s, i, j, k)
        else:
            # No pair has benefit > 0. Force reduction on most complex output.
            max_idx = 0
            max_size = 0
            for idx, s in enumerate(out_sets):
                if len(s) > max_size:
                    max_size = len(s)
                    max_idx = idx
            
            if max_size <= 1:
                break
                
            s = out_sets[max_idx]
            items = list(s)
            if len(items) < 2:
                break
            i, j = items[0], items[1]
            
            w_val = wires_val[i] ^ wires_val[j]
            
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wires_val.append(w_val)
                wire_map[w_val] = k
                gates.append([i, j])
                next_wire_idx += 1
                
            old_items = list(s)
            s.remove(i)
            s.remove(j)
            s.add(k)
            update_pair_count_for_set(s, i, j, k)

    # Finalize: Ensure all outputs are single wires
    for s in out_sets:
        while len(s) > 1:
            if time.time() > deadline - 5:
                break
            items = list(s)
            i, j = items[0], items[1]
            w_val = wires_val[i] ^ wires_val[j]
            
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wires_val.append(w_val)
                wire_map[w_val] = k
                gates.append([i, j])
                next_wire_idx += 1
                
            s.remove(i)
            s.remove(j)
            s.add(k)
            
            if len(s) >= len(items):
                break

    outputs = []
    for s in out_sets:
        if len(s) == 1:
            outputs.append(list(s)[0])
        else:
            val = 0
            for idx in s:
                val ^= wires_val[idx]
            if val in wire_map:
                outputs.append(wire_map[val])
            else:
                k = next_wire_idx
                wires_val.append(val)
                wire_map[val] = k
                outputs.append(k)
                
    return {'gates': gates, 'outputs': outputs}
