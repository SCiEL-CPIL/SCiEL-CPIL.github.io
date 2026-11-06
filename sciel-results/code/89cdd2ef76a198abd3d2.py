def synthesize(matrix, depth_limit):
    import numpy as np
    from itertools import combinations

    m, n = matrix.shape
    
    # Convert matrix rows to bitmasks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        targets.append((i, mask))
        
    # Pool of available wires
    # wire_id: bitmask
    # 0..n-1 are inputs
    # n is zero
    pool = {}
    for i in range(n):
        pool[i] = (1 << i)
    pool[n] = 0
    
    gates = []
    next_wire = n + 1
    outputs = [None] * m
    
    # To avoid re-synthesizing the same target if it appears multiple times?
    # The task implies unique outputs, but if two rows are identical, we can reuse the wire.
    # We will track which targets are done.
    
    remaining_targets = targets[:]
    
    while remaining_targets:
        best_target_idx = -1
        best_cost = float('inf')
        best_wires = None # list of wire ids to use
        
        for idx, (row_idx, mask) in enumerate(remaining_targets):
            # Try to find representation in pool
            # Cost 0: already in pool
            for w_id, w_mask in pool.items():
                if w_mask == mask:
                    cost = 0
                    wires = [w_id]
                    if cost < best_cost:
                        best_cost = cost
                        best_target_idx = idx
                        best_wires = wires
                    break
            
            if best_cost == 0:
                break # Can't do better than 0
                
            # Cost 1: x = a ^ b
            # We need to check pairs in pool
            # Optimization: If pool is large, this is slow.
            # But n is small.
            pool_ids = list(pool.keys())
            found = False
            for i in range(len(pool_ids)):
                for j in range(i + 1, len(pool_ids)):
                    w1 = pool_ids[i]
                    w2 = pool_ids[j]
                    if pool[w1] ^ pool[w2] == mask:
                        cost = 1
                        wires = [w1, w2]
                        if cost < best_cost:
                            best_cost = cost
                            best_target_idx = idx
                            best_wires = wires
                        found = True
                        break
                if found:
                    break
                    
            if best_cost == 1:
                break
                
            # Cost 2: x = a ^ b ^ c
            # This is O(P^3). If P is ~20, 8000 ops. Acceptable.
            found = False
            for i in range(len(pool_ids)):
                for j in range(i + 1, len(pool_ids)):
                    for k in range(j + 1, len(pool_ids)):
                        w1 = pool_ids[i]
                        w2 = pool_ids[j]
                        w3 = pool_ids[k]
                        if pool[w1] ^ pool[w2] ^ pool[w3] == mask:
                            cost = 2
                            wires = [w1, w2, w3]
                            if cost < best_cost:
                                best_cost = cost
                                best_target_idx = idx
                                best_wires = wires
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
                    
            if best_cost == 2:
                break
                
            # Fallback: Use inputs directly
            # Count bits
            weight = bin(mask).count('1')
            cost = weight - 1
            wires = [j for j in range(n) if (mask >> j) & 1]
            if cost < best_cost:
                best_cost = cost
                best_target_idx = idx
                best_wires = wires
                
        # Synthesize the best target
        if best_target_idx == -1:
            # Should not happen
            break
            
        row_idx, mask = remaining_targets[best_target_idx]
        
        if best_cost == 0:
            outputs[row_idx] = best_wires[0]
        else:
            # Create gates to form the wire
            # We have a list of wires to XOR
            # Chain them
            current = best_wires[0]
            for w in best_wires[1:]:
                gates.append([current, w])
                new_mask = pool[current] ^ pool[w]
                pool[next_wire] = new_mask
                current = next_wire
                next_wire += 1
                
            outputs[row_idx] = current
            # The new wire 'current' is now in the pool (added above)
            
        # Remove from remaining
        remaining_targets.pop(best_target_idx)
        
    return {'gates': gates, 'outputs': outputs}
