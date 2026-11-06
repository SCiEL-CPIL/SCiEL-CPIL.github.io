def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    # Parse matrix
    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.uint8)
    elif not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=np.uint8)
    
    m, n = matrix.shape
    if m == 0:
        return {'gates': [], 'outputs': [n] * 0}  # n is zero wire
    
    # Represent vectors as Python ints for fast XOR
    def vec_to_int(row):
        v = 0
        for i in range(n):
            if row[i]:
                v |= (1 << i)
        return v
    
    targets = [vec_to_int(matrix[i]) for i in range(m)]
    
    # Available wires: index -> vector (as int)
    # Wires 0..n-1 are inputs, wire n is zero
    wires = {}
    for i in range(n):
        wires[i] = (1 << i)
    wires[n] = 0
    
    gates = []  # list of (wire_a, wire_b) for each gate
    next_wire = n + 1
    
    # Track which targets are already available
    target_remaining = list(range(m))
    
    # Check if a target is available
    def is_available(target_idx):
        return targets[target_idx] in wires.values()
    
    # Greedy: at each step, find the best new wire to add
    max_steps = 1000
    step = 0
    
    while target_remaining and step < max_steps:
        if time.time() > DEADLINE:
            break
        
        best_val = -1
        best_pair = None
        best_vec = None
        
        # Try all pairs of available wires
        wire_list = list(wires.keys())
        L = len(wire_list)
        
        for i in range(L):
            for j in range(i+1, L):
                a = wire_list[i]
                b = wire_list[j]
                c = wires[a] ^ wires[b]
                
                if c in wires.values():
                    continue
                
                # Compute value: how much does adding c help?
                val = 0
                for t in target_remaining:
                    # Check if c helps compute target t
                    # If target t XOR c is "simpler" than target t, it helps
                    # Simple heuristic: if popcount(target_t XOR c) < popcount(target_t), it helps
                    # Even better: if target_t XOR c is already in wires, then we can finish t with 1 more gate
                    if (targets[t] ^ c) in wires.values():
                        val += 100  # can complete target t with one more gate
                    elif c in wires.values():
                        pass
                    else:
                        # Partial benefit: reduce effective complexity
                        # Use popcount reduction as proxy
                        pc_t = bin(targets[t]).count('1')
                        pc_tc = bin(targets[t] ^ c).count('1')
                        if pc_tc < pc_t:
                            val += (pc_t - pc_tc)
                
                if val > best_val:
                    best_val = val
                    best_pair = (a, b)
                    best_vec = c
        
        if best_pair is None or best_val <= 0:
            # No beneficial XOR found; force completion
            # For each remaining target, compute it directly from inputs
            for t in target_remaining[:]:
                tv = targets[t]
                if tv == 0:
                    # Output is zero wire
                    pass
                elif tv in wires.values():
                    pass
                else:
                    # Build tv from input wires
                    # Find subset of inputs that XOR to tv
                    # Simple: use Gaussian elimination or just use all bits set
                    # Use the bits set in tv
                    bits = [i for i in range(n) if (tv >> i) & 1]
                    if len(bits) == 0:
                        out_wire = n  # zero
                    elif len(bits) == 1:
                        out_wire = bits[0]
                    else:
                        # XOR them together
                        cur = bits[0]
                        for b in bits[1:]:
                            new_wire = next_wire
                            next_wire += 1
                            gates.append((cur, b))
                            wires[new_wire] = wires[cur] ^ wires[b]
                            cur = new_wire
                        out_wire = cur
                target_remaining.remove(t)
            break
        
        a, b = best_pair
        new_wire = next_wire
        next_wire += 1
        gates.append((a, b))
        wires[new_wire] = best_vec
        
        step += 1
    
    # Build outputs: for each target, find a wire that equals it
    outputs = []
    for t in range(m):
        tv = targets[t]
        # Find which wire has this value
        found = False
        for w, v in wires.items():
            if v == tv:
                outputs.append(w)
                found = True
                break
        if not found:
            # Shouldn't happen, but fallback: compute from inputs
            bits = [i for i in range(n) if (tv >> i) & 1]
            if len(bits) == 0:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                # Build it
                cur = bits[0]
                for b in bits[1:]:
                    new_wire = next_wire
                    next_wire += 1
                    gates.append((cur, b))
                    wires[new_wire] = wires[cur] ^ wires[b]
                    cur = new_wire
                outputs.append(cur)
    
    return {'gates': gates, 'outputs': outputs}
