import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 45  # Safety margin
    
    if not matrix:
        return {'gates': [], 'outputs': []}
        
    n = len(matrix[0])
    m = len(matrix)
    
    # Baseline calculation
    baseline_gates = 0
    for row in matrix:
        wt = sum(row)
        if wt > 1:
            baseline_gates += wt - 1
            
    # Greedy Synthesis
    target_masks = []
    for row in matrix:
        mask = 0
        for j, val in enumerate(row):
            if val:
                mask |= (1 << j)
        target_masks.append(mask)
        
    available_masks = [1 << i for i in range(n)]
    mask_to_wire = {1 << i: i for i in range(n)}
    wire_depth = [0] * n # Inputs have depth 0
    gates = []
    
    required = set(target_masks)
    
    def add_gate(w1, w2, result_mask):
        gates.append([w1, w2])
        new_wire = len(mask_to_wire)
        mask_to_wire[result_mask] = new_wire
        available_masks.append(result_mask)
        wire_depth.append(max(wire_depth[w1], wire_depth[w2]) + 1)
        
    while required:
        if time.time() > deadline:
            return _baseline_synthesize(matrix)
            
        # Remove any required masks that are now available
        still_required = set()
        for r in required:
            if r not in mask_to_wire:
                still_required.add(r)
        required = still_required
        
        if not required:
            break
            
        best_benefit = -1
        best_w1 = -1
        best_w2 = -1
        best_mask = -1
        
        for i in range(len(available_masks)):
            m1 = available_masks[i]
            d1 = wire_depth[i]
            for j in range(i + 1, len(available_masks)):
                m2 = available_masks[j]
                d2 = wire_depth[j]
                
                # Check depth constraint
                if max(d1, d2) + 1 > depth_limit:
                    continue
                    
                candidate = m1 ^ m2
                
                if candidate in mask_to_wire:
                    continue
                    
                benefit = 0
                for r in required:
                    if candidate == r:
                        benefit += 1000
                        break
                    wt_r = bin(r).count('1')
                    wt_res = bin(r ^ candidate).count('1')
                    benefit += (wt_r - wt_res)
                    
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_w1 = mask_to_wire[m1]
                    best_w2 = mask_to_wire[m2]
                    best_mask = candidate
                    
        if best_w1 == -1:
            # No valid candidates due to depth or other reasons
            # Try to relax? Or fallback.
            # If depth is the issue, we might need a different structure.
            # For now, fallback to baseline.
            return _baseline_synthesize(matrix)
            
        add_gate(best_w1, best_w2, best_mask)
        
    # Check if any output is missing
    missing = [mask for mask in target_masks if mask not in mask_to_wire]
    if missing:
        return _baseline_synthesize(matrix)
        
    outputs = [mask_to_wire[mask] for mask in target_masks]
    
    # Compare with baseline
    if len(gates) > baseline_gates:
        return _baseline_synthesize(matrix)
        
    return {'gates': gates, 'outputs': outputs}

def _baseline_synthesize(matrix):
    n = len(matrix[0])
    m = len(matrix)
    gates = []
    outputs = []
    current_wire = n
    
    for i in range(m):
        row = matrix[i]
        ones = [j for j, val in enumerate(row) if val]
        if not ones:
            outputs.append(n) # Constant 0
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            acc = ones[0]
            for k in range(1, len(ones)):
                gates.append([acc, ones[k]])
                acc = current_wire
                current_wire += 1
            outputs.append(acc)
            
    return {'gates': gates, 'outputs': outputs}
