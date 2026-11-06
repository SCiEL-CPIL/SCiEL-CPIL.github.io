import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0 # 5s safety margin for 60s limit
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix to numpy array for faster operations
    mat = np.array(matrix, dtype=np.uint8) % 2
    
    # Parse rows as frozensets of column indices
    rows = []
    for i in range(m):
        row_set = frozenset(np.where(mat[i] == 1)[0].tolist())
        rows.append(row_set)
    
    # Wire assignment
    next_wire = n + 1
    gates = []
    
    # Memo: frozenset -> (wire_id, depth)
    memo = {}
    for j in range(n):
        memo[frozenset([j])] = (j, 0)
    memo[frozenset()] = (n, 0)
    
    def elapsed():
        return time.time() - start_time
    
    def add_gate(a, b, d_a, d_b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        depth = max(d_a, d_b) + 1
        return w, depth
    
    def compute_set(s):
        if s in memo:
            return memo[s]
        
        # If deadline exceeded, fallback to balanced tree to ensure termination
        if elapsed() > DEADLINE:
            # Balanced tree fallback
            if len(s) == 0:
                memo[s] = (n, 0)
                return memo[s]
            if len(s) == 1:
                j = next(iter(s))
                memo[s] = (j, 0)
                return memo[s]
            
            elems = list(s)
            wires_depths = [(j, 0) for j in elems]
            while len(wires_depths) > 1:
                new_wires = []
                for i in range(0, len(wires_depths), 2):
                    if i + 1 < len(wires_depths):
                        w1, d1 = wires_depths[i]
                        w2, d2 = wires_depths[i+1]
                        w, d = add_gate(w1, w2, d1, d2)
                        new_wires.append((w, d))
                    else:
                        new_wires.append(wires_depths[i])
                wires_depths = new_wires
            result = wires_depths[0]
            memo[s] = result
            return result

        # Strategy: Try to find a split s = A U B (disjoint) where A, B are in memo or cheap
        # We prefer A, B in memo (cost 1 gate)
        # Then we prefer A in memo, B small (cost 1 + cost(B))
        
        best_cost = float('inf')
        best_A = None
        best_B = None
        best_wire_A = None
        best_depth_A = None
        best_wire_B = None
        best_depth_B = None
        best_is_both_memo = False

        # Iterate through memo keys that are subsets of s
        # Optimization: Only check keys with size <= len(s)//2 to avoid duplicates?
        # Or just check all.
        
        for A, (wire_A, depth_A) in memo.items():
            if not A.issubset(s):
                continue
            
            B = s - A # Disjoint difference
            
            if B in memo:
                wire_B, depth_B = memo[B]
                # Cost is 1 gate
                # Prefer deeper? No, prefer valid.
                # We want to minimize total gates, so any 1-gate solution is good.
                # However, we might want to avoid creating deep chains.
                cost = 1
                if cost < best_cost or (cost == best_cost and max(depth_A, depth_B) < max(best_depth_A, best_depth_B) if best_A else True):
                    best_cost = cost
                    best_A = A
                    best_B = B
                    best_wire_A = wire_A
                    best_depth_A = depth_A
                    best_wire_B = wire_B
                    best_depth_B = depth_B
                    best_is_both_memo = True
                    if cost == 1:
                        break # Found optimal cost for this step? 
                        # Actually, we might find another 1-gate solution with lower depth.
                        # But for now, any 1-gate solution is a strong candidate.
                        # Let's continue to find lower depth if possible?
                        # No, breaking is faster. We can refine later if needed.
            else:
                # B is not in memo. Estimate cost of B.
                # Cost of B is roughly popcount(B) - 1 if computed from scratch, 
                # but it might share sub-expressions.
                # For now, use popcount(B) as a heuristic cost.
                cost = 1 + len(B) # 1 gate for XOR + estimated cost of B
                if cost < best_cost:
                    best_cost = cost
                    best_A = A
                    best_B = B
                    best_wire_A = wire_A
                    best_depth_A = depth_A
                    best_wire_B = None
                    best_depth_B = None
                    best_is_both_memo = False

        if best_A is not None:
            if best_is_both_memo:
                w, d = add_gate(best_wire_A, best_wire_B, best_depth_A, best_depth_B)
                memo[s] = (w, d)
                return (w, d)
            else:
                # Compute B
                w_B, d_B = compute_set(best_B)
                w, d = add_gate(best_wire_A, w_B, best_depth_A, d_B)
                memo[s] = (w, d)
                return (w, d)
        
        # Fallback: Balanced tree
        if len(s) == 0:
            memo[s] = (n, 0)
            return memo[s]
        if len(s) == 1:
            j = next(iter(s))
            memo[s] = (j, 0)
            return memo[s]
        
        elems = list(s)
        wires_depths = [(j, 0) for j in elems]
        while len(wires_depths) > 1:
            new_wires = []
            for i in range(0, len(wires_depths), 2):
                if i + 1 < len(wires_depths):
                    w1, d1 = wires_depths[i]
                    w2, d2 = wires_depths[i+1]
                    w, d = add_gate(w1, w2, d1, d2)
                    new_wires.append((w, d))
                else:
                    new_wires.append(wires_depths[i])
            wires_depths = new_wires
        result = wires_depths[0]
        memo[s] = result
        return result
    
    outputs = []
    # Sort rows by popcount to compute simpler ones first?
    # Or just process in order.
    # Sorting by popcount might help populate memo with small sets first.
    sorted_indices = sorted(range(m), key=lambda i: len(rows[i]))
    
    for i in sorted_indices:
        if elapsed() > DEADLINE:
            # Fallback for remaining
            # We need to fill outputs. 
            # If we haven't computed row i, we can't just use n (zero) unless row is zero.
            # We must compute it. If we hit deadline, we use the fallback logic inside compute_set
            # which is fast (balanced tree).
            pass
        
        w, d = compute_set(rows[i])
        outputs.append(w)
        
    # Re-map outputs to original order
    # sorted_indices maps original index -> position in sorted list? No.
    # sorted_indices is a list of original indices, sorted by popcount.
    # We computed rows in that order.
    # outputs[i] corresponds to rows[sorted_indices[i]]
    # We need outputs[original_index] = wire
    final_outputs = [0] * m
    for idx, orig_idx in enumerate(sorted_indices):
        final_outputs[orig_idx] = outputs[idx]
    
    return {'gates': gates, 'outputs': final_outputs}
