# EVOLVE-BLOCK-START
import time
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 45.0  # Safety margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse to masks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        targets.append(mask)

    # Wire management
    # Wire 0..n-1 are inputs
    # Wire n is constant 0
    # Wire n+1... are gates
    next_wire = n + 1
    gates = []
    
    # Map from mask to wire index
    wire_map = {}
    for j in range(n):
        wire_map[1 << j] = j
    wire_map[0] = n
    
    # Map from wire index to mask
    mask_map = {j: 1 << j for j in range(n)}
    mask_map[n] = 0
    
    def elapsed():
        return time.time() - start_time

    def deadline_reached():
        return elapsed() > DEADLINE

    def add_gate(w1, w2):
        nonlocal next_wire
        gates.append([w1, w2])
        w_new = next_wire
        next_wire += 1
        return w_new

    # Active targets: list of (original_index, current_mask)
    # We track the current mask of each row as we reduce it.
    active_targets = [(i, targets[i]) for i in range(m)]
    solved = [False] * m
    output_wires = [None] * m

    # If a target is 0, it's solved with wire n
    # If a target is a single bit, it's solved with that input wire
    # We handle these in the loop or pre-check.
    
    # Pre-check simple cases
    for i in range(m):
        if targets[i] == 0:
            solved[i] = True
            output_wires[i] = n
        elif targets[i] & (targets[i] - 1) == 0:
            solved[i] = True
            output_wires[i] = targets[i].bit_length() - 1
        else:
            active_targets[i] = (i, targets[i]) # Keep in active list if not solved

    # Filter active_targets to only unsolved
    active_targets = [t for t in active_targets if not solved[t[0]]]

    while active_targets and not deadline_reached():
        best_gain = 0
        best_w1 = -1
        best_w2 = -1
        best_new_mask = -1
        
        # Get list of current wire masks
        current_masks = list(mask_map.values())
        current_wires = list(mask_map.keys())
        
        # Optimization: Only consider pairs that produce a mask that is a submask of some active target?
        # Actually, any mask that reduces a target is useful.
        # A mask M reduces target T if (T & M) == M and T != M.
        # Or if T == M, it solves it.
        
        # To find the best gate, we iterate over all pairs of existing wires.
        # Number of wires grows, but stays small.
        L = len(current_wires)
        
        # We can prune: if a pair produces a mask already in wire_map, skip.
        
        for i in range(L):
            w1 = current_wires[i]
            m1 = current_masks[i]
            for j in range(i + 1, L):
                w2 = current_wires[j]
                m2 = current_masks[j]
                
                new_m = m1 ^ m2
                
                if new_m in wire_map:
                    continue
                    
                # Calculate gain
                gain = 0
                for idx, t_mask in active_targets:
                    if t_mask == new_m:
                        # This solves the target immediately
                        # Value is high, proportional to complexity avoided
                        # Let's say gain += 100 + popcount(t_mask)
                        gain += 100 + t_mask.bit_count()
                    elif (t_mask & new_m) == new_m:
                        # This reduces the target
                        # Gain is reduction in popcount
                        gain += t_mask.bit_count() - (t_mask ^ new_m).bit_count()
                        # Note: (t_mask ^ new_m).bit_count() is the popcount of the reduced mask
                        # Since new_m is subset, popcount(t_mask) = popcount(new_m) + popcount(rest)
                        # So gain = popcount(new_m)
                        # Wait.
                        # T = A + B (disjoint bits). M = A.
                        # T ^ M = B.
                        # popcount(T) = popcount(A) + popcount(B).
                        # popcount(T^M) = popcount(B).
                        # Gain = popcount(A).
                        # So gain is just popcount(new_m).
                        # Let's verify:
                        # gain = t_mask.bit_count() - (t_mask ^ new_m).bit_count()
                        # This is correct.
                        
                if gain > best_gain:
                    best_gain = gain
                    best_w1 = w1
                    best_w2 = w2
                    best_new_mask = new_m

        if best_gain == 0:
            # No useful gate found? This shouldn't happen unless we are stuck.
            # Fallback: just build the remaining targets independently
            break

        # Add the gate
        w_new = add_gate(best_w1, best_w2)
        wire_map[best_new_mask] = w_new
        mask_map[w_new] = best_new_mask
        
        # Update active targets
        new_active = []
        for idx, t_mask in active_targets:
            if t_mask == best_new_mask:
                solved[idx] = True
                output_wires[idx] = w_new
            elif (t_mask & best_new_mask) == best_new_mask:
                # Reduce
                new_t_mask = t_mask ^ best_new_mask
                if new_t_mask == 0:
                    solved[idx] = True
                    output_wires[idx] = n
                elif new_t_mask & (new_t_mask - 1) == 0:
                    solved[idx] = True
                    output_wires[idx] = new_t_mask.bit_length() - 1
                else:
                    new_active.append((idx, new_t_mask))
            else:
                new_active.append((idx, t_mask))
                
        active_targets = new_active

    # Fallback for any unsolved targets (should be rare or if deadline hit)
    for i in range(m):
        if not solved[i]:
            # Simple pairwise XOR fallback
            mask = targets[i]
            bits = [j for j in range(n) if (mask >> j) & 1]
            if not bits:
                output_wires[i] = n
            elif len(bits) == 1:
                output_wires[i] = bits[0]
            else:
                wires = bits[:]
                while len(wires) > 1:
                    new_wires = []
                    for k in range(0, len(wires), 2):
                        if k + 1 < len(wires):
                            w = add_gate(wires[k], wires[k+1])
                            new_wires.append(w)
                        else:
                            new_wires.append(wires[k])
                    wires = new_wires
                output_wires[i] = wires[0]

    return {'gates': gates, 'outputs': output_wires}
# EVOLVE-BLOCK-END
