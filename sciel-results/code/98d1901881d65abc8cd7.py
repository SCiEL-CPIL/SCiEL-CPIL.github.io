"""XOR-circuit synthesis with simulated annealing over synthesis order."""
import numpy as np
import time
import random
import math
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    A = np.array(matrix, dtype=np.int64)
    m, n = A.shape
    
    # Convert rows to bitmasks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i, j] % 2 == 1:
                mask |= (1 << j)
        targets.append(mask)
    
    # Baseline: independent row synthesis
    # Cost is sum(popcount(row) - 1) for rows with popcount > 1
    baseline_gates = 0
    for t in targets:
        bits = bin(t).count('1')
        if bits > 1:
            baseline_gates += bits - 1
    baseline = max(baseline_gates, 1)
    
    def time_left():
        return deadline - time.time()
    
    def evaluate_order(order):
        """
        Greedily synthesize targets in the given order.
        Returns (gates, outputs, valid, depth).
        """
        # wires: list of bitmasks
        # wire_map: bitmask -> wire_index
        # depths: list of depths for each wire
        
        wires = [1 << i for i in range(n)]
        wires.append(0) # zero wire
        wire_map = {w: i for i, w in enumerate(wires)}
        depths = [0] * (n + 1)
        gates = []
        
        # To speed up finding pairs, we maintain a list of available wire indices
        # and use the wire_map for O(1) lookup of the partner.
        
        for idx in order:
            target = targets[idx]
            
            if target in wire_map:
                continue
            
            found = False
            # Try to find a, b in wire_map such that a ^ b = target
            # Iterate over available wires. 
            # Heuristic: Prefer wires that are inputs (0..n) or recently added?
            # For now, iterate over all, but break on first find? 
            # No, we should pick the "best" pair to allow for better future sharing?
            # Actually, in a greedy pass, any valid pair works for correctness.
            # However, the choice affects the *set* of available wires for future rows.
            # Different pairs create different intermediate wires.
            # To keep it fast, we just pick the first valid pair found.
            # The search space is the *order* of rows, which determines which targets 
            # are available to share.
            
            # Optimization: If target is a single bit, it's an input.
            # If target is 0, it's the zero wire.
            
            # Iterate over a subset of wires to find a partner?
            # Full iteration is O(K) where K is num wires.
            # For each a, b = a ^ target. Check if b in wire_map.
            
            # To improve sharing, we might prefer pairs where one operand is an input 
            # or a "simple" wire, but let's stick to first-found for speed in the inner loop,
            # and let the SA handle the order.
            
            # Actually, a better heuristic for the greedy step:
            # Try to use inputs first, then intermediate wires.
            # But simply iterating over wire_map keys is fine.
            
            # To make the evaluation deterministic and fast, we iterate over 
            # the keys of wire_map. 
            # Note: The set of keys changes as we add wires.
            
            # We can optimize by pre-computing that if target has popcount 1, 
            # it's handled by inputs.
            
            # Let's just iterate.
            # We need to be careful: if we add a wire, it becomes available for subsequent rows.
            
            # For a fixed order, the greedy choice of *which* pair to use for a target
            # can also be optimized. Here we use a simple heuristic:
            # Prefer pairs where one wire is an input (0..n) or zero (n).
            # This keeps the circuit "flat" and might help with depth, but for gate count,
            # any pair is 1 gate. The difference is which *new* wire is created.
            # Actually, the new wire is always the target. The operands are existing wires.
            # So the choice of operands does NOT change the set of available wires (targets) 
            # for future steps, only the gate count (which is always 1) and the depth.
            # Wait, if I compute target = a ^ b, the new wire is target.
            # The available wires for future steps are all previous wires + target.
            # So the choice of a and b does NOT affect the future availability of targets.
            # Therefore, the greedy synthesis for a fixed order is deterministic in terms 
            # of gate count (each new target costs 1 gate) IF we can always find a pair.
            # If we can't find a pair, we must build it from inputs (cost popcount-1).
            
            # So the "cost" depends on whether target can be formed by XOR of two existing wires.
            # If yes: cost 1.
            # If no: cost popcount(target) - 1 (and we add popcount-1 new wires).
            
            # To minimize cost, we want to maximize the number of targets that can be 
            # formed by XOR of two existing wires.
            # This depends on the order.
            
            # Let's implement the check efficiently.
            
            # We maintain a set of available bitmasks.
            # For each target, check if there exists a in available such that a^target in available.
            
            # To speed this up, we can just check.
            
            # Let's refine the greedy step:
            # 1. If target in wire_map: cost 0.
            # 2. Else, check if exists a in wire_map keys such that (a^target) in wire_map.
            #    If yes, add gate, add target to wire_map. Cost 1.
            # 3. Else, build from inputs.
            #    Find all inputs j where target has bit j.
            #    If count == 1: it's an input, already in wire_map? 
            #       Wait, if target is a single bit, it is an input wire. 
            #       It should have been in wire_map from the start.
            #       So this case is covered by 1.
            #    If count > 1:
            #       We need to XOR these inputs.
            #       This creates intermediate wires.
            #       These intermediate wires are NOT added to the global wire_map in the simple 
            #       "cost 1" case above? 
            #       In the "cost 1" case, we only add the target.
            #       In the "build from inputs" case, we add the target AND the intermediates.
            #       The intermediates might be useful for future targets!
            #       So we MUST add them to wire_map.
            
            # Let's re-evaluate the "build from inputs" logic.
            # If target = x1 ^ x2 ^ ... ^ xk.
            # We compute w1 = x1 ^ x2. Add w1 to wire_map.
            # w2 = w1 ^ x3. Add w2 to wire_map.
            # ...
            # wk-1 = wk-2 ^ xk = target. Add target to wire_map.
            # Total gates: k-1.
            # New wires added: w1, w2, ..., target.
            
            # So the state (wire_map) evolves differently depending on whether we take the 
            # "cheap" path (1 gate) or the "expensive" path (k-1 gates).
            # The expensive path adds more wires to the pool, which might help future targets.
            
            # Therefore, the greedy choice (cheap vs expensive) is not trivial.
            # However, the "cheap" path is always strictly better for immediate cost.
            # Does the "expensive" path ever lead to a lower total cost?
            # Yes, if the intermediates generated by the expensive path are crucial for 
            # future targets that otherwise would be very expensive.
            
            # But in the SA, we only vary the ORDER. We assume a fixed greedy strategy 
            # for the synthesis of a single target given the current pool.
            # Strategy:
            # 1. If target in pool: use it.
            # 2. Else, if exists a, b in pool such that a^b=target: use a^b. (Cost 1)
            # 3. Else: build from inputs. (Cost k-1)
            
            # This strategy is deterministic for a given order and initial pool.
            # So we just need to implement this evaluation fast.
            
            # Implementation details for speed:
            # We maintain `wire_map` (dict: mask -> index).
            # We also need to iterate over keys to find a pair.
            # For large n (up to 16), masks fit in int.
            # Number of wires can grow. 
            # In worst case, if no sharing, wires ~ sum(popcounts).
            # For 16x16, max popcount 16. 16 rows * 16 = 256 wires.
            # Checking pairs for each target: O(K) where K is current num wires.
            # Total eval cost: O(M * K^2) roughly?
            # M=16, K~256. 16 * 256 * 256 = 1M ops. 
            # SA iterations: ~1000-5000.
            # 1M * 5000 = 5B ops. Too slow in Python.
            
            # We need to optimize the "find pair" step.
            # Instead of iterating all keys, we can just iterate?
            # Or we can limit the search?
            # Actually, if we just want to know IF a pair exists, we can iterate.
            # But we also need to update the pool.
            
            # Let's try to keep the evaluation simple and see if it fits in time.
            # If not, we'll reduce SA iterations or simplify the greedy.
            
            # Alternative: Pre-compute all possible XORs? No, pool changes.
            
            # Let's optimize the inner loop:
            # For a target, iterate over available masks.
            # If we find a match, we add 1 gate.
            # If not, we add k-1 gates.
            
            # We will implement the evaluation and monitor time.
            
            pass # Placeholder for logic below
            
        # Re-implementing the loop properly inside the function scope for clarity
        
        wires = [1 << i for i in range(n)]
        wires.append(0)
        wire_map = {w: i for i, w in enumerate(wires)}
        # depths not strictly needed for gate count, but for validity check later.
        # We can track depth if needed, but gate count is the primary metric.
        
        gates = []
        
        for idx in order:
            target = targets[idx]
            
            if target in wire_map:
                continue
            
            # Try to find a, b in wire_map such that a ^ b = target
            found_pair = False
            # Iterate over a subset? 
            # To make it deterministic and fast, we iterate over the keys.
            # We can break early if found.
            
            # Optimization: If target is 0, handled.
            # If target is single bit, handled (in wire_map).
            
            # Iterate
            # To avoid O(K^2) in worst case, we can just pick the first match.
            # The choice of match doesn't change the pool (only target is added).
            # So any match is equivalent in terms of future state.
            
            for a_mask in wire_map.keys():
                b_mask = a_mask ^ target
                if b_mask in wire_map:
                    # Found a pair
                    wa = wire_map[a_mask]
                    wb = wire_map[b_mask]
                    new_wire_idx = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(target)
                    wire_map[target] = new_wire_idx
                    found_pair = True
                    break
            
            if found_pair:
                continue
            
            # Build from inputs
            bits = []
            for j in range(n):
                if target & (1 << j):
                    bits.append(j)
            
            if not bits:
                # Target is 0, should have been in wire_map
                continue
                
            if len(bits) == 1:
                # Target is input, should have been in wire_map
                continue
            
            # XOR inputs
            current_wire = wire_map[1 << bits[0]]
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire_idx = n + 1 + len(gates)
                gates.append([current_wire, input_wire])
                new_mask = wires[current_wire] ^ (1 << j)
                wires.append(new_mask)
                wire_map[new_mask] = new_wire_idx
                current_wire = new_wire_idx
                
            # The final current_wire is the target
            # It is already in wire_map from the last step
            
        # Map outputs
        outputs = []
        for i in range(m):
            target = targets[i]
            if target in wire_map:
                outputs.append(wire_map[target])
            else:
                outputs.append(n) # Should not happen
        
        return gates, outputs
    
    # Simulated Annealing
    
    # Initial order: sorted by popcount? Or random?
    # Random is better for SA.
    current_order = list(range(m))
    random.shuffle(current_order)
    
    current_gates, current_outputs = evaluate_order(current_order)
    current_cost = len(current_gates)
    
    best_order = current_order[:]
    best_gates = current_gates
    best_outputs = current_outputs
    best_cost = current_cost
    
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    
    max_iters = 5000 # Adjust based on time
    
    for i in range(max_iters):
        if time_left() < 5.0:
            break
            
        if T < min_T:
            break
            
        # Generate neighbor: swap two random positions
        i1 = random.randint(0, m-1)
        i2 = random.randint(0, m-1)
        if i1 == i2:
            continue
            
        new_order = current_order[:]
        new_order[i1], new_order[i2] = new_order[i2], new_order[i1]
        
        new_gates, new_outputs = evaluate_order(new_order)
        new_cost = len(new_gates)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_order = new_order
            current_gates = new_gates
            current_outputs = new_outputs
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = current_order[:]
                best_gates = current_gates
                best_outputs = current_outputs
                
        T *= cooling_rate
        
    # Final local search: 2-opt
    improved = True
    while improved and time_left() > 2.0:
        improved = False
        for i in range(m):
            for j in range(i+1, m):
                if time_left() < 2.0:
                    break
                new_order = best_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_gates, new_outputs = evaluate_order(new_order)
                new_cost = len(new_gates)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_order = new_order
                    best_gates = new_gates
                    best_outputs = new_outputs
                    improved = True
            if time_left() < 2.0:
                break
    
    # Verify depth
    # We need to check if the depth is <= depth_limit
    # The greedy evaluation didn't track depth.
    # Let's add a quick depth check.
    
    def check_depth(gates, n, limit):
        # wires: 0..n-1 inputs, n zero, n+1.. gates
        # depth[i] for wire i
        depth = [0] * (n + 1 + len(gates))
        # Inputs have depth 0
        # Zero has depth 0
        for k, g in enumerate(gates):
            wire_a, wire_b = g
            wire_out = n + 1 + k
            depth[wire_out] = max(depth[wire_a], depth[wire_b]) + 1
        max_d = max(depth)
        return max_d <= limit
    
    if not check_depth(best_gates, n, depth_limit):
        # Fallback to a safer synthesis: build from inputs for all
        # This ensures depth is low (log or linear)
        # Actually, building from inputs sequentially gives depth = popcount-1.
        # If popcount-1 > depth_limit, we might need to balance.
        # But depth_limit is 16, and max popcount is 16, so depth 15 is ok.
        # So the fallback is just the "build from inputs" strategy for all rows.
        
        wires = [1 << i for i in range(n)] + [0]
        wire_map = {w: i for i, w in enumerate(wires)}
        gates = []
        for i in range(m):
            target = targets[i]
            if target in wire_map:
                continue
            bits = [j for j in range(n) if target & (1 << j)]
            if not bits or len(bits) == 1:
                continue
            current_wire = wire_map[1 << bits[0]]
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire_idx = n + 1 + len(gates)
                gates.append([current_wire, input_wire])
                new_mask = wires[current_wire] ^ (1 << j)
                wires.append(new_mask)
                wire_map[new_mask] = new_wire_idx
                current_wire = new_wire_idx
        outputs = [wire_map[targets[i]] if targets[i] in wire_map else n for i in range(m)]
        best_gates = gates
        best_outputs = outputs
    
    return {'gates': best_gates, 'outputs': best_outputs}
