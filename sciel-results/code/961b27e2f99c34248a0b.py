"""Exact XOR-circuit synthesis: minimize mean gate-count / independent-row baseline over 8x8,12x16,16x16 maps."""
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Wire definitions:
    # 0..n_cols-1: Input wires
    # n_cols: Constant zero wire
    # Gates produce wires starting at n_cols + 1
    
    # State:
    # rem[i] is the set of input indices (as a frozenset) that output i still needs to XOR.
    # If rem[i] is empty, output i is satisfied (currently 0).
    # If rem[i] has 1 element, output i is that input wire.
    # Otherwise, we need to build a circuit for it.
    
    rem = []
    for i in range(n_rows):
        ones = [j for j in range(n_cols) if M[i, j] == 1]
        rem.append(frozenset(ones))
    
    # Map from frozenset of inputs to the wire index that computes their XOR.
    # Initialize with inputs and zero.
    wire_map = {}
    for j in range(n_cols):
        wire_map[frozenset([j])] = j
    wire_map[frozenset()] = n_cols  # Zero wire
    
    next_wire = n_cols + 1
    gates = []
    
    # Iteratively extract common subexpressions
    # We look for sets S (size 2 to 4) that appear as a subset of many current residuals.
    # Creating a wire for S allows us to "XOR out" S from all residuals that contain it,
    # potentially simplifying them.
    
    # To be efficient, we only consider S if it is "common".
    # For small matrices (max 16 cols), the number of possible S of size 2-4 is manageable.
    # Size 2: C(16,2) = 120
    # Size 3: C(16,3) = 560
    # Size 4: C(16,4) = 1820
    # Total ~2500 candidates.
    
    # However, checking if S is a subset of rem[i] for all i is O(n_rows * |S|).
    # We can optimize by only considering S that are subsets of at least 2 residuals.
    
    # Let's iterate. In each iteration, we find the best S to add.
    # "Best" is defined by the reduction in total "cost" or simply the number of residuals it simplifies.
    # A simpler heuristic: Pick the S that appears in the most residuals (with size > 1).
    
    # We need to be careful: adding a gate for S costs 1 gate.
    # It benefits k residuals.
    # If a residual becomes empty or size 1, it's "done".
    # If a residual shrinks, it might be easier to finish.
    
    # Let's try a greedy approach:
    # 1. Count occurrences of all 2-subsets in current residuals (size > 1).
    # 2. Pick the most frequent 2-subset S.
    # 3. If freq(S) > 1 (or some threshold), add gate for S.
    # 4. Update residuals: for any rem[i] containing S, new_rem[i] = rem[i] ^ S.
    #    Note: rem[i] ^ S means symmetric difference. Since S is a subset, it's just removing S.
    # 5. Repeat.
    
    # Why symmetric difference? Because XOR is its own inverse.
    # If rem[i] = {1, 2, 3} and we compute wire w for {1, 2}, then rem[i] becomes {3} and output[i] becomes w XOR 3.
    # Wait, the circuit structure is:
    # output[i] = XOR of wires corresponding to the decomposition.
    # Actually, it's easier to track the "current expression" as a set of wires.
    # But the problem asks for gates and outputs.
    # If we create a wire w for S, and rem[i] contained S, then the new residual for i is rem[i] \ S.
    # And the output for i will be w XOR (circuit for new residual).
    
    # Let's refine the state representation.
    # Instead of just `rem`, let's keep `residuals` as a list of frozensets of input indices.
    # And `expressions` as a list of lists of wires that are XORed to produce the output.
    # Initially, expressions[i] = [] and residuals[i] = M[i].
    # When we add a gate for S:
    #   For each i where S is a subset of residuals[i]:
    #       residuals[i] = residuals[i] - S
    #       expressions[i].append(wire_for_S)
    
    residuals = list(rem)
    expressions = [[] for _ in range(n_rows)]
    
    # Helper to get wire for a set, creating it if necessary?
    # No, we explicitly decide which sets to create.
    
    # We will run this loop until no more beneficial extraction is found or deadline.
    # "Beneficial" heuristic: 
    # Creating a gate for S costs 1.
    # It reduces the popcount of k residuals by |S|.
    # If any residual becomes size <= 1, it is "closed".
    # Let's just pick the most frequent subset of size 2 or 3.
    
    # To avoid infinite loops or useless gates, we require that the subset S is not already in wire_map?
    # Actually, if S is in wire_map, we can just use it without a new gate.
    # But we are generating gates.
    
    # Let's restrict S to size 2 and 3 for efficiency and impact.
    
    while time.time() < deadline - 5.0:
        # Count frequencies of 2-subsets and 3-subsets
        # Only consider residuals with size > 1
        active_residuals = [r for r in residuals if len(r) > 1]
        
        if not active_residuals:
            break
            
        # If all residuals are small, we can just finish them.
        # Let's check if extraction is still worth it.
        # For now, always try.
        
        freq = defaultdict(int)
        
        for r in active_residuals:
            r_list = list(r)
            # 2-subsets
            for i in range(len(r_list)):
                for j in range(i + 1, len(r_list)):
                    s = frozenset([r_list[i], r_list[j]])
                    # Only count if s is not already a wire? 
                    # If it is, we should have used it already? 
                    # No, we might not have created it as a shared gate yet if it wasn't frequent enough.
                    # But if it's in wire_map, it's free to use.
                    # So we should prefer creating NEW wires for common patterns.
                    # If s in wire_map, we don't need to create a gate. We can just "use" it in the expression.
                    # But our current loop structure creates a gate and updates expressions.
                    # If s is already in wire_map, we should just add it to expressions and remove from residual.
                    # Let's handle that separately or check here.
                    
                    # Actually, if s is in wire_map, we can immediately simplify residuals containing s without a new gate.
                    # Let's do a pass to apply existing wires first?
                    # That might be complex. Let's just count frequencies for NEW wires.
                    if s not in wire_map:
                        freq[s] += 1
            
            # 3-subsets
            for i in range(len(r_list)):
                for j in range(i + 1, len(r_list)):
                    for k in range(j + 1, len(r_list)):
                        s = frozenset([r_list[i], r_list[j], r_list[k]])
                        if s not in wire_map:
                            freq[s] += 1
        
        if not freq:
            break
            
        # Find the most frequent subset
        best_s = max(freq, key=lambda x: freq[x])
        
        # Threshold: Is it worth creating a gate?
        # Cost: 1 gate.
        # Benefit: Reduces popcount of freq[best_s] residuals by len(best_s).
        # If freq[best_s] >= 2, it's usually good.
        if freq[best_s] < 2:
            break
            
        # Create gate for best_s
        # We need to build the gate. Since best_s has size 2 or 3, we can build it from inputs or existing wires.
        # To keep depth low, we should build from existing wires if possible.
        # For size 2: s = {a, b}. If a and b are wires, gate is [a, b].
        # For size 3: s = {a, b, c}. 
        #   If {a,b} is a wire w1, gate is [w1, c].
        #   Else if {a,c} is w1, gate is [w1, b].
        #   Else, create w1 for {a,b} (temporarily? No, we need to add that gate too).
        #   This complicates the "single gate" assumption.
        
        # Let's simplify: Only extract size 2 subsets first?
        # Size 2 is always 1 gate.
        # Size 3 might be 2 gates if no size 2 intermediate exists.
        # Let's stick to size 2 for the shared extraction loop to ensure 1 gate cost.
        # Then finish the rest with trees.
        
        # Filter freq to only size 2
        freq_2 = {s: c for s, c in freq.items() if len(s) == 2}
        if not freq_2:
            break
            
        best_s = max(freq_2, key=lambda x: freq_2[x])
        
        if freq_2[best_s] < 2:
            break
            
        # Create wire for best_s
        a, b = list(best_s)
        # Check if a and b are wires (they are inputs or previously created)
        # Since we start with inputs and only add wires, a and b should be in wire_map.
        wa = wire_map[frozenset([a])]
        wb = wire_map[frozenset([b])]
        
        gate = [wa, wb]
        gates.append(gate)
        wire_map[best_s] = next_wire
        next_wire += 1
        
        # Update residuals and expressions
        for i in range(n_rows):
            if len(residuals[i]) > 1 and best_s.issubset(residuals[i]):
                residuals[i] = residuals[i] - best_s
                expressions[i].append(wire_map[best_s])
        
        # Check if any residual became simple
        # No special action needed, just continue loop.
        
    # Final step: Finish all remaining residuals
    for i in range(n_rows):
        r = residuals[i]
        if len(r) == 0:
            # Output is XOR of expressions[i]
            # If expressions[i] is empty, output is 0 (wire n_cols)
            # If expressions[i] has 1 wire, output is that wire
            # If more, need to XOR them.
            pass
        elif len(r) == 1:
            # Output is XOR of expressions[i] and the single input
            r_list = list(r)
            expressions[i].append(wire_map[frozenset([r_list[0]])])
        else:
            # Build a tree for the remaining residual r
            # And add the resulting wire to expressions[i]
            # We can use the existing wire_map to share sub-trees if possible.
            
            # Let's build a balanced tree for r
            level = [frozenset([x]) for x in r]
            while len(level) > 1:
                new_level = []
                for k in range(0, len(level) - 1, 2):
                    s1 = level[k]
                    s2 = level[k + 1]
                    combined = s1 | s2
                    if combined in wire_map:
                        new_level.append(combined)
                    else:
                        w1 = wire_map[s1]
                        w2 = wire_map[s2]
                        gates.append([w1, w2])
                        wire_map[combined] = next_wire
                        new_level.append(combined)
                        next_wire += 1
                if len(level) % 2 == 1:
                    new_level.append(level[-1])
                level = new_level
            
            # The root of this tree is level[0]
            expressions[i].append(wire_map[level[0]])
        
        # Now, expressions[i] contains a list of wires that need to be XORed to produce output i.
        # If len(expressions[i]) == 0: output is 0
        # If len(expressions[i]) == 1: output is that wire
        # If len > 1: build a tree for these wires?
        # Note: The wires in expressions[i] are distinct (mostly).
        # We can just XOR them sequentially or in a tree.
        # Since we want to minimize gates, and these wires are "high level",
        # we can build a tree.
        
        expr_wires = expressions[i]
        if not expr_wires:
            final_output = n_cols  # Zero
        elif len(expr_wires) == 1:
            final_output = expr_wires[0]
        else:
            # Build tree for expr_wires
            # These are wire indices, not input sets.
            # We can treat them as "leaves" for a new tree.
            # But we don't have a map from "set of wires" to wire.
            # We can just create a new tree.
            
            # To reuse, we could key by frozenset of wire indices?
            # That might be overkill for the final step.
            # Let's just build a balanced tree.
            
            level = expr_wires[:]
            while len(level) > 1:
                new_level = []
                for k in range(0, len(level) - 1, 2):
                    w1 = level[k]
                    w2 = level[k + 1]
                    gates.append([w1, w2])
                    new_level.append(next_wire)
                    next_wire += 1
                if len(level) % 2 == 1:
                    new_level.append(level[-1])
                level = new_level
            
            final_output = level[0]
        
        # Store output
        # We need to return a list of outputs.
        # The return format is {'gates': ..., 'outputs': ...}
        # We haven't stored outputs yet. Let's collect them.
        
        # Wait, I didn't initialize an outputs list.
        # Let's restructure to collect outputs at the end.
        
        # Actually, I can just append to a list `outputs`.
        # But I'm inside the loop. Let's define `outputs = []` before the loop.
        
        # Re-doing the final loop logic to be safe.
        
        # I will rewrite the final part to be cleaner.
        
        pass

    # The above logic was a bit messy with state. Let's rewrite the final synthesis part cleanly.
    # The loop above modified `residuals` and `expressions`.
    # Now we just need to finalize `outputs`.
    
    outputs = []
    for i in range(n_rows):
        expr_wires = expressions[i]
        if not expr_wires:
            outputs.append(n_cols)
        elif len(expr_wires) == 1:
            outputs.append(expr_wires[0])
        else:
            level = expr_wires[:]
            while len(level) > 1:
                new_level = []
                for k in range(0, len(level) - 1, 2):
                    w1 = level[k]
                    w2 = level[k + 1]
                    gates.append([w1, w2])
                    new_level.append(next_wire)
                    next_wire += 1
                if len(level) % 2 == 1:
                    new_level.append(level[-1])
                level = new_level
            outputs.append(level[0])
            
    return {'gates': gates, 'outputs': outputs}
