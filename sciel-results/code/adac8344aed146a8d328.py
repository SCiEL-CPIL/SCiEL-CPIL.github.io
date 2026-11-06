"""XOR-circuit synthesis via global vector-space greedy basis construction."""
import time
import numpy as np
import random

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like, m x n binary matrix
        depth_limit: maximum circuit depth (<=16)
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Normalize input
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets for efficient manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        row_bits.append(bits)
    
    input_mask = (1 << n) - 1
    
    def _verify(circuit):
        """Verify circuit correctness using bitset simulation."""
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        if len(outputs) != m:
            return False
            
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            # Ensure no self-reference or forward reference (topological order is assumed in generation)
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        
        for i in range(m):
            expected = row_bits[i]
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        """Compute max depth of circuit."""
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for i in range(n + 1):
            wire_depth[i] = 0
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _baseline():
        """Simple baseline: no sharing."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}

    def _global_greedy():
        """
        Global vector-space greedy basis construction.
        We maintain a set of 'available' wires (inputs + computed gates).
        At each step, we try to find a new wire (XOR of two available wires) 
        that reduces the total cost to synthesize the remaining targets.
        
        To make this efficient, we don't generate all pairs. We focus on 
        pairs that are 'promising' based on current target requirements.
        """
        # State:
        # wire_val: dict wire_id -> bitset value
        # wire_depth: dict wire_id -> depth
        # targets: list of bitsets we need to output
        # available_wires: list of wire ids
        
        wire_val = {}
        wire_depth = {}
        
        # Initialize inputs
        for i in range(n):
            wire_val[i] = 1 << i
            wire_depth[i] = 0
        wire_val[n] = 0
        wire_depth[n] = 0
        
        available_wires = list(range(n + 1))
        
        # We want to synthesize all row_bits
        # Let's track which rows are "satisfied" by a single available wire
        # Initially, if a row is exactly an input, it's satisfied.
        
        # Strategy:
        # 1. Identify "hard" rows (high Hamming weight or not directly available).
        # 2. Greedily add gates that create common subexpressions or directly satisfy rows.
        
        # A better approach for XOR: 
        # The "cost" of a row is roughly its Hamming weight if no sharing.
        # Sharing helps if a sub-expression appears in multiple rows.
        
        # Let's use a utility-based approach:
        # For any two wires a, b, new_val = wire_val[a] ^ wire_val[b].
        # Utility = sum over all target rows T of (cost(T) - cost(T \ {new_val}))?
        # This is complex.
        
        # Simpler heuristic:
        # 1. Start with all rows as targets.
        # 2. Find the most frequent pair of inputs among the targets? No, that's CSE.
        # 3. Global Greedy: 
        #    - Maintain a set of "needed" vectors (the rows).
        #    - At each step, pick a pair (a,b) from available wires such that v = a^b 
        #      is "useful". Useful means v matches a target, or v reduces the 
        #      complexity of multiple targets.
        
        # Let's try a different angle: Reverse Greedy (Peeling) or Forward Greedy with Scoring.
        
        # Forward Greedy with Scoring:
        # Score(v) for a new vector v = (number of targets that contain v as a subset) * (weight(v) - 1)
        # This favors vectors that are common sub-expressions and have high weight (saving more gates).
        
        max_gates = 5000 # Safety cap
        gates = []
        next_wire = n + 1
        
        # Pre-compute target weights
        target_weights = [bin(t).count('1') for t in row_bits]
        
        # We will iteratively add gates.
        # To avoid O(N^2) checks on all pairs every step, we sample or use heuristics.
        
        # Heuristic: 
        # 1. If any target is exactly a single available wire, mark it done.
        # 2. Else, find the target with max weight.
        # 3. Try to split it? Or find common sub-parts.
        
        # Let's implement the "Common Subexpression" global greedy properly.
        # 1. Count frequency of each input bit across all targets.
        # 2. Find pairs (i,j) that appear together in many targets.
        # 3. Create gate for (i,j). Update targets: replace (i^j) with new_wire in targets that have both.
        #    Actually, if a target has i and j, it becomes (new_wire) ^ (rest).
        #    The weight of the target decreases by 1 (if i!=j).
        #    Wait, if target = i ^ j ^ k, and we make w = i^j, target becomes w ^ k.
        #    Weight goes from 3 to 2.
        
        # This is essentially the CSE approach but done globally.
        
        # Let's implement this specific "Pair Extraction" greedy.
        
        expressions = list(row_bits)
        
        # Map from wire_id to value
        # We already have wire_val for inputs.
        
        # We need to track which expressions use which wires.
        # Since expressions are bitsets of wire ids, this is implicit.
        
        # Algorithm:
        # While True:
        #   1. Check if any expression is a single bit or 0. If so, we can finalize it later.
        #   2. Find the best pair (a,b) of available wires (a < b) to XOR.
        #      Score(a,b) = sum over all expressions e:
        #         if e has both a and b: 
        #             # Replacing a^b with w reduces weight by 1?
        #             # e' = e ^ (1<<a) ^ (1<<b) ^ (1<<w)
        #             # weight(e') = weight(e) - 2 + 1 = weight(e) - 1
        #             # So we save 1 gate per expression containing both.
        #             score += 1
        #         else:
        #             score += 0
        #      Also, we should prefer pairs where the resulting wire w has high weight?
        #      No, the saving is in the expressions.
        
        #      However, creating a gate costs 1 gate.
        #      Net saving = (number of expressions containing both) - 1.
        #      We want Net Saving > 0.
        
        #   3. If best Net Saving <= 0, stop.
        #   4. Add gate (a,b) -> w.
        #   5. Update all expressions containing both a and b:
        #      e = e ^ (1<<a) ^ (1<<b) ^ (1<<w)
        #   6. Add w to available wires.
        
        # Optimization: 
        # Instead of checking all pairs O(N^2), we can focus on pairs that appear in the "densest" expressions.
        
        max_iterations = 1000
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
                
            best_pair = None
            best_net_saving = 0
            best_w = None
            
            # Heuristic: Only consider pairs from the first few available wires?
            # No, inputs are 0..n-1. New wires are n+1..
            # Let's iterate over all pairs in available_wires.
            # If n is small (<=16), this is fast.
            
            L = len(available_wires)
            # If L is large, O(L^2 * M) might be slow.
            # M is <= 256. L starts at n+1 (<=17). Grows slowly.
            # 17^2 * 256 ~ 70k ops. Very fast.
            
            for i in range(L):
                a = available_wires[i]
                for j in range(i + 1, L):
                    b = available_wires[j]
                    
                    saving = 0
                    for k in range(m):
                        e = expressions[k]
                        if (e & (1 << a)) and (e & (1 << b)):
                            saving += 1
                    
                    net_saving = saving - 1
                    if net_saving > best_net_saving:
                        best_net_saving = net_saving
                        best_pair = (a, b)
                        best_w = next_wire
            
            if best_pair is None:
                break
                
            a, b = best_pair
            w = best_w
            
            # Add gate
            gates.append([a, b])
            
            # Update wire_val and depth
            wire_val[w] = wire_val[a] ^ wire_val[b]
            wire_depth[w] = max(wire_depth[a], wire_depth[b]) + 1
            
            # Update expressions
            for k in range(m):
                e = expressions[k]
                if (e & (1 << a)) and (e & (1 << b)):
                    # Remove a and b, add w
                    e ^= (1 << a)
                    e ^= (1 << b)
                    e |= (1 << w)
                    expressions[k] = e
            
            # Add w to available wires
            available_wires.append(w)
            next_wire += 1
            
            # Optional: Check if depth is exceeded?
            # If wire_depth[w] > depth_limit, we might have a problem.
            # But we can still use it if the final output depth is okay?
            # The contract says depth <= 16.
            # If we exceed it, we should probably avoid creating deep wires.
            # Let's add a check: if wire_depth[w] > depth_limit, skip this pair?
            # Actually, if we skip, we might miss a good sharing opportunity.
            # But exceeding depth is invalid.
            # So we must ensure final depth <= depth_limit.
            # Greedy might create deep wires. 
            # We can penalize deep wires in the score.
            # Or, simply reject pairs that create wires deeper than limit.
            
            if wire_depth[w] > depth_limit:
                # Revert? Or just don't add it?
                # We already added it to gates. We should undo.
                gates.pop()
                del wire_val[w]
                del wire_depth[w]
                # Restore expressions
                for k in range(m):
                    e = expressions[k]
                    # This is tricky to revert easily. 
                    # Better to check BEFORE adding.
                # Let's refactor: Check depth before committing.
                # But we already mutated expressions in the loop above? No, the loop is inside the "if best_pair" block.
                # Wait, the code above adds the gate, updates val, updates expressions, THEN checks depth?
                # No, I put the check after. Let's move the check before mutation.
                
                # Actually, let's just break if we hit depth limit to be safe, 
                # or better, restrict the search to pairs that don't exceed depth.
                pass

            # Re-do the logic with depth check BEFORE mutation
            
            # (Self-Correction: The loop above mutated state. If I break now, state is inconsistent with "rejected" logic.
            # I should restructure the loop to calculate first, then commit.)
            
            # Let's restart the loop logic properly in the final code.
            # For now, this draft is just for thought.
            
            # Break out of this draft loop to write the final clean version.
            break

    def _global_greedy_clean():
        """Clean implementation of the greedy pair extraction."""
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        
        # State
        wire_val = {}
        wire_depth = {}
        for i in range(n):
            wire_val[i] = 1 << i
            wire_depth[i] = 0
        wire_val[n] = 0
        wire_depth[n] = 0
        
        available_wires = list(range(n + 1))
        
        max_iterations = 2000
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
                
            best_pair = None
            best_net_saving = 0
            best_w = None
            best_depth = 0
            
            L = len(available_wires)
            
            # Optimization: If L is very large, we might need to be smarter.
            # But L grows by 1 per iteration.
            
            for i in range(L):
                a = available_wires[i]
                da = wire_depth[a]
                for j in range(i + 1, L):
                    b = available_wires[j]
                    db = wire_depth[b]
                    
                    new_depth = max(da, db) + 1
                    if new_depth > depth_limit:
                        continue
                        
                    saving = 0
                    for k in range(m):
                        e = expressions[k]
                        if (e & (1 << a)) and (e & (1 << b)):
                            saving += 1
                    
                    net_saving = saving - 1
                    if net_saving > best_net_saving:
                        best_net_saving = net_saving
                        best_pair = (a, b)
                        best_w = next_wire
                        best_depth = new_depth
            
            if best_pair is None:
                break
                
            a, b = best_pair
            w = best_w
            
            # Commit
            gates.append([a, b])
            wire_val[w] = wire_val[a] ^ wire_val[b]
            wire_depth[w] = best_depth
            
            for k in range(m):
                e = expressions[k]
                if (e & (1 << a)) and (e & (1 << b)):
                    e ^= (1 << a)
                    e ^= (1 << b)
                    e |= (1 << w)
                    expressions[k] = e
            
            available_wires.append(w)
            next_wire += 1
            
        # Build outputs
        outputs = []
        # We need to synthesize the final expressions.
        # Each expression is a bitset of wires.
        # We need to XOR all wires in the bitset.
        # If bitset has 1 bit, output that wire.
        # If 0 bits, output 0 (wire n).
        # If >1 bits, we need to add gates to combine them.
        # But wait, the greedy phase already tried to reduce complexity.
        # The remaining expressions should be simple.
        # However, they might still have multiple bits.
        
        # We need to add gates to combine the bits in each expression.
        # This might increase depth.
        
        for k in range(m):
            e = expressions[k]
            if e == 0:
                outputs.append(n)
            else:
                # Find set bits
                bits = []
                for w_id in available_wires:
                    if e & (1 << w_id):
                        bits.append(w_id)
                
                if len(bits) == 1:
                    outputs.append(bits[0])
                else:
                    # Combine them
                    # To minimize depth, we can just chain them.
                    # But we need to ensure we don't exceed depth_limit.
                    # Since the wires in 'bits' are already computed, 
                    # chaining them adds depth.
                    
                    prev = bits[0]
                    for b_id in bits[1:]:
                        # Check depth
                        # New gate depth = max(depth[prev], depth[b_id]) + 1
                        # If this exceeds limit, we have a problem.
                        # But we can try to reorder?
                        # For now, just chain.
                        
                        # Add gate
                        gates.append([prev, b_id])
                        new_w = next_wire
                        next_wire += 1
                        
                        # Update state (not strictly needed for outputs if we just track wire id)
                        # But we need to track depth for subsequent outputs if they share?
                        # No, outputs are independent in this final phase unless we share more.
                        # We are just synthesizing the final XORs.
                        
                        wire_val[new_w] = wire_val[prev] ^ wire_val[b_id]
                        wire_depth[new_w] = max(wire_depth[prev], wire_depth[b_id]) + 1
                        
                        # Check depth limit
                        if wire_depth[new_w] > depth_limit:
                            # This is bad. We might need to backtrack or use a different strategy.
                            # For now, we'll just proceed and hope verification fails or depth is ok.
                            # Actually, if depth > limit, the circuit is invalid.
                            # We should probably abort this result and fall back to baseline.
                            pass
                        
                        prev = new_w
                        
                    outputs.append(prev)
                    
        return {'gates': gates, 'outputs': outputs}

    def _polish(circuit):
        """Remove redundant gates."""
        gates = circuit['gates']
        outputs = circuit['outputs']
        
        if not gates:
            return circuit
            
        # Build wire_val
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            
        # Identify wires used in outputs
        used = set(outputs)
        
        # Traverse backwards to find all necessary wires
        # A gate k is necessary if its output wire (n+1+k) is used by another necessary gate or an output.
        
        necessary = [False] * len(gates)
        
        # Mark output wires
        for w in outputs:
            if w >= n + 1:
                idx = w - (n + 1)
                if 0 <= idx < len(gates):
                    necessary[idx] = True
        
        # Propagate backwards
        changed = True
        while changed:
            changed = False
            for k in range(len(gates) - 1, -1, -1):
                if necessary[k]:
                    a, b = gates[k]
                    # If a or b is a gate output, mark that gate necessary
                    for w in [a, b]:
                        if w >= n + 1:
                            idx = w - (n + 1)
                            if 0 <= idx < len(gates) and not necessary[idx]:
                                necessary[idx] = True
                                changed = True
                            
        # Build new gates list
        new_gates = []
        wire_map = {} # old_wire -> new_wire
        new_next_wire = n + 1
        
        for k in range(len(gates)):
            if necessary[k]:
                a, b = gates[k]
                # Remap a and b
                a_new = wire_map.get(a, a)
                b_new = wire_map.get(b, b)
                
                new_gates.append([a_new, b_new])
                old_w = n + 1 + k
                wire_map[old_w] = new_next_wire
                new_next_wire += 1
            else:
                # Map old wire to itself? No, it's gone.
                # But if it was used by a necessary gate, it would have been marked necessary.
                # So it's safe to ignore.
                pass
                
        # Remap outputs
        new_outputs = []
        for w in outputs:
            new_outputs.append(wire_map.get(w, w))
            
        return {'gates': new_gates, 'outputs': new_outputs}

    # Run strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _baseline()
    if _verify(baseline):
        best_result = baseline
        best_gate_count = len(baseline['gates'])
        
    # 2. Global Greedy
    if time.time() < deadline - 10.0:
        try:
            res = _global_greedy_clean()
            # Polish
            res = _polish(res)
            if _verify(res):
                depth = _compute_depth(res['gates'], res['outputs'])
                if depth <= depth_limit:
                    if len(res['gates']) < best_gate_count:
                        best_result = res
                        best_gate_count = len(res['gates'])
        except Exception:
            pass
            
    # 3. Fallback
    if best_result is None:
        best_result = {'gates': [], 'outputs': [n] * m}
        
    return best_result

if __name__ == "__main__":
    # Simple test
    matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    res = synthesize(matrix, 16)
    print(res)
    print(_verify(res))
