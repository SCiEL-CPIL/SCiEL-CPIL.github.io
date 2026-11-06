"""XOR-circuit synthesis via Enhanced Greedy CSE with Global Final Sharing."""
import time
import numpy as np
import random
import heapq

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
    deadline = start_time + 85.0
    
    # Normalize input
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets
    orig_row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        orig_row_bits.append(bits)

    def _verify(circuit):
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
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        for i in range(m):
            expected = orig_row_bits[i]
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _baseline():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = orig_row_bits[i]
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

    def _polish(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        if not gates:
            return circuit
            
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            
        used = set(outputs)
        for a, b in gates:
            used.add(a)
            used.add(b)
            
        necessary = [False] * len(gates)
        wire_to_gate = {}
        for k in range(len(gates)):
            w = n + 1 + k
            wire_to_gate[w] = k
            
        for w in outputs:
            if w in wire_to_gate:
                necessary[wire_to_gate[w]] = True
                
        changed = True
        while changed:
            changed = False
            for k in range(len(gates) - 1, -1, -1):
                if necessary[k]:
                    a, b = gates[k]
                    for w in [a, b]:
                        if w in wire_to_gate:
                            idx = wire_to_gate[w]
                            if not necessary[idx]:
                                necessary[idx] = True
                                changed = True
                                
        new_gates = []
        wire_map = {}
        new_next_wire = n + 1
        
        for k in range(len(gates)):
            if necessary[k]:
                a, b = gates[k]
                a_new = wire_map.get(a, a)
                b_new = wire_map.get(b, b)
                new_gates.append([a_new, b_new])
                old_w = n + 1 + k
                wire_map[old_w] = new_next_wire
                new_next_wire += 1
                
        new_outputs = []
        for w in outputs:
            new_outputs.append(wire_map.get(w, w))
            
        return {'gates': new_gates, 'outputs': new_outputs}

    def _global_final_sharing(expressions, available_wires, wire_depth, next_wire, current_gates):
        """
        Recursively find and extract common subexpressions from the list of expressions.
        This is a global CSE pass on the final set of tails.
        """
        if time.time() > deadline - 2.0:
            return expressions, next_wire, current_gates

        # Find the pair of wires (a, b) that appears most frequently across the expressions
        # Or rather, find the pair of wires that appear together in the most expressions.
        # Actually, we want to find a subset of wires that appears in multiple expressions.
        # For simplicity, we look for pairs of wires that co-occur in >= 2 expressions.
        
        best_pair = None
        best_count = 0
        best_exprs = []
        
        # To speed up, only consider wires that are actually present in the expressions
        active_wires = set()
        for e in expressions:
            for w in available_wires:
                if e & (1 << w):
                    active_wires.add(w)
        
        active_list = list(active_wires)
        L = len(active_list)
        
        # If L is small, check all pairs
        if L > 30: 
            # Heuristic: just pick the two most frequent wires? Or skip.
            # For now, if too many, we might miss some sharing, but it's safe.
            # Let's limit to top 20 most frequent wires for pairing.
            freq = {w: 0 for w in active_list}
            for e in expressions:
                for w in active_list:
                    if e & (1 << w):
                        freq[w] += 1
            top_wires = sorted(active_list, key=lambda w: freq[w], reverse=True)[:20]
            active_list = top_wires
            L = len(active_list)

        for i in range(L):
            a = active_list[i]
            for j in range(i + 1, L):
                b = active_list[j]
                count = 0
                exprs = []
                for k, e in enumerate(expressions):
                    if (e & (1 << a)) and (e & (1 << b)):
                        count += 1
                        exprs.append(k)
                if count > best_count:
                    best_count = count
                    best_pair = (a, b)
                    best_exprs = exprs
        
        if best_pair and best_count >= 2:
            a, b = best_pair
            new_w = next_wire
            next_wire += 1
            
            # Check depth
            new_depth = max(wire_depth[a], wire_depth[b]) + 1
            if new_depth > depth_limit:
                # Can't create this gate due to depth. 
                # We might need to prune or just skip.
                # For now, skip this pair and try others? 
                # To keep it simple, we stop if the best pair violates depth.
                # This might be suboptimal, but safe.
                pass
            else:
                current_gates.append([a, b])
                wire_depth[new_w] = new_depth
                available_wires.append(new_w)
                
                # Update expressions
                for k in best_exprs:
                    e = expressions[k]
                    e ^= (1 << a)
                    e ^= (1 << b)
                    e |= (1 << new_w)
                    expressions[k] = e
                
                # Recurse
                return _global_final_sharing(expressions, available_wires, wire_depth, next_wire, current_gates)
        
        return expressions, next_wire, current_gates

    def _greedy_cse_enhanced():
        expressions = list(orig_row_bits)
        gates = []
        next_wire = n + 1
        
        wire_val = {}
        wire_depth = {}
        for i in range(n):
            wire_val[i] = 1 << i
            wire_depth[i] = 0
        wire_val[n] = 0
        wire_depth[n] = 0
        
        available_wires = list(range(n + 1))
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
                
            best_pair = None
            best_score = -1
            best_w = None
            best_depth = 0
            
            L = len(available_wires)
            
            # Optimization: Only consider wires that are actually used in expressions?
            # This can be expensive to compute every step. 
            # For n=16, m=16, L is small.
            
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
                    
                    # Enhanced Scoring:
                    # 1. Primary: Saving (number of expressions containing both)
                    # 2. Secondary: Depth penalty (prefer flatter)
                    # 3. Tertiary: Weight reduction (if saving is low)
                    
                    # If saving >= 2, it's a net positive.
                    # If saving == 1, it's net 0, but might reduce max weight.
                    
                    if saving >= 2:
                        score = saving * 100 - new_depth
                    elif saving == 1:
                        # Check if it reduces the max weight of the expressions
                        # We estimate weight reduction by checking if a or b is "heavy" in other contexts?
                        # Simple heuristic: Accept if new_depth is small.
                        score = 10 - new_depth
                    else:
                        score = -100
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_w = next_wire
                        best_depth = new_depth
            
            if best_pair is None or best_score <= 0:
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
            
        # Final Synthesis with Global Sharing
        final_gates = list(gates)
        
        # Run global final sharing on the expressions
        expressions, next_wire, final_gates = _global_final_sharing(
            expressions, available_wires, wire_depth, next_wire, final_gates
        )
        
        # Now synthesize the remaining expressions (should be weight <= 1 mostly, or simple)
        outputs = []
        expr_to_wire = {}
        
        for k in range(m):
            e = expressions[k]
            if e in expr_to_wire:
                outputs.append(expr_to_wire[e])
            else:
                if e == 0:
                    wire_out = n
                else:
                    bits = []
                    for w_id in available_wires:
                        if e & (1 << w_id):
                            bits.append(w_id)
                    
                    if len(bits) == 1:
                        wire_out = bits[0]
                    else:
                        # Balanced tree synthesis for remaining bits
                        # This minimizes depth
                        if len(bits) == 2:
                            final_gates.append([bits[0], bits[1]])
                            wire_out = next_wire
                            next_wire += 1
                        else:
                            # Recursive balanced split
                            def build_tree(w_list):
                                nonlocal next_wire
                                nonlocal final_gates
                                if len(w_list) == 1:
                                    return w_list[0]
                                mid = len(w_list) // 2
                                left = build_tree(w_list[:mid])
                                right = build_tree(w_list[mid:])
                                final_gates.append([left, right])
                                w_out = next_wire
                                next_wire += 1
                                return w_out
                            
                            wire_out = build_tree(bits)
                
                expr_to_wire[e] = wire_out
                outputs.append(wire_out)
        
        return {'gates': final_gates, 'outputs': outputs}

    # Run strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _baseline()
    if _verify(baseline):
        best_result = baseline
        best_gate_count = len(baseline['gates'])
        
    # 2. Enhanced Greedy CSE
    if time.time() < deadline - 15.0:
        try:
            res = _greedy_cse_enhanced()
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
    matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    res = synthesize(matrix, 16)
    print(res)
