import time
import numpy as np
import random
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit using Minimum-Weight Basis Construction.
    
    The strategy is to view the target rows as vectors in GF(2)^n.
    We iteratively construct a basis of "intermediate wires" (XORs of existing wires)
    that allows the target rows to be expressed with minimal total Hamming weight
    (number of terms in the XOR sum).
    
    This is a structural change from local CSE: instead of just finding pairs that
    appear in multiple targets, we look for combinations that reduce the overall
    complexity (weight) of the set of targets, effectively finding a low-weight
    generating set for the row space.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

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
    
    # Helper: Verify circuit correctness
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

    # Helper: Compute depth
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

    # Helper: Baseline circuit (no sharing, just direct XOR chains)
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

    # Helper: Polish circuit (remove unused gates, remap)
    def _polish(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        if not gates:
            return circuit
            
        num_wires = n + 1 + len(gates)
        wire_to_gate = {}
        for k in range(len(gates)):
            w = n + 1 + k
            wire_to_gate[w] = k
            
        necessary = [False] * len(gates)
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
                
        new_outputs = [wire_map.get(w, w) for w in outputs]
        return {'gates': new_gates, 'outputs': new_outputs}

    # Main Strategy: Minimum-Weight Basis Construction
    def _min_weight_basis():
        # State:
        # expressions[k] is the current bitset representation of target row k
        # using the current set of available wires.
        # available_wires is the list of wire indices we can use.
        # wire_val[w] is the GF(2) vector (bitmask of original inputs) for wire w.
        
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
        
        # We will iteratively add wires that minimize the total Hamming weight
        # of the expressions, subject to depth constraints.
        # Total Hamming weight is a proxy for gate count in the final synthesis
        # (if we just chain them), but more importantly, it identifies redundancy.
        
        max_iterations = 1000
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
                
            # Calculate current total weight
            current_total_weight = sum(bin(e).count('1') for e in expressions)
            
            best_pair = None
            best_new_weight = current_total_weight
            best_w = None
            best_depth = 0
            best_score = -1
            
            L = len(available_wires)
            
            # Optimization: If L is large, sampling might be needed, but n is small (<=16)
            # and we add at most a few wires. L stays small.
            
            for i in range(L):
                a = available_wires[i]
                va = wire_val[a]
                da = wire_depth[a]
                
                for j in range(i + 1, L):
                    b = available_wires[j]
                    vb = wire_val[b]
                    db = wire_depth[b]
                    
                    new_depth = max(da, db) + 1
                    if new_depth > depth_limit:
                        continue
                    
                    # Candidate wire value
                    vw = va ^ vb
                    
                    # Check if this wire is already available (redundant)
                    # Actually, if it's available, creating it again is useless.
                    # But we are constructing a basis, so we assume we only create new ones.
                    # We can check if vw is in {wire_val[w] for w in available_wires}
                    # This is O(L).
                    is_new = True
                    for w_existing in available_wires:
                        if wire_val[w_existing] == vw:
                            is_new = False
                            break
                    if not is_new:
                        continue
                    
                    # Calculate new total weight if we add this wire
                    # and replace a,b with w in all expressions where possible.
                    new_total_weight = 0
                    # We need to simulate the replacement.
                    # If expression E contains both a and b, it becomes E' = E ^ (1<<a) ^ (1<<b) | (1<<w_new)
                    # The weight of E' is weight(E) - 2 + 1 = weight(E) - 1.
                    # If E contains only a or only b, weight stays same.
                    # If E contains neither, weight stays same.
                    
                    # To be precise, we should check if the replacement actually reduces weight.
                    # weight(E) = popcount(E)
                    # weight(E') = popcount(E ^ (1<<a) ^ (1<<b) | (1<<w_new))
                    # Since w_new is a new bit position, it adds 1 to the count.
                    # The bits a and b are removed.
                    # So delta = 1 - (1 if (E & (1<<a)) else 0) - (1 if (E & (1<<b)) else 0)
                    
                    # We want to minimize the sum of weights.
                    # So we want to maximize the number of expressions that contain BOTH a and b.
                    
                    count_both = 0
                    for k in range(m):
                        e = expressions[k]
                        if (e & (1 << a)) and (e & (1 << b)):
                            count_both += 1
                    
                    # New total weight = current_total_weight - count_both
                    # (Because each such expression loses 1 from weight)
                    new_total_weight = current_total_weight - count_both
                    
                    # Score: Prefer larger reduction in total weight.
                    # Tie-breaker: Lower depth.
                    score = count_both * 100 - new_depth
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_new_weight = new_total_weight
                        best_w = next_wire
                        best_depth = new_depth
                        
            if best_pair is None or best_score <= 0:
                # No beneficial move found.
                # We could try "neutral" moves (count_both == 1, score ~ 1 - depth) to enable future moves?
                # But usually, if count_both < 2, it's not worth it for global weight minimization.
                # Let's break to avoid wasting time.
                break
                
            a, b = best_pair
            w = best_w
            vw = wire_val[a] ^ wire_val[b]
            
            # Commit
            gates.append([a, b])
            wire_val[w] = vw
            wire_depth[w] = best_depth
            
            # Update expressions
            for k in range(m):
                e = expressions[k]
                if (e & (1 << a)) and (e & (1 << b)):
                    e ^= (1 << a)
                    e ^= (1 << b)
                    e |= (1 << w)
                    expressions[k] = e
            
            available_wires.append(w)
            next_wire += 1
            
        # Final Synthesis
        # Now we have a set of expressions in terms of available_wires.
        # We need to synthesize these expressions into a circuit.
        # Since we minimized total weight, the expressions are "sparse".
        # We can synthesize each expression independently, but we can share
        # if two expressions are identical.
        
        outputs = []
        expr_to_wire = {}
        final_gates = []
        final_next_wire = next_wire
        
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
                        # Chain the bits
                        prev = bits[0]
                        for b_id in bits[1:]:
                            final_gates.append([prev, b_id])
                            new_w = final_next_wire
                            final_next_wire += 1
                            prev = new_w
                        wire_out = prev
                
                expr_to_wire[e] = wire_out
                outputs.append(wire_out)
        
        gates.extend(final_gates)
        
        return {'gates': gates, 'outputs': outputs}

    # Run strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _baseline()
    if _verify(baseline):
        best_result = baseline
        best_gate_count = len(baseline['gates'])
        
    # 2. Min Weight Basis
    if time.time() < deadline - 10.0:
        try:
            res = _min_weight_basis()
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
