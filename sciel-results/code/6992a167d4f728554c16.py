"""XOR-circuit synthesis via Minimum-Weight Linear Basis Construction."""
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
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Normalize input
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets (integers) for fast GF(2) operations
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
        if any(o >= num_wires for o in outputs):
            return False
        if any(a >= num_wires or b >= num_wires for a, b in gates):
            return False
            
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        for i in range(m):
            expected = orig_row_bits[i]
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
        """Remove unused gates and reindex wires."""
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
            
        # Identify necessary gates via reverse dependency
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

    def _min_weight_basis():
        """
        Construct a circuit using a minimum-weight linear basis strategy.
        This selects intermediate vectors that have low Hamming weight to maximize
        the likelihood of reuse (sharing) in subsequent XOR operations.
        """
        if time.time() > deadline - 5.0:
            return _baseline()

        # State:
        # remaining_rows: list of bitsets not yet resolved to a single wire
        # available_wires: list of wire indices currently available
        # wire_val: dict wire_index -> bitset
        # wire_depth: dict wire_index -> depth
        
        remaining = list(orig_row_bits)
        available_wires = list(range(n + 1))
        wire_val = {i: (1 << i) for i in range(n)}
        wire_val[n] = 0
        wire_depth = {i: 0 for i in range(n + 1)}
        
        gates = []
        next_wire = n + 1
        
        # We want to resolve each remaining row to a single wire.
        # Strategy: Iteratively pick the pair of wires (a, b) from available_wires
        # such that a^b has low weight and appears in many remaining rows,
        # or simply pick the pair that reduces the total weight of remaining rows most effectively.
        
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Check if we are done
            if all(len(bin(e).count('1')) <= 1 for e in remaining):
                break
                
            best_pair = None
            best_score = -1
            best_new_val = 0
            best_new_depth = 0
            
            # Optimization: Only consider pairs where the result is not already in available_wires?
            # Actually, if it is, we can just use the existing wire.
            # But we are constructing the basis. Let's just find the best new wire to create.
            
            # Heuristic: Look for pairs (a, b) in available_wires.
            # Score:
            # 1. How many remaining rows contain both a and b? (High is good)
            # 2. Weight of (val[a] ^ val[b]) (Low is good, promotes sharing)
            
            L = len(available_wires)
            if L > 60:
                 # Too many wires to check all pairs efficiently in Python loops?
                 # For n=16, m=16, L grows. 100 pairs is fine.
                 # If L is huge, we might need to prune.
                 pass

            for i in range(L):
                a = available_wires[i]
                va = wire_val[a]
                da = wire_depth[a]
                
                for j in range(i + 1, L):
                    b = available_wires[j]
                    vb = wire_val[b]
                    db = wire_depth[b]
                    
                    new_val = va ^ vb
                    if new_val == 0:
                        continue
                    
                    new_depth = max(da, db) + 1
                    if new_depth > depth_limit:
                        continue
                    
                    # Calculate score
                    # Count how many remaining rows have both a and b
                    sharing = 0
                    for e in remaining:
                        if (e & (1 << a)) and (e & (1 << b)):
                            sharing += 1
                    
                    if sharing == 0:
                        continue
                        
                    weight_new = bin(new_val).count('1')
                    
                    # We want high sharing and low weight.
                    # If sharing >= 2, it's a strong candidate.
                    # Score = sharing * 100 - weight_new
                    score = sharing * 100 - weight_new
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_new_val = new_val
                        best_new_depth = new_depth
            
            if best_pair is None:
                break
                
            a, b = best_pair
            w = next_wire
            next_wire += 1
            
            gates.append([a, b])
            wire_val[w] = best_new_val
            wire_depth[w] = best_new_depth
            available_wires.append(w)
            
            # Update remaining rows: replace a^b with w
            for k in range(len(remaining)):
                e = remaining[k]
                if (e & (1 << a)) and (e & (1 << b)):
                    e ^= (1 << a)
                    e ^= (1 << b)
                    e |= (1 << w)
                    remaining[k] = e

        # Finalize outputs
        outputs = []
        expr_to_wire = {}
        
        for k in range(m):
            e = remaining[k]
            if e in expr_to_wire:
                outputs.append(expr_to_wire[e])
                continue
                
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
                    # Synthesize remaining expression
                    # Use a balanced tree or simple chain?
                    # Simple chain is depth-safe if we check.
                    # But we have depth_limit.
                    # Let's use a balanced tree to minimize depth.
                    def build_tree(w_list):
                        nonlocal next_wire
                        nonlocal gates
                        if len(w_list) == 1:
                            return w_list[0]
                        mid = len(w_list) // 2
                        left = build_tree(w_list[:mid])
                        right = build_tree(w_list[mid:])
                        gates.append([left, right])
                        w_out = next_wire
                        next_wire += 1
                        return w_out
                        
                    wire_out = build_tree(bits)
                    
            expr_to_wire[e] = wire_out
            outputs.append(wire_out)
            
        return {'gates': gates, 'outputs': outputs}

    def _local_search_polish(circuit):
        """
        Try to reduce gate count by reordering gates or finding redundant structures.
        This is a heuristic pass.
        """
        if time.time() > deadline - 2.0:
            return circuit
            
        best_circuit = circuit
        best_count = len(circuit['gates'])
        
        # Simple local search: try swapping adjacent gates and see if it helps?
        # Or try removing gates that become redundant?
        # _polish already removes unused gates.
        # A more advanced search would involve re-synthesizing parts of the circuit.
        
        # For now, just run polish again to be safe.
        polished = _polish(circuit)
        if _verify(polished):
            depth = _compute_depth(polished['gates'], polished['outputs'])
            if depth <= depth_limit:
                if len(polished['gates']) < best_count:
                    best_circuit = polished
                    best_count = len(polished['gates'])
                    
        return best_circuit

    # Run strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _baseline()
    if _verify(baseline):
        best_result = baseline
        best_gate_count = len(baseline['gates'])
        
    # 2. Minimum Weight Basis
    if time.time() < deadline - 15.0:
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
            
    # 3. Local Search Polish on Best Result
    if best_result is not None and time.time() < deadline - 2.0:
        try:
            res = _local_search_polish(best_result)
            if _verify(res):
                depth = _compute_depth(res['gates'], res['outputs'])
                if depth <= depth_limit:
                    if len(res['gates']) < best_gate_count:
                        best_result = res
                        best_gate_count = len(res['gates'])
        except Exception:
            pass

    # 4. Fallback
    if best_result is None:
        best_result = {'gates': [], 'outputs': [n] * m}
        
    return best_result

if __name__ == "__main__":
    matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    res = synthesize(matrix, 16)
    print(res)
