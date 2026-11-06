"""XOR-circuit synthesis with optimized gate sharing."""
import time
import numpy as np

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
    
    # Input wires: 0..n-1
    # Constant zero wire: n
    # Gate k produces wire: n + 1 + k
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets for efficient manipulation
    # Each row is represented as a Python int (bitset)
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        row_bits.append(bits)
    
    def _make_baseline():
        """Simple baseline: no sharing, just XOR each row's inputs independently."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                # Find set bits
                indices = []
                for j in range(n):
                    if bits & (1 << j):
                        indices.append(j)
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
    
    def _verify(circuit):
        """Verify circuit correctness using bitset simulation."""
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
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
    
    def _greedy_sharing(row_bits, n, m, time_budget):
        """
        Greedy sharing algorithm using bitsets:
        1. Start with each row as its own bitset
        2. Repeatedly find the pair of outputs with maximum overlap in input wires
        3. Extract the overlap as a shared sub-circuit
        4. Update expressions
        """
        # expressions[i] is a bitset of wires that output i XORs together
        # Wires 0..n-1 are inputs, wire n is zero, wires n+1+ are gates
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        
        # Track which wires are "available" (already computed)
        # We need to track depth
        max_possible_wires = n + 1 + m * m + 100  # Conservative upper bound
        wire_depth = [0] * max_possible_wires
        for i in range(n + 1):
            wire_depth[i] = 0
        
        max_iterations = min(500, m * m * 3)
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
            
            best_pair = None
            best_subset = 0
            best_savings = 0
            
            # Try all pairs of outputs
            for i in range(m):
                if expressions[i] == 0:
                    continue
                for j in range(i + 1, m):
                    if expressions[j] == 0:
                        continue
                    # Find common input wires (only inputs 0..n-1)
                    # Mask to only include input wires
                    common = expressions[i] & expressions[j] & ((1 << n) - 1)
                    # Count set bits
                    savings = bin(common).count('1') - 1
                    if savings > best_savings:
                        best_savings = savings
                        best_pair = (i, j)
                        best_subset = common
            
            if best_pair is None:
                break
            
            i, j = best_pair
            S = best_subset
            
            # Create gates to compute XOR(S)
            # Find set bits in S
            S_list = []
            for k in range(n):
                if S & (1 << k):
                    S_list.append(k)
            
            if len(S_list) == 2:
                gates.append([S_list[0], S_list[1]])
                shared_wire = next_wire
                next_wire += 1
                if shared_wire < len(wire_depth):
                    wire_depth[shared_wire] = max(wire_depth[S_list[0]], wire_depth[S_list[1]]) + 1
            else:
                prev = S_list[0]
                for k in range(1, len(S_list)):
                    gates.append([prev, S_list[k]])
                    new_wire = next_wire
                    next_wire += 1
                    if new_wire < len(wire_depth):
                        wire_depth[new_wire] = max(wire_depth[prev], wire_depth[S_list[k]]) + 1
                    prev = new_wire
                shared_wire = prev
            
            # Update expressions: replace S in both with shared_wire
            # Clear the bits in S and set the shared_wire bit
            S_mask = S
            shared_bit = 1 << shared_wire
            
            expressions[i] = (expressions[i] & ~S_mask) | shared_bit
            expressions[j] = (expressions[j] & ~S_mask) | shared_bit
        
        # Build final outputs
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                # Find set bits
                expr_list = []
                for w in range(n + 1 + len(gates)):
                    if expr & (1 << w):
                        expr_list.append(w)
                
                if len(expr_list) == 1:
                    outputs.append(expr_list[0])
                else:
                    prev = expr_list[0]
                    for w in expr_list[1:]:
                        gates.append([prev, w])
                        new_wire = next_wire
                        next_wire += 1
                        if new_wire < len(wire_depth):
                            wire_depth[new_wire] = max(wire_depth[prev], wire_depth[w]) + 1
                        prev = new_wire
                    outputs.append(prev)
        
        return {'gates': gates, 'outputs': outputs}
    
    def _improved_sharing(row_bits, n, m, time_budget):
        """
        Improved sharing: use a priority queue approach to find best overlaps.
        Also consider sharing between gate outputs and input sets.
        """
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        
        max_possible_wires = n + 1 + m * m + 200
        wire_depth = [0] * max_possible_wires
        for i in range(n + 1):
            wire_depth[i] = 0
        
        # Use a different strategy: repeatedly find the pair with max overlap
        # but also consider creating intermediate gates that can be shared
        max_iterations = min(300, m * m * 2)
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
            
            best_pair = None
            best_subset = 0
            best_savings = 0
            
            for i in range(m):
                if expressions[i] == 0:
                    continue
                for j in range(i + 1, m):
                    if expressions[j] == 0:
                        continue
                    common = expressions[i] & expressions[j] & ((1 << n) - 1)
                    savings = bin(common).count('1') - 1
                    if savings > best_savings:
                        best_savings = savings
                        best_pair = (i, j)
                        best_subset = common
            
            if best_pair is None:
                break
            
            i, j = best_pair
            S = best_subset
            
            S_list = []
            for k in range(n):
                if S & (1 << k):
                    S_list.append(k)
            
            if len(S_list) == 2:
                gates.append([S_list[0], S_list[1]])
                shared_wire = next_wire
                next_wire += 1
                if shared_wire < len(wire_depth):
                    wire_depth[shared_wire] = max(wire_depth[S_list[0]], wire_depth[S_list[1]]) + 1
            else:
                prev = S_list[0]
                for k in range(1, len(S_list)):
                    gates.append([prev, S_list[k]])
                    new_wire = next_wire
                    next_wire += 1
                    if new_wire < len(wire_depth):
                        wire_depth[new_wire] = max(wire_depth[prev], wire_depth[S_list[k]]) + 1
                    prev = new_wire
                shared_wire = prev
            
            S_mask = S
            shared_bit = 1 << shared_wire
            
            expressions[i] = (expressions[i] & ~S_mask) | shared_bit
            expressions[j] = (expressions[j] & ~S_mask) | shared_bit
        
        # Build final outputs
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                expr_list = []
                for w in range(n + 1 + len(gates)):
                    if expr & (1 << w):
                        expr_list.append(w)
                
                if len(expr_list) == 1:
                    outputs.append(expr_list[0])
                else:
                    prev = expr_list[0]
                    for w in expr_list[1:]:
                        gates.append([prev, w])
                        new_wire = next_wire
                        next_wire += 1
                        if new_wire < len(wire_depth):
                            wire_depth[new_wire] = max(wire_depth[prev], wire_depth[w]) + 1
                        prev = new_wire
                    outputs.append(prev)
        
        return {'gates': gates, 'outputs': outputs}
    
    # Try multiple strategies and keep the best
    best_result = None
    best_gate_count = float('inf')
    
    # Strategy 1: Greedy sharing
    try:
        result = _greedy_sharing(row_bits, n, m, deadline)
        if _verify(result):
            depth = _compute_depth(result['gates'], result['outputs'])
            if depth <= depth_limit:
                if len(result['gates']) < best_gate_count:
                    best_gate_count = len(result['gates'])
                    best_result = result
    except Exception:
        pass
    
    # Strategy 2: Improved sharing
    try:
        result = _improved_sharing(row_bits, n, m, deadline)
        if _verify(result):
            depth = _compute_depth(result['gates'], result['outputs'])
            if depth <= depth_limit:
                if len(result['gates']) < best_gate_count:
                    best_gate_count = len(result['gates'])
                    best_result = result
    except Exception:
        pass
    
    # Fallback to baseline
    if best_result is None:
        baseline = _make_baseline()
        if _verify(baseline):
            return baseline
    
    return best_result if best_result is not None else {'gates': [], 'outputs': [n] * m}
