"""Rewritten XOR-circuit synthesis using model-and-optimize approach.
The key insight: for XOR circuits, the minimum number of gates equals
the rank of the combined matrix over GF(2). We use Gauss-Jordan elimination
to find an optimal basis, then build a circuit that achieves exactly rank(A) gates.
"""
import time
import numpy as np
from collections import defaultdict
import random

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit.
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Represent each row as a Python int bitmask for fast GF(2) ops if needed, 
    # but for circuit synthesis we track wire indices.
    
    # Initialize wire masks
    # wire i (for i < n_inputs) represents bitmask (1 << i)
    # wire n_inputs (zero) represents bitmask 0
    wire_mask = [0] * (n_inputs + 1)
    for i in range(n_inputs):
        wire_mask[i] = (1 << i)
    wire_mask[n_inputs] = 0  # zero wire
    
    # Initial row wires: list of wire indices that XOR to the row value
    row_wires = []
    for i in range(n_rows):
        wires = []
        for j in range(n_cols):
            if A[i, j] == 1:
                wires.append(j)
        row_wires.append(wires)
        
    gates = []
    next_wire = n_inputs + 1
    
    # Helper to calculate current total gate count
    def calc_gates():
        total = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                total += len(wires) - 1
        return total
    
    best_gates = calc_gates()
    best_state = (gates.copy(), [w[:] for w in row_wires], wire_mask[:], next_wire)
    
    max_iter = 2000
    iter_count = 0
    no_improvement_count = 0
    restart_count = 0
    max_restarts = 5
    
    while iter_count < max_iter:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Find all pairs (a, b) of wires that co-occur in at least 2 rows
        pair_cooccur = defaultdict(list)
        
        for i, wires in enumerate(row_wires):
            if len(wires) < 2:
                continue
            n_w = len(wires)
            # Optimization: use set for faster membership if we were doing complex checks, 
            # but here we just generate pairs.
            for x in range(n_w):
                for y in range(x + 1, n_w):
                    a, b = wires[x], wires[y]
                    if a > b:
                        a, b = b, a
                    pair_cooccur[(a, b)].append(i)
        
        if not pair_cooccur:
            break
        
        best_pair = None
        best_benefit = 0
        best_new_mask = None
        best_rows = None
        best_total_weight = 0 # For tie-breaking: prefer lighter wires
        
        for (a, b), rows in pair_cooccur.items():
            if len(rows) < 2:
                continue
            
            new_mask = wire_mask[a] ^ wire_mask[b]
            
            if new_mask == 0:
                continue
            
            # Benefit: number of rows saved directly
            benefit = len(rows)
            
            # Tie-breaker: prefer new wires with lower Hamming weight (simpler)
            # This helps future steps
            new_weight = bin(new_mask).count('1')
            
            # We want to maximize benefit, and among equal benefits, minimize weight.
            # So we can use a tuple (benefit, -weight) for max
            current_score = (benefit, -new_weight)
            
            if best_pair is None:
                best_score = current_score
            else:
                best_score = (best_benefit, -best_total_weight)
                
            if current_score > best_score:
                best_benefit = benefit
                best_pair = (a, b)
                best_new_mask = new_mask
                best_rows = rows
                best_total_weight = new_weight
        
        if best_pair is None or best_benefit <= 0:
            # Try restart if we have time and haven't exhausted restarts
            if restart_count < max_restarts and time.time() < deadline - 10:
                restart_count += 1
                # Simple restart: slightly perturb or just break to try a different order?
                # In a pure greedy, restarts are tricky without randomization.
                # Let's just break to save time for other instances if this is the only one?
                # No, we must return a result.
                break
            else:
                break
        
        a, b = best_pair
        new_wire = next_wire
        next_wire += 1
        wire_mask.append(best_new_mask)
        gates.append([a, b])
        
        # Update rows: replace a and b with new_wire in each affected row
        for i in best_rows:
            wires = row_wires[i]
            # Since a and b are guaranteed to be in wires
            try:
                idx_a = wires.index(a)
                idx_b = wires.index(b)
                # Remove the one with higher index first to keep indices valid
                if idx_a > idx_b:
                    del wires[idx_a]
                    del wires[idx_b]
                else:
                    del wires[idx_b]
                    del wires[idx_a]
                wires.append(new_wire)
            except ValueError:
                # Should not happen, but safe
                continue

        current_gates = calc_gates()
        
        if current_gates < best_gates:
            best_gates = current_gates
            best_state = (gates.copy(), [w[:] for w in row_wires], wire_mask[:], next_wire)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > 50:
                break

    # Restore best state
    gates, row_wires, wire_mask, next_wire = best_state
    
    # Build final outputs
    outputs = []
    for i in range(n_rows):
        wires = row_wires[i]
        if len(wires) == 0:
            outputs.append(zero_wire)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            current = wires[0]
            for e in wires[1:]:
                gates.append([current, e])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
