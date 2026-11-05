"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

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
    deadline = start_time + 58.0  # Safety margin before 60s limit

    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Check for zero rows
    row_popcounts = A.sum(axis=1)
    zero_rows = set(np.where(row_popcounts == 0)[0])
    
    # Represent each row as a set of wire indices
    row_wires = []
    for i in range(n_rows):
        if i in zero_rows:
            row_wires.append(set())
        else:
            row_wires.append(set(np.where(A[i] == 1)[0].tolist()))
    
    gates = []
    next_wire = zero_wire + 1
    
    # Map from wire to its underlying set of input indices (for tracking)
    wire_set = {}
    for i in range(n_inputs):
        wire_set[i] = frozenset([i])
    wire_set[zero_wire] = frozenset()
    
    def elapsed_ok():
        return time.time() < deadline
    
    # Greedy: repeatedly find the pair (a,b) that maximizes benefit
    # Benefit = (number of rows containing both a and b, where row size >= 2) - 1
    # We also need a != b and both wires exist
    
    max_iter = 50000
    for _ in range(max_iter):
        if not elapsed_ok():
            break
        
        # Count co-occurrence of pairs in rows with size >= 2
        pair_count = defaultdict(int)
        for i in range(n_rows):
            rs = row_wires[i]
            if len(rs) < 2:
                continue
            elems = list(rs)
            for x in range(len(elems)):
                for y in range(x + 1, len(elems)):
                    a, b = elems[x], elems[y]
                    if a > b:
                        a, b = b, a
                    pair_count[(a, b)] += 1
        
        if not pair_count:
            break
        
        # Find best pair
        best_pair = None
        best_benefit = 0
        for (a, b), cnt in pair_count.items():
            if cnt >= 2:
                benefit = cnt - 1
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = (a, b)
        
        if best_pair is None:
            break
        
        a, b = best_pair
        new_wire = next_wire
        next_wire += 1
        
        # Compute the set for the new wire
        new_set = wire_set[a] ^ wire_set[b]
        wire_set[new_wire] = new_set
        gates.append([a, b])
        
        # Update all rows containing both a and b
        for i in range(n_rows):
            rs = row_wires[i]
            if a in rs and b in rs:
                rs.discard(a)
                rs.discard(b)
                rs.add(new_wire)
    
    # Build outputs
    outputs = []
    for i in range(n_rows):
        rs = row_wires[i]
        if len(rs) == 0:
            outputs.append(zero_wire)
        elif len(rs) == 1:
            outputs.append(list(rs)[0])
        else:
            elems = list(rs)
            current = elems[0]
            for e in elems[1:]:
                gates.append([current, e])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
