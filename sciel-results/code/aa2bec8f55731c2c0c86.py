"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq
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
    
    # Map from wire to its underlying frozenset of input indices
    wire_set = {}
    for i in range(n_inputs):
        wire_set[i] = frozenset([i])
    wire_set[zero_wire] = frozenset()
    
    # Depth tracking: depth of each wire
    wire_depth = {}
    for i in range(n_inputs):
        wire_depth[i] = 0
    wire_depth[zero_wire] = 0
    
    def elapsed_ok():
        return time.time() < deadline
    
    def try_greedy(seed=None):
        """Run greedy synthesis with a given random seed for tie-breaking."""
        if seed is not None:
            random.seed(seed)
        
        gates = []
        next_wire = zero_wire + 1
        
        # Local copies
        rw = [set(s) for s in row_wires]
        ws = dict(wire_set)
        wd = dict(wire_depth)
        
        max_iter = 20000
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            # Count co-occurrence of pairs in rows with size >= 2
            pair_count = defaultdict(int)
            for i in range(n_rows):
                rs = rw[i]
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a > b:
                            a, b = b, a
                        # Skip if either is zero wire
                        if a == zero_wire or b == zero_wire:
                            continue
                        # Skip if they represent the same set (XOR would be 0)
                        if ws[a] == ws[b]:
                            continue
                        pair_count[(a, b)] += 1
            
            if not pair_count:
                break
            
            # Find best pair with tie-breaking
            best_pair = None
            best_benefit = 0
            best_depth = 0
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                benefit = cnt - 1
                # New depth would be max(depth[a], depth[b]) + 1
                new_depth = max(wd[a], wd[b]) + 1
                if new_depth > depth_limit:
                    continue
                if benefit > best_benefit or (benefit == best_benefit and new_depth < best_depth):
                    best_benefit = benefit
                    best_depth = new_depth
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_wire = next_wire
            next_wire += 1
            
            # Compute the set for the new wire
            new_set = ws[a] ^ ws[b]
            if len(new_set) == 0:
                # This would produce zero, skip
                continue
            ws[new_wire] = new_set
            wd[new_wire] = max(wd[a], wd[b]) + 1
            gates.append([a, b])
            
            # Update all rows containing both a and b
            for i in range(n_rows):
                rs = rw[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
        
        # Build outputs
        outputs = []
        for i in range(n_rows):
            rs = rw[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                # Sort by depth to minimize overall depth
                elems.sort(key=lambda w: wd.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    # Check depth
                    new_depth = max(wd.get(current, 0), wd.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        # Can't add more gates due to depth, just output as is
                        break
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    ws[current] = ws.get(current, frozenset())  # placeholder
                    wd[current] = new_depth
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
    
    # Try multiple seeds and keep the best
    best_result = None
    best_gate_count = float('inf')
    
    # First try deterministic (seed=0)
    result = try_greedy(seed=0)
    if result is not None:
        best_result = result
        best_gate_count = len(result['gates'])
    
    # Try a few more seeds if time allows
    for seed in [1, 42, 123, 7, 99]:
        if not elapsed_ok():
            break
        result = try_greedy(seed=seed)
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    if best_result is None:
        # Fallback: independent row synthesis
        gates = []
        next_wire = zero_wire + 1
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
    
    return best_result

# EVOLVE-BLOCK-END
