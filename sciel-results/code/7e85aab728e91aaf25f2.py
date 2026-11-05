"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit for the given binary matrix.
    
    Uses a dynamic greedy common-subexpression elimination (CSE) approach.
    It iteratively identifies pairs of wires that frequently co-occur in output
    expressions, creates new intermediate wires for their XOR, and updates the
    output expressions to use these new wires. This minimizes the total gate count
    by sharing sub-expressions.
    
    Args:
        matrix: 2D binary array where matrix[i][j] = 1 means output i includes input j
        depth_limit: maximum circuit depth (we aim for depth <= 16)
    
    Returns:
        dict with 'gates' (list of [wire1, wire2] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    DEADLINE = start_time + 50.0  # Safety margin before 90s kill
    
    # Convert to numpy for efficiency
    if isinstance(matrix, np.ndarray):
        mat = matrix
    else:
        mat = np.array(matrix)
        
    if mat.size == 0:
        return {'gates': [], 'outputs': []}
        
    n_outputs, n_inputs = mat.shape
    
    # Input wires: 0..n_inputs-1
    # Constant-zero wire: n_inputs
    # Gate k produces wire n_inputs + 1 + k
    
    # Use bitmasks for expressions since n_inputs is small (<=16)
    # wire_val[w] = bitmask of input variables for wire w
    wire_val = [1 << i for i in range(n_inputs)]
    wire_val.append(0)  # Wire n_inputs is 0
    
    # Map from value to wire index for O(1) lookup
    wire_map = {val: idx for idx, val in enumerate(wire_val)}
    
    # Each output is represented as a set of wire indices that XOR to give the output
    # Initially, each output is the set of input wires it depends on
    out_sets = []
    for i in range(n_outputs):
        s = set()
        row = mat[i]
        for j in range(n_inputs):
            if row[j] == 1:
                s.add(j)
        out_sets.append(s)
        
    gates = []
    next_wire_idx = n_inputs + 1  # First gate output wire index
    
    # Greedy CSE loop
    while True:
        if time.time() > DEADLINE - 5:  # Leave time for finalization
            break
            
        # Check if all outputs are single wires
        all_single = True
        for s in out_sets:
            if len(s) > 1:
                all_single = False
                break
        if all_single:
            break
            
        # Count how many outputs contain each pair of wires
        pair_count = defaultdict(int)
        for s in out_sets:
            if len(s) <= 1:
                continue
            items = list(s)
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    i = items[a]
                    j = items[b]
                    pair_count[(i, j)] += 1
        
        # Find the pair with max count
        best_benefit = 0
        best_i = -1
        best_j = -1
        
        for (i, j), cnt in pair_count.items():
            benefit = cnt - 1
            if benefit > best_benefit:
                best_benefit = benefit
                best_i = i
                best_j = j
        
        if best_benefit > 0:
            i, j = best_i, best_j
        else:
            # No pair has positive benefit. Force reduction on the largest set.
            max_idx = 0
            max_size = 0
            for idx, s in enumerate(out_sets):
                if len(s) > max_size:
                    max_size = len(s)
                    max_idx = idx
                    
            if max_size <= 1:
                break
                
            s = out_sets[max_idx]
            items = list(s)
            i, j = items[0], items[1]
        
        # Create new wire k = i XOR j
        w_val = wire_val[i] ^ wire_val[j]
        
        if w_val in wire_map:
            k = wire_map[w_val]
        else:
            k = next_wire_idx
            wire_val.append(w_val)
            wire_map[w_val] = k
            gates.append([i, j])
            next_wire_idx += 1
        
        # Update outputs: replace i, j with k in all sets containing both
        for s in out_sets:
            if i in s and j in s:
                s.remove(i)
                s.remove(j)
                if k in s:
                    # i ^ j ^ k = k ^ k = 0, so k cancels out
                    s.remove(k)
                else:
                    s.add(k)

    # Finalize: Ensure all outputs are single wires
    for s in out_sets:
        while len(s) > 1:
            if time.time() > DEADLINE - 1:
                break
            items = list(s)
            i, j = items[0], items[1]
            w_val = wire_val[i] ^ wire_val[j]
            
            if w_val in wire_map:
                k = wire_map[w_val]
            else:
                k = next_wire_idx
                wire_val.append(w_val)
                wire_map[w_val] = k
                gates.append([i, j])
                next_wire_idx += 1
                
            s.remove(i)
            s.remove(j)
            if k in s:
                s.remove(k)
            else:
                s.add(k)
                
            # Safety check to avoid infinite loops
            if len(s) >= len(items):
                break

    # Determine output wires
    output_wires = []
    for s in out_sets:
        if len(s) == 0:
            output_wires.append(n_inputs)  # Zero wire
        elif len(s) == 1:
            output_wires.append(next(iter(s)))
        else:
            # Fallback: chain remaining wires
            items = list(s)
            current = items[0]
            for w in items[1:]:
                w_val = wire_val[current] ^ wire_val[w]
                if w_val in wire_map:
                    k = wire_map[w_val]
                else:
                    k = next_wire_idx
                    wire_val.append(w_val)
                    wire_map[w_val] = k
                    gates.append([current, w])
                    next_wire_idx += 1
                current = k
            output_wires.append(current)

    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
