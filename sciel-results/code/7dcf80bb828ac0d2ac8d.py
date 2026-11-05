"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit for the given binary matrix.
    
    Each row represents an output that is the XOR of certain input variables.
    We find common subexpressions to minimize total gate count.
    
    Args:
        matrix: 2D binary array where matrix[i][j] = 1 means output i includes input j
        depth_limit: maximum circuit depth (we aim for depth <= 16)
    
    Returns:
        dict with 'gates' (list of [wire1, wire2] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m
    
    output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    def popcount(m):
        return bin(m).count('1')
    
    # Available expressions: mask -> wire
    expr_to_wire = {}
    for i in range(n_inputs):
        expr_to_wire[(1 << i)] = i
    expr_to_wire[0] = n_inputs
    
    next_wire = n_inputs + 1
    gates = []
    
    # Precompute bits for each output
    output_bits = []
    for mask in output_masks:
        bits = []
        m = mask
        j = 0
        while m:
            if m & 1:
                bits.append(j)
            m >>= 1
            j += 1
        output_bits.append(bits)
    
    # Count frequency of subsets of size 2-4
    subset_freq = defaultdict(int)
    for i in range(n_outputs):
        bits = output_bits[i]
        n = len(bits)
        for a in range(n):
            for b in range(a+1, n):
                subset_freq[(bits[a], bits[b])] += 1
                for c in range(b+1, n):
                    subset_freq[(bits[a], bits[b], bits[c])] += 1
                    for d in range(c+1, n):
                        subset_freq[(bits[a], bits[b], bits[c], bits[d])] += 1
    
    # Sort by benefit
    candidates = []
    for subset, freq in subset_freq.items():
        k = len(subset)
        if k >= 2 and freq >= 2:
            benefit = (k - 1) * (freq - 1)
            if benefit > 0:
                candidates.append((benefit, subset))
    
    candidates.sort(key=lambda x: -x[0])
    
    remaining_masks = list(output_masks)
    output_wires_used = [[] for _ in range(n_outputs)]
    
    for benefit, subset in candidates:
        if time.time() > DEADLINE:
            break
        
        expr = 0
        for bit in subset:
            expr |= (1 << bit)
        
        if expr in expr_to_wire:
            continue
        
        # Check if still beneficial
        count = sum(1 for i in range(n_outputs) if (remaining_masks[i] & expr) == expr)
        if count < 2:
            continue
        
        # Create wire for this expression
        bits = list(subset)
        k = len(bits)
        
        if k == 2:
            gates.append([bits[0], bits[1]])
            w = next_wire
            next_wire += 1
        elif k == 3:
            gates.append([bits[0], bits[1]])
            t = next_wire
            next_wire += 1
            expr_to_wire[(1 << bits[0]) | (1 << bits[1])] = t
            gates.append([t, bits[2]])
            w = next_wire
            next_wire += 1
        elif k == 4:
            gates.append([bits[0], bits[1]])
            t1 = next_wire
            next_wire += 1
            expr_to_wire[(1 << bits[0]) | (1 << bits[1])] = t1
            gates.append([bits[2], bits[3]])
            t2 = next_wire
            next_wire += 1
            expr_to_wire[(1 << bits[2]) | (1 << bits[3])] = t2
            gates.append([t1, t2])
            w = next_wire
            next_wire += 1
        else:
            w = bits[0]
            for b in bits[1:]:
                gates.append([w, b])
                w = next_wire
                next_wire += 1
        
        expr_to_wire[expr] = w
        
        for i in range(n_outputs):
            if (remaining_masks[i] & expr) == expr:
                remaining_masks[i] ^= expr
                output_wires_used[i].append(w)
    
    # Assemble outputs
    output_wires = []
    for i in range(n_outputs):
        wires = list(output_wires_used[i])
        rem = remaining_masks[i]
        
        m = rem
        j = 0
        while m:
            if m & 1:
                wires.append(j)
            m >>= 1
            j += 1
        
        if not wires:
            output_wires.append(n_inputs)
        elif len(wires) == 1:
            output_wires.append(wires[0])
        else:
            level = wires
            while len(level) > 1:
                new_level = []
                for k in range(0, len(level) - 1, 2):
                    gates.append([level[k], level[k+1]])
                    new_level.append(next_wire)
                    next_wire += 1
                if len(level) % 2 == 1:
                    new_level.append(level[-1])
                level = new_level
            output_wires.append(level[0])
    
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
