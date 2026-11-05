"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like of 0/1, shape (m, n)
        depth_limit: maximum circuit depth
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b]) and 'outputs' (list of wire ids)
    """
    start_time = time.time()
    DEADLINE = 80.0
    
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    if n <= 16:
        result = _synthesize_small_n(matrix, n, m, depth_limit, start_time, DEADLINE)
    else:
        result = _synthesize_heuristic(matrix, n, m, depth_limit, start_time, DEADLINE)
    
    return result


def _synthesize_small_n(matrix, n, m, depth_limit, start_time, deadline):
    """For n <= 16, use subset-based sharing to minimize gate count."""
    
    row_sets = []
    for i in range(m):
        s = frozenset(j for j in range(n) if matrix[i, j] == 1)
        row_sets.append(s)
    
    wire_to_set = {}
    set_to_wire = {}
    
    for i in range(n):
        wire_to_set[i] = frozenset([i])
        set_to_wire[frozenset([i])] = i
    wire_to_set[n] = frozenset()
    set_to_wire[frozenset()] = n
    
    gates = []
    outputs = []
    
    def get_or_compute_xor(set_a, set_b):
        result_set = set_a.symmetric_difference(set_b)
        if result_set in set_to_wire:
            return set_to_wire[result_set]
        wa = set_to_wire[set_a]
        wb = set_to_wire[set_b]
        if wa == wb:
            return n
        new_wid = n + 1 + len(gates)
        gates.append([wa, wb])
        wire_to_set[new_wid] = result_set
        set_to_wire[result_set] = new_wid
        return new_wid
    
    for i in range(m):
        target = row_sets[i]
        
        if target in set_to_wire:
            outputs.append(set_to_wire[target])
            continue
        
        found = False
        for existing_set in list(set_to_wire.keys()):
            complement = existing_set.symmetric_difference(target)
            if complement in set_to_wire:
                w1 = set_to_wire[existing_set]
                w2 = set_to_wire[complement]
                if w1 != w2:
                    new_wid = get_or_compute_xor(existing_set, complement)
                    outputs.append(new_wid)
                    found = True
                    break
        
        if not found:
            if len(target) == 0:
                outputs.append(n)
            elif len(target) == 1:
                outputs.append(set_to_wire[target])
            else:
                inputs_sorted = sorted(target)
                prev_set = frozenset([inputs_sorted[0]])
                prev_wid = set_to_wire[prev_set]
                for inp in inputs_sorted[1:]:
                    inp_set = frozenset([inp])
                    prev_wid = get_or_compute_xor(prev_set, inp_set)
                    prev_set = wire_to_set[prev_wid]
                outputs.append(prev_wid)
    
    return {'gates': gates, 'outputs': outputs}


def _synthesize_heuristic(matrix, n, m, depth_limit, start_time, deadline):
    gates = []
    outputs = []
    n_zero = n
    
    for i in range(m):
        row = matrix[i]
        inputs = [j for j in range(n) if row[j] == 1]
        if len(inputs) == 0:
            outputs.append(n_zero)
        elif len(inputs) == 1:
            outputs.append(inputs[0])
        else:
            prev = inputs[0]
            for inp in inputs[1:]:
                gates.append([prev, inp])
                prev = n_zero + 1 + len(gates) - 1
            outputs.append(prev)
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
