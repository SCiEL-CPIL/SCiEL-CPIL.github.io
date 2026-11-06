import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    
    m, n = len(matrix), len(matrix[0])
    
    # Convert to list of frozensets
    rows = []
    for i in range(m):
        row_set = frozenset(j for j in range(n) if matrix[i][j] == 1)
        rows.append(row_set)
    
    gates = []
    wire_map = {}  # frozenset -> wire index
    
    # Initialize
    for j in range(n):
        wire_map[frozenset([j])] = j
    wire_map[frozenset()] = n  # zero wire
    
    def get_wire_for_set(s):
        """Get wire for set s, computing if needed. Returns wire index."""
        nonlocal gates
        if s in wire_map:
            return wire_map[s]
        
        s_list = sorted(s)
        if len(s_list) == 0:
            w = n
        elif len(s_list) == 1:
            w = s_list[0]
        else:
            # Find best proper subset already computed
            best_sub = None
            for sub in wire_map:
                if sub < s and len(sub) > (len(best_sub) if best_sub else 0):
                    best_sub = sub
            
            if best_sub is not None and len(best_sub) > 0:
                rem = s - best_sub
                rem_wire = get_wire_for_set(rem)
                sub_wire = wire_map[best_sub]
                w = n + 1 + len(gates)
                gates.append([sub_wire, rem_wire])
            else:
                # Compute from scratch using balanced tree
                w = _compute_xor_balanced(gates, s_list, n)
        
        wire_map[s] = w
        return w
    
    def _compute_xor_balanced(gates, lst, n):
        """Compute XOR of list of wires using balanced tree."""
        if len(lst) == 1:
            return lst[0]
        # Pairwise XOR
        mid = len(lst) // 2
        left = _compute_xor_balanced(gates, lst[:mid], n)
        right = _compute_xor_balanced(gates, lst[mid:], n)
        w = n + 1 + len(gates)
        gates.append([left, right])
        return w
    
    outputs = [0] * m
    # Process rows - maybe sort by size for better sharing
    row_order = sorted(range(m), key=lambda i: len(rows[i]))
    for idx in row_order:
        outputs[idx] = get_wire_for_set(rows[idx])
    
    return {'gates': gates, 'outputs': outputs}
