import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 80.0
    
    mat = np.array(matrix, dtype=np.uint8)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n
    
    row_sets = []
    for i in range(n_rows):
        cols = np.nonzero(mat[i])[0]
        row_sets.append(frozenset(cols.tolist()))
    
    # Identify frequent pairs
    pair_counts = defaultdict(int)
    for s in row_sets:
        elems = sorted(s)
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                pair = frozenset([elems[i], elems[j]])
                pair_counts[pair] += 1
                
    # Keep pairs that appear in at least 2 rows
    frequent_pairs = {p for p, c in pair_counts.items() if c >= 2}
    
    gates = []
    next_wire = n + 1
    wire_cache = {}
    
    def get_wire(subset):
        nonlocal next_wire
        if len(subset) == 0:
            return zero_wire
        if len(subset) == 1:
            return next(iter(subset))
        
        if subset in wire_cache:
            return wire_cache[subset]
        
        if time.time() > DEADLINE:
            # Fallback
            elems = sorted(subset)
            w = elems[0]
            for e in elems[1:]:
                gates.append([w, e])
                w = next_wire
                next_wire += 1
            wire_cache[subset] = w
            return w
            
        elems = sorted(subset)
        
        # Check if we can split by a frequent pair
        s1 = None
        s2 = None
        for p in frequent_pairs:
            if p.issubset(subset):
                # Found a frequent pair in this subset
                # Use it as one half
                s1 = p
                s2 = frozenset(subset - p)
                break
                
        if s1 is None:
            # No frequent pair found, do standard balanced split
            mid = len(elems) // 2
            s1 = frozenset(elems[:mid])
            s2 = frozenset(elems[mid:])
            
        w1 = get_wire(s1)
        w2 = get_wire(s2)
        
        if w1 == w2:
            result = zero_wire
        else:
            gates.append([w1, w2])
            result = next_wire
            next_wire += 1
            
        wire_cache[subset] = result
        return result
        
    outputs = []
    for s in row_sets:
        outputs.append(get_wire(s))
        
    return {'gates': gates, 'outputs': outputs}
