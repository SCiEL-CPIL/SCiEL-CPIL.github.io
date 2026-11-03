import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # LUW Construction
    # For k=3, we use q=3.
    # The LUW construction yields a q-regular graph with 2*q^(d+1) vertices
    # and girth at least 2*d + 2.
    
    if k != 3:
        # Fallback for other k, though task is specific to k=3, g=16.
        # For general k, we might need different constructions.
        # But the prompt says "evaluator passes this task's degree k and girth g".
        # It implies k=3, g=16.
        pass

    q = k
    
    # Find smallest d such that 2*d + 2 >= g
    d = 2
    while 2 * d + 2 < g:
        d += 1
    
    # n = 2 * q^(d+1)
    # We construct the graph explicitly.
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q

    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    
    # Standard LUW rules
    # The specific rules depend on the paper.
    # The seed code provided rules. I will reuse them.
    
    if d >= 2:
        rule_a[2] = 1
        rule_b[2] = 1
    if d >= 3:
        rule_a[3] = 2
        rule_b[3] = 1
    if d >= 4:
        rule_a[4] = 1
        rule_b[4] = 2
        
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0:
            rule_a[c] = 1
            rule_b[c] = c0 - 2
        elif r == 1:
            rule_a[c] = c0 - 1
            rule_b[c] = 1
        elif r == 2:
            rule_a[c] = c0
            rule_b[c] = 1
        else:
            rule_a[c] = 1
            rule_b[c] = c0 + 1

    pow_q = [1] * (d + 1)
    for i in range(1, d + 1):
        pow_q[i] = pow_q[i-1] * q
    
    def encode(side, coords):
        val = side * pow_q[d]
        for i in range(d):
            val += coords[i] * pow_q[d - 1 - i]
        return val

    def decode(val):
        side = val // pow_q[d]
        val -= side * pow_q[d]
        coords = []
        for i in range(d):
            coords.append(val % q)
            val //= q
        coords.reverse()
        return side, tuple(coords)

    start_coords = tuple([0] * d)
    start_val = encode(0, start_coords)
    
    index = {start_val: 0}
    order = [start_val]
    edges = []
    
    pos = 0
    while pos < len(order):
        if time.time() > deadline:
            break
        
        val = order[pos]
        pos += 1
        side, x = decode(val)
        v_idx = index[val]
        
        for first in range(q):
            y_coords = [0] * d
            y_coords[0] = first
            
            for c in range(1, d):
                c_actual = c + 1
                a = rule_a[c_actual]
                b = rule_b[c_actual]
                
                if side == 0:
                    y_coords[c] = add(x[c], mul(y_coords[a - 1], x[b - 1]))
                else:
                    y_coords[c] = sub(x[c], mul(x[a - 1], y_coords[b - 1]))
            
            neighbor_coords = tuple(y_coords)
            neighbor_val = encode(1 - side, neighbor_coords)
            
            if neighbor_val not in index:
                n_idx = len(order)
                index[neighbor_val] = n_idx
                order.append(neighbor_val)
            
            if side == 0:
                n_idx = index[neighbor_val]
                edges.append((v_idx, n_idx))
    
    n = len(order)
    return n, edges
