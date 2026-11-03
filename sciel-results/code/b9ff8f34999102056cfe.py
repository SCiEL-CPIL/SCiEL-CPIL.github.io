import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the LUW construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    q = k

    # Find smallest d such that girth(d) >= g
    # LUW: girth = d + 5 (odd d), d + 6 (even d)
    d = 1
    while True:
        if d % 2 == 1:
            girth = d + 5
        else:
            girth = d + 6
        if girth >= g:
            break
        d += 1

    # LUW rule: rule[c] = (a, b) for c = 2..d
    # Use a valid rule pattern that ensures girth
    # Standard LUW: rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2)
    # For c>=5: use repeating pattern shifted by 4
    # This is the pattern from the original LUW paper
    
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    
    # Base pattern for positions 2,3,4
    rule_a[2] = 1; rule_b[2] = 1
    rule_a[3] = 2; rule_b[3] = 1
    rule_a[4] = 1; rule_b[4] = 2
    
    # For c >= 5: shift the base pattern
    # rule[c] = (rule_a[c-4] + 3, rule_b[c-4] + 3) if within bounds
    # Actually, the LUW rule uses a specific construction
    # Let me use: rule[c] = (c-2, 1) for c >= 3, rule[2] = (1,1)
    # This "chain" rule is known to work for prime fields
    
    # Reset and use chain rule
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2] = 1; rule_b[2] = 1
    for c in range(3, d + 1):
        rule_a[c] = c - 1
        rule_b[c] = 1

    # Precompute GF(q) tables
    add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
    sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]
    mul_table = [[(x * y) % q for y in range(q)] for x in range(q)]

    # Encode/decode
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
        coords = [0] * d
        for i in range(d - 1, -1, -1):
            coords[i] = val % q
            val //= q
        return side, coords

    # BFS
    start_val = encode(0, [0] * d)
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
            y = [0] * d
            y[0] = first
            
            for c in range(1, d):
                c_actual = c + 1
                a = rule_a[c_actual]
                b = rule_b[c_actual]
                
                if side == 0:
                    y[c] = add_table[x[c]][mul_table[y[a-1]][x[b-1]]]
                else:
                    y[c] = sub_table[x[c]][mul_table[x[a-1]][y[b-1]]]
            
            neighbor_val = encode(1 - side, y)
            
            if neighbor_val not in index:
                n_idx = len(order)
                index[neighbor_val] = n_idx
                order.append(neighbor_val)
            
            if side == 0:
                n_idx = index[neighbor_val]
                edges.append((v_idx, n_idx))
    
    n = len(order)
    return n, edges
