import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the LUW construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    q = k
    # LUW: girth = d + 5 (odd d), d + 6 (even d)
    # We need girth >= g
    # For k=3, g=16: need d+5 >= 16 -> d >= 11 (odd) or d+6 >= 16 -> d >= 10 (even)
    # Smallest d: d=11 (odd, girth=16) or d=10 (even, girth=16)
    # d=10 gives 2*q^10 = 2*3^10 = 118098 vertices (if q=3)
    # d=11 gives 2*q^11 = 2*3^11 = 354294 vertices
    
    # Let's use d=10 for smaller graph if possible
    # girth(10) = 10+6 = 16 >= 16 ✓
    # girth(11) = 11+5 = 16 >= 16 ✓
    
    # Use d=10 for fewer vertices
    d = 10
    while True:
        if d % 2 == 1:
            girth = d + 5
        else:
            girth = d + 6
        if girth >= g:
            break
        d += 1

    # For q=3, d=10: n = 2 * 3^10 = 118098
    # For q=3, d=11: n = 2 * 3^11 = 354294
    # Let's stick with d=10 for efficiency
    
    # GF(q) operations for q=3
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q

    # Rule for LUW
    # rule[c] = (a, b) for c = 2..d
    # rule[2] = (1, 1), rule[3] = (2, 1), rule[4] = (1, 2)
    # For c >= 5: repeat pattern with shift
    rule_a = [0] * (d + 1)  # 1-indexed
    rule_b = [0] * (d + 1)
    
    rule_a[2] = 1
    rule_b[2] = 1
    rule_a[3] = 2
    rule_b[3] = 1
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

    # Generate vertices using BFS
    # Vertex: (side, x1, x2, ..., xd) where side in {0, 1}
    # side=0: points, side=1: lines
    # x1 is free (0..q-1), x2..xd are determined
    
    # Use integer encoding for speed
    # Encode (side, x1, ..., xd) as a single integer
    # side * q^d + x1 * q^(d-1) + x2 * q^(d-2) + ... + xd
    
    pow_q = [1] * (d + 1)
    for i in range(1, d + 1):
        pow_q[i] = pow_q[i-1] * q
    
    def encode(side, coords):
        """coords is a tuple of length d: (x1, x2, ..., xd)"""
        val = side * pow_q[d]
        for i in range(d):
            val += coords[i] * pow_q[d - 1 - i]
        return val

    def decode(val):
        """Decode integer back to (side, coords)"""
        side = val // pow_q[d]
        val -= side * pow_q[d]
        coords = []
        for i in range(d):
            coords.append(val % q)
            val //= q
        coords.reverse()
        return side, tuple(coords)

    # BFS
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
            # Build neighbor coordinates
            y_coords = [0] * d
            y_coords[0] = first  # x1 = first
            
            for c in range(1, d):  # c indexes x2..xd (1-indexed in our coords)
                c_actual = c + 1  # actual position 2..d
                a = rule_a[c_actual]
                b = rule_b[c_actual]
                # a, b are 1-indexed positions in the tuple (1..d)
                # In y_coords, position i corresponds to x_{i+1}
                # So x_a is at y_coords[a-1], x_b is at y_coords[b-1]
                
                if side == 0:
                    # Point -> Line: y[c] = x[c] + y[a] * x[b]
                    # y_coords[c-1] = x[c-1] + y_coords[a-1] * x[b-1]
                    y_coords[c] = add(x[c], mul(y_coords[a - 1], x[b - 1]))
                else:
                    # Line -> Point: y[c] = x[c] - x[a] * y[b]
                    # y_coords[c-1] = x[c-1] - x[a-1] * y_coords[b-1]
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

# EVOLVE-BLOCK-END
