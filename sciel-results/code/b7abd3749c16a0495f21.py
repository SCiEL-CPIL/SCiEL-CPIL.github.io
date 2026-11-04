"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # LUW construction for k-regular graphs with large girth
    # For k=4, g=10: use d=5 (minimum odd >= g-5=5)
    
    min_d_odd = max(3, g - 5)
    if min_d_odd % 2 == 0:
        min_d_odd += 1
        
    min_d_even = max(4, g - 4)
    if min_d_even % 2 == 1:
        min_d_even += 1
        
    d = min(min_d_odd, min_d_even)
    q = k
    
    # GF(4) arithmetic for k=4 using bit operations
    if q == 4:
        # GF(4) = {0, 1, 2, 3} with addition = XOR
        # Multiplication table
        mul_table = (
            0, 0, 0, 0,
            0, 1, 2, 3,
            0, 2, 3, 1,
            0, 3, 1, 2
        )
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return mul_table[x * 4 + y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return (x * y) % q

    # LUW rules
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            a, b = 1, c0 - 2
        elif r == 1:
            a, b = c0 - 1, 1
        elif r == 2:
            a, b = c0, 1
        else:
            a, b = 1, c0 + 1
        rule_a[c] = a
        rule_b[c] = b

    # Precompute transitions for each side
    # For side 0: y[c] = x[c] + y[a] * x[b]
    # For side 1: y[c] = x[c] - x[a] * y[b]
    
    # BFS from origin to build the graph
    # Vertex = (side, tuple of d+1 GF(q) elements)
    start = (0, tuple([0] * (d + 1)))
    index = {start: 0}
    order = [start]
    edges = []
    
    head = 0
    while head < len(order):
        if time.time() > deadline - 10:
            break
        
        side, x = order[head]
        head += 1
        x_idx = index[(side, x)]
        
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            
            other = (1 - side, tuple(y))
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
            
            # Only add edge from side 0 to avoid duplicates
            if side == 0:
                edges.append((x_idx, index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
