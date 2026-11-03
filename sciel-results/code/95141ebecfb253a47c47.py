import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # We stick to the LUW family as per directive.
    q = k
    if q != 4:
        # Fallback for other k, though task specifies k=4
        # For now, assume k=4
        pass
        
    # Determine d for LUW
    # Girth of CD(d, q) is d + 5
    # We need girth >= g
    # d + 5 >= g => d >= g - 5
    # d must be odd
    d_candidate = g - 5
    if d_candidate % 2 == 0:
        d_candidate += 1
    d = max(3, d_candidate)
    
    # For k=4, g=10: d = 5.
    # Vertices = 2 * q^d = 2 * 4^5 = 2048.
    
    if q == 4:
        # GF(4) operations
        # Represent elements as 0, 1, 2, 3
        # 0: 0
        # 1: 1
        # 2: alpha
        # 3: alpha+1
        # alpha^2 = alpha + 1 = 3
        # alpha^3 = 1
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        
        def gf_add(x, y):
            return add_table[x][y]
            
        def gf_mul(x, y):
            return mul_table[x][y]
            
        def gf_sub(x, y):
            # In char 2, sub is add
            return add_table[x][y]
    else:
        # Generic field for prime q
        def gf_add(x, y):
            return (x + y) % q
            
        def gf_mul(x, y):
            return (x * y) % q
            
        def gf_sub(x, y):
            return (x - y) % q

    # Rule for LUW
    # l[c] - p[c] = l[a] * p[b]
    # The rule (a, b) depends on c.
    # In the seed code:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d:
    #   r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #   rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The indices in the list correspond to r=0,1,2,3
        # r=0: (1, c0 - 2)
        # r=1: (c0 - 1, 1)
        # r=2: (c0, 1)
        # r=3: (1, c0 + 1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Generate vertices
    # Points: (0, x) where x is tuple of length d+1 (index 0 is None/0)
    # Lines: (1, y)
    
    # We use BFS from start to ensure connectivity and get indices
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    # Precompute neighbors for speed?
    # Number of vertices is 2 * q^d.
    # For d=5, q=4, N=2048.
    # We can iterate through all possible vertices and check if they are in the component.
    # But BFS is fine.
    
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        # x is tuple of length d+1
        # x[0] is ignored (or 0)
        # x[1]..x[d] are the coordinates
        
        for first in range(q):
            # y[0] is ignored
            # y[1] = first
            y = [0] * (d + 1)
            y[1] = first
            
            # Compute y[2]..y[d]
            valid = True
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # y[c] - x[c] = y[a] * x[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    # Line to Point
                    # x[c] - y[c] = x[a] * y[b]
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            
            other = (1 - side, tuple(y))
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
                
            # Add edge only if we are on the point side to avoid duplicates
            if side == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
