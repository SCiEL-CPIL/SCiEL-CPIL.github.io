import time
from collections import deque

def construct(k, g):
    # Calculate minimum d for LUW construction
    # Girth of LUW(q, d) is at least 2d + 2
    # We need 2d + 2 >= g  => d >= (g - 2) / 2
    d = (g - 2 + 1) // 2
    if d < 1:
        d = 1
    
    q = k
    if q < 2:
        q = 2
        
    # LUW Construction
    # Vertices: (side, x) where side in {0, 1}, x is a tuple of length d
    # x[0] ... x[d-1] in range(q)
    # Total vertices: 2 * q^d
    
    # Define operations
    if q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y # This is not correct for field multiplication, but q=2 is special
        # Actually for q=2, the field is GF(2). x*y = x AND y? No, 1*1=1, 1*0=0. Yes.
        # But LUW usually requires q to be a prime power.
        # For k=3, q=3.
        
    # For general q, we assume q is prime or prime power.
    # For k=3, q=3.
    
    # Rule definition for LUW
    # The rule depends on the position in the tuple.
    # Standard LUW rule:
    # y_c = x_c + y_{c-1} x_{c-2} (mod q) ?
    # The seed code used a specific rule.
    # Let's use the standard LUW rule from the paper "Large Girth Graphs".
    # The neighbours of (0, x) are (1, y) where y is computed from x.
    # y_1 = x_1
    # y_2 = x_2 + x_1 * y_1 ?
    
    # Actually, the seed code's rule was:
    # rule[c] = (a, b)
    # y[c] = x[c] + y[a] * x[b]
    
    # Let's stick to the seed code's rule structure but with correct d.
    # The seed code's rule generation:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #    r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This rule is specific. We will reuse it.
    
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def get_neighbours(side, x):
        # x is a tuple of length d
        # We need to generate q neighbours
        # The first element of x is fixed?
        # In the seed code, x had length d+1, with x[0]=None.
        # Here x has length d.
        # The "first" variable in the seed code iterated range(q).
        # This suggests x[1] (or x[0] in 0-indexed) is the free variable.
        
        # Let's adapt the seed code's logic directly.
        # Seed code:
        # start = (0, (None,) + (0,) * d)
        # x has length d+1. x[0]=None. x[1..d] are values.
        # Neighbours:
        # for first in range(q):
        #    y = [None, first] + [0]*(d-1)
        #    for c in range(2, d+1):
        #       ...
        
        # So we will use the same representation: x is a tuple of length d+1.
        # x[0] is None. x[1]..x[d] are in range(q).
        
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    # Generate all vertices
    # Vertices are (side, x)
    # x[0] is None. x[1]..x[d] are in 0..q-1.
    # Total q^d combinations for x.
    # 2 sides.
    
    vertices = []
    index = {}
    
    # Iterate over all x tuples
    # We can use itertools.product
    import itertools
    for x_vals in itertools.product(range(q), repeat=d):
        x_tuple = (None,) + x_vals
        for side in [0, 1]:
            v = (side, x_tuple)
            index[v] = len(vertices)
            vertices.append(v)
            
    n = len(vertices)
    edges = []
    seen_edges = set()
    
    for i, v in enumerate(vertices):
        side, x = v
        for nb in get_neighbours(side, x):
            j = index[nb]
            # Add edge
            if i < j:
                e = (i, j)
                if e not in seen_edges:
                    seen_edges.add(e)
                    edges.append(e)
            else:
                e = (j, i)
                if e not in seen_edges:
                    seen_edges.add(e)
                    edges.append(e)
                    
    return n, edges
