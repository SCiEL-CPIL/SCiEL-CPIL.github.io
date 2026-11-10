<<<<<<< SEARCH
    q = k
    
    # Determine minimal d
    # Case 1: d is odd. Girth = d + 5. Need d + 5 >= g => d >= g - 5.
    # d must be odd.
    d_odd = g - 5
    if d_odd % 2 == 0:
        d_odd += 1
    if d_odd < 1: d_odd = 1
    
    # Case 2: d is even. Girth = d + 4. Need d + 4 >= g => d >= g - 4.
    # d must be even.
    d_even = g - 4
    if d_even % 2 != 0:
        d_even += 1
    if d_even < 2: d_even = 2
    
    # Choose d that gives smaller n = 2 * q^d
    # Since q >= 2, smaller d is better.
    if d_odd < d_even:
        d = d_odd
    else:
        d = d_even
        
    # For q = 4, we need GF(4) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW graph
    # l[c] - p[c] = l[a] * p[b]
    # rule[c] = (a, b)
    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            elif r == 3:
                rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
=======
    q = k
    
    # Determine minimal d
    # Case 1: d is odd. Girth = d + 5. Need d + 5 >= g => d >= g - 5.
    # d must be odd.
    d_odd = g - 5
    if d_odd % 2 == 0:
        d_odd += 1
    if d_odd < 1: d_odd = 1
    
    # Case 2: d is even. Girth = d + 4. Need d + 4 >= g => d >= g - 4.
    # d must be even.
    d_even = g - 4
    if d_even % 2 != 0:
        d_even += 1
    if d_even < 2: d_even = 2
    
    # Choose d that gives smaller n = 2 * q^d
    # Since q >= 2, smaller d is better.
    if d_odd < d_even:
        d = d_odd
    else:
        d = d_even
        
    # For q = 4, we need GF(4) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Precompute rule as a list for faster access
    rule = [None] * (d + 1)
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            elif r == 3:
                rule[c] = (1, c0 + 1)

    # Precompute multiplication tables for speed
    if q <= 1000:
        mul_table = [[mul(x, y) for y in range(q)] for x in range(q)]
    else:
        mul_table = None

    def neighbours(vertex):
        side, x = vertex
        if side == 0:
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if mul_table is not None:
                        y[c] = add(x[c], mul_table[y[a]][x[b]])
                    else:
                        y[c] = add(x[c], mul(y[a], x[b]))
                yield 1, tuple(y)
        else:
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if mul_table is not None:
                        y[c] = sub(x[c], mul_table[x[a]][y[b]])
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 0, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > _deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
>>>>>>> REPLACE
