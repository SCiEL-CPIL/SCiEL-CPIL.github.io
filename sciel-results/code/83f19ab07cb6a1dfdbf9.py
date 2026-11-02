import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # For k=5, g=9: use incidence graph of generalized hexagon of order (4,4)
    # This is a 5-regular graph with girth 12 on 2*(4+1)*(4^2+4+1) = 330 vertices
    if k == 5 and g <= 12:
        q = 4
        # Generalized hexagon GQ(4,4) or the standard hexagon of order (q,q)
        # Incidence graph: points and lines
        # Number of points = q^3 + q^2 + q + 1 = 64+16+4+1 = 85? No.
        # For hexagon of order (s,t): n_points = s^3 + s^2*t + s*t^2 + t^3? 
        # Actually for a generalized hexagon of order (s,t):
        # Number of points = s^3 + s^2*t + s*t^2 + t^3? Let me reconsider.
        # For a generalized hexagon of order (s,t), the number of points is:
        # (s+1)(s^2+st+t^2) if s=t? No.
        # Standard formula: for GQ(s,t) hexagon:
        # points = s^3 + s^2*t + s*t^2 + t^3? 
        # Actually the number of points in a generalized hexagon of order (s,t) is:
        # n = (s+1)(s^2 + s*t + t^2) ... no that's for something else.
        # 
        # Let me use the explicit construction. The hexagon of order (q,q) over GF(q)
        # has points and lines each parameterized by (x,y) in GF(q)^2 plus some at infinity.
        # 
        # Actually, let me use a simpler explicit construction.
        # The incidence graph of the generalized hexagon of order (4,4) 
        # can be constructed as follows:
        # Points: (a, b, c) where a,b,c in GF(4), with certain identifications
        # Lines: similar
        # 
        # Let me use the standard construction from Coxeter.
        # For a hexagon of order (s,t) with s=t=q:
        # Points are triples (x, y, z) in GF(q)^3 not all zero, modulo scalar?
        # No, that gives projective plane.
        # 
        # Let me just hardcode the known result: 330 vertices, 5-regular, girth 12.
        # I'll construct it explicitly.
        # 
        # The generalized hexagon of order (4,4) has:
        # 2*(4+1)*(4^2+4+1) = 2*5*21 = 210? No, 2*5*21=210, not 330.
        # 
        # Let me recalculate. For a generalized hexagon of order (s,t):
        # Number of points = (s+1)(s^2+st+t^2) ... 
        # For s=t=4: 5*(16+16+16) = 5*48 = 240. Still not 330.
        # 
        # Actually, I recall the formula differently. Let me look at this more carefully.
        # A generalized hexagon of order (s,t) has:
        # - Each point is on s+1 lines, each line has t+1 points
        # - Number of points = (s+1)(s^2 + st + t^2) ... hmm
        # 
        # Wait, let me just use the Moore bound. For 5-regular, girth 9:
        # Moore bound = 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1+5+20+80+320 = 426
        # So 330 < 426, which is impossible! The Moore bound is a LOWER bound.
        # So n must be >= 426. 330 is below the Moore bound, which is impossible.
        # 
        # I made an error. Let me reconsider.
        # For a d-regular graph with girth g (odd), the Moore bound is:
        # 1 + d + d(d-1) + d(d-1)^2 + ... + d(d-1)^((g-1)/2)
        # For d=5, g=9: 1 + 5 + 5*4 + 5*4^2 + 5*4^3 + 5*4^4 = 1+5+20+80+320+1280 = 1706
        # Wait, g=9 is odd, so (g-1)/2 = 4.
        # 1 + 5 + 5*4 + 5*16 + 5*64 + 5*256 = 1+5+20+80+320+1280 = 1706
        # 
        # Hmm, that's much larger. Let me recalculate.
        # For odd girth g=2r+1, the Moore bound is:
        # 1 + d * sum_{i=0}^{r-1} (d-1)^i
        # For d=5, g=9, r=4:
        # 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 1 + 425 = 426
        # 
        # So the Moore bound is 426. 330 < 426 is impossible.
        # 
        # So the generalized hexagon approach doesn't work for girth 9 with 330 vertices.
        # Let me fall back to the LUW construction.
        pass
    
    # Fall back to LUW construction
    q = k
    if q not in [2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        pass

    d = 1
    while True:
        girth = 2 * ((d + 1) // 2) + 4
        if girth >= g:
            break
        d += 1
        if time.time() > deadline:
            break

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
