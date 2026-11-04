"""
Construct a 4-regular graph of girth at least 10 on as few vertices as possible.

Strategy: Use the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q) with q=4, d=5
to obtain a 4-regular bipartite graph with girth >= 10. This gives 2048 vertices.

The best known 4-regular girth-10 cage has 320 vertices, but that graph is complex
to construct. The LUW construction is explicit and guaranteed correct.

Key insight: The LUW graph CD(d,q) with q=k gives a k-regular graph. For k=4, q=4
works (using GF(4) arithmetic). With d=5 (since g=10, d = g-5 = 5), we get girth >= 10.

The number of vertices in CD(d,q) is 2*q^(d+1) in the full graph, but we only take
the connected component of the zero vertex, which for q=4, d=5 gives a specific
number of vertices.

Actually, let me reconsider. The LUW construction CD(d,q) has:
- Points: (d+1)-tuples over GF(q) with first coordinate None, so q^d points per side
- Lines: same structure
- Total: 2*q^d vertices
- Each vertex has degree q
- Girth is at least d+5

For q=4, d=5: 2*4^5 = 2048 vertices, degree 4, girth >= 10.

But we can potentially do better. Let me think about other constructions.

Actually, for a 4-regular graph of girth 10, the Moore bound gives:
n >= 2*(1 + 3 + 3^2 + 3^3 + 3^4) = 2*(1+3+9+27+81) = 2*121 = 242
So n >= 243 (since 242 is even and we need... actually the Moore bound for girth g
is 1 + k*sum_{i=0}^{(g-3)/2} (k-1)^i for odd girth g).

For k=4, g=10:
Moore bound = 2*(1 + 3 + 3^2 + 3^3 + 3^4) = 2*(1+3+9+27+81) = 242
But since girth is even, the Moore bound is 2*sum_{i=0}^{g/2-1} (k-1)^i = 2*(1+3+9+27) = 92 for g=8?

Let me recalculate. For a k-regular graph with girth g:
- If g is odd: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
- If g is even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i

For k=4, g=10 (even):
n >= 2 * (1 + 3 + 9 + 27) = 2 * 40 = 80

Hmm, that seems too low. Let me double-check.

For girth g (even), the Moore bound is:
n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i

For g=10: sum_{i=0}^{4} (k-1)^i = 1 + 3 + 9 + 27 + 81 = 121
n >= 2 * 121 = 242

Yes, so the Moore bound is 242, and since n must be an integer >= 242, the lower bound is 242.
The best known is 320.

The LUW graph gives 2048 vertices. That's far from optimal but valid.

Let me just use the LUW construction as it's the only one I can guarantee correctness for.
"""

import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 100 second safety margin
    
    # Use LUW construction CD(d, q) with q = k
    q = k
    # For girth g, we need d such that d + 5 >= g, so d >= g - 5
    # d should be odd for the standard LUW construction
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    # GF(q) arithmetic (q=4, use XOR-based representation)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LUW construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Build the graph using BFS from the zero vertex
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
            yield (1 - side), tuple(y)
    
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
        
        if time.time() > deadline:
            break
    
    return len(order), edges
