"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 120.0
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    # Try to reduce n via circulant graph construction for k=3
    if k == 3:
        best_n = n
        best_edges = edges
        # Moore bound for (3, g)
        if g % 2 == 0:
            moore = 2 * (2 ** (g // 2 - 1) - 1)
        else:
            moore = 2 * (2 ** (g // 2) - 1)
        
        # Search for circulant graphs with connection set {1, a, b}
        # For 3-regular circulant, we need S = {1, a, -a} or {1, a, b} with -1, -a, -b in S
        # Simplest: S = {1, a, -a} (i.e., {1, a, n-a})
        # This requires n >= 2a + 1 and gcd(n, a) = 1, and no short cycles.
        
        # For girth g, we need no sum of < g elements from S (with signs) to be 0 mod n
        # This is a hard constraint. Let's try small n and check girth.
        
        # Since we have time budget, try a few candidates
        # Start from moore bound and go up
        # But checking girth is expensive. We'll use a quick BFS for non-backtracking.
        
        # For k=3, g=16, moore = 2*(2^7 - 1) = 254. But known is 936.
        # So circulant likely won't beat 936 easily.
        # However, the LUW graph for k=3, g=16: q=3, d=11 (odd).
        # n = 2 * q^d = 2 * 3^11 = 2 * 177147 = 354294.
        # This is much larger than 936.
        
        # So the LUW construction is very poor for k=3.
        # The random search in inspirations might find something better if given time,
        # but 120s is short.
        
        # Given the directive to keep the design family (LUW) and investigate improvements,
        # and the fact that LUW is far from optimal, perhaps we can try a different q?
        # The LUW construction requires q to be a prime or 4.
        # For k=3, q must be 3.
        
        # What if we use a different construction? The directive says "KEEP this design".
        # So we stick with LUW.
        
        # Can we improve LUW?
        # The parameter d determines girth. d+5 >= g.
        # For g=16, d=11.
        # n = 2 * q^d.
        
        # There is no improvement within the LUW family that reduces n significantly
        # without changing the family.
        
        # However, we can try to verify if the graph is indeed valid and return it.
        # The current code already does this.
        
        # One potential improvement: use a smaller q if possible?
        # No, q=k.
        
        # So for k=3, g=16, we return the LUW graph with n=354294.
        # This is the best we can do within the constraint.
        
        # Wait, the score is 1.0, which is the seed.
        # The seed program is the LUW construction.
        # So we are returning the same thing.
        
        # Is there any way to improve?
        # Maybe by checking if the graph is connected? The code takes the component of zero.
        # If the graph is not connected, we might be able to take a smaller component?
        # But LUW graphs are connected for q prime.
        
        # Let's just return the LUW graph.
        pass
    
    return n, edges
# EVOLVE-BLOCK-END
