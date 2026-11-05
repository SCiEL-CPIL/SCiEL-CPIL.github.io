"""Construct k-regular graph of girth g using L-U-W construction."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Use the Lazebnik-Ustimenko-Woldar (L-U-W) construction CD(d, q).
    For small parameters, also try known optimal constructions.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    time_budget = 130.0  # Leave 20s margin for safety
    
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q).
        
        The graph has 2*q^d vertices and is q-regular with girth at least 2d-2 (for q odd)
        or 2d-2 (for q=4 with appropriate d).
        
        We choose d such that 2d-2 >= g, i.e., d >= (g+2)//2.
        """
        q = k
        # Need girth >= g, so need 2d - 2 >= g => d >= (g+2)//2
        d = max(3, (g + 2) // 2)
        
        # For q=4, use GF(4) arithmetic
        if q == 4:
            # GF(4) = {0, 1, alpha, alpha+1} with alpha^2 + alpha + 1 = 0
            # Represent as integers: 0->0, 1->1, alpha->2, alpha+1->3
            # Addition is XOR
            # Multiplication table:
            #   * | 0 1 2 3
            #   0 | 0 0 0 0
            #   1 | 0 1 2 3
            #   2 | 0 2 3 1
            #   3 | 0 3 1 2
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule for computing coordinates
        # For CD(d, q), the rule is defined recursively
        rule = {}
        rule[2] = (1, 1)
        if d >= 3:
            rule[3] = (2, 1)
        if d >= 4:
            rule[4] = (1, 2)
        for c in range(5, d + 1):
            # The standard L-U-W rule cycles through patterns
            # For c >= 5: use periodic pattern with period 4
            r = (c - 5) % 4
            c0 = c - r
            patterns = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = patterns[r]
        
        def neighbours(vertex):
            """Generate neighbors of a vertex in the L-U-W graph."""
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
        
        # Build the graph by BFS from a starting vertex
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vi, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # Add edge only once (from the "0" side)
                if vertex[0] == 0:
                    edges.append((vi, index[other]))
        
        return len(order), edges
    
    # Special case: k=3, g=5 -> Petersen graph (10 vertices)
    if k == 3 and g == 5:
        # Petersen graph
        edges = []
        # Outer 5-cycle
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Inner 5-cycle (star)
        for i in range(5):
            edges.append((5 + i, 5 + (2 * i + 1) % 5))
        # Spokes
        for i in range(5):
            edges.append((i, 5 + i))
        return 10, edges
    
    # Special case: k=3, g=6 -> 30-vertex graph or use L-U-W
    if k == 3 and g == 6:
        # Use L-U-W with d=4, q=3: n = 2*3^4 = 162
        # Or try to find something better
        pass
    
    # Main construction: L-U-W
    luw_n, luw_edges = luw_construct(k, g)
    
    # Verify we haven't exceeded time budget
    if time.time() - start_time > time_budget:
        return luw_n, luw_edges
    
    # For small cases, we could potentially search for better constructions,
    # but L-U-W is generally good. Let's just return it.
    return luw_n, luw_edges
# EVOLVE-BLOCK-END
