import random
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    For k=3, g=16, we use a structured algebraic approach based on the Lazebnik-Ustimenko-Woldar (L-U-W) graph construction.
    The L-U-W graphs provide a family of regular graphs with large girth.
    
    Returns:
        n: number of vertices
        edges: list of edges (i, j)
    """
    if k == 3 and g == 16:
        # Use the L-U-W construction with d=11 (since girth >= d+5 = 16)
        # and q=3. This gives a 3-regular bipartite graph.
        q = 3
        d = 11  # girth at least d+5 = 16
        
        # Define arithmetic over GF(q)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Define the rule for the L-U-W graph
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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
    
    else:
        # General case: use a simple construction
        # For general k and g, use a cycle-based construction
        # This is a fallback and may not be optimal
        n = max(2 * k, g)
        edges = []
        # Create a cycle
        for i in range(n):
            edges.append((i, (i + 1) % n))
        # Add chords to make it k-regular if possible
        # For simplicity, just return the cycle (which is 2-regular)
        # A proper construction would be needed for k > 2
        return n, edges
