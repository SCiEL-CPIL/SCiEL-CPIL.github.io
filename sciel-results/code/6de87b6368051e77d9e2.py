import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if k == 4 and g == 10:
        # Try to construct the 320-vertex cage
        # Using voltage graph on C4 with group Z_2^4 x Z_5
        # This is a heuristic construction.
        n = 320
        # Vertices are (x0, x1, x2, x3, y) where xi in {0,1}, y in {0..4}
        # Index: ((x0*8 + x1*4 + x2*2 + x3) * 5) + y
        # Or simply: x in 0..15, y in 0..4, index = x*5 + y
        
        # Base edges: (0,1), (1,2), (2,3), (3,0)
        # Voltages:
        # e0 = (1,0,0,0,0) -> x+1, y
        # e1 = (0,1,0,0,1) -> x+2, y+1
        # e2 = (0,0,1,0,2) -> x+4, y+2
        # e3 = (0,0,0,1,3) -> x+8, y+3
        
        # Check if this yields girth 10.
        # If not, fallback to LUW.
        
        def get_neighbours(v):
            x, y = divmod(v, 5)
            # x is 0..15, y is 0..4
            # x bits: b0 b1 b2 b3
            b0 = x & 1
            b1 = (x >> 1) & 1
            b2 = (x >> 2) & 1
            b3 = (x >> 3) & 1
            
            # Neighbour 0: x+1, y
            nx0 = x ^ 1
            ny0 = y
            n0 = nx0 * 5 + ny0
            
            # Neighbour 1: x+2, y+1
            nx1 = x ^ 2
            ny1 = (y + 1) % 5
            n1 = nx1 * 5 + ny1
            
            # Neighbour 2: x+4, y+2
            nx2 = x ^ 4
            ny2 = (y + 2) % 5
            n2 = nx2 * 5 + ny2
            
            # Neighbour 3: x+8, y+3
            nx3 = x ^ 8
            ny3 = (y + 3) % 5
            n3 = nx3 * 5 + ny3
            
            return [n0, n1, n2, n3]

        # Build graph
        adj = [[] for _ in range(n)]
        for v in range(n):
            for u in get_neighbours(v):
                adj[v].append(u)
                # Ensure symmetry? 
                # The voltage graph construction should be symmetric if voltages are inverses.
                # But here we defined specific directions.
                # We need to ensure the graph is undirected.
                # The edge (v, u) implies (u, v).
                # In voltage graphs, if we have an edge with voltage g, the reverse edge has voltage g^-1.
                # Here we are just defining a set of neighbors.
                # We must ensure that if u is a neighbor of v, v is a neighbor of u.
                # Let's check symmetry.
                # n0 = (x^1, y). From n0, neighbor 0 is (x^1^1, y) = (x, y) = v. OK.
                # n1 = (x^2, y+1). From n1, neighbor 1 is (x^2^2, y+1+1) = (x, y+2). Not v.
                # So this is not symmetric.
                # We need to add the reverse edges or define the graph as undirected.
                # If we just add all directed edges, we get a directed graph.
                # The task requires a simple undirected graph.
                # We can treat it as undirected by adding (v, u) and (u, v).
                # But we must ensure no multiple edges.
                
                pass

        # This heuristic might not yield a valid 4-regular graph of girth 10.
        # I will fallback to LUW if I cannot verify quickly.
        # However, 320 is small. I can do a BFS to check girth.
        
        # Let's try a different known construction:
        # The (4,10)-cage is the **Klein graph** K(320)?
        # I will use the LUW graph as it is guaranteed valid.
        # But I will try to find a smaller one if possible.
        # Since I am not sure about the 320 construction, I will stick to LUW 
        # but optimize the code to be faster.
        
        pass

    # General LUW construction (Seed logic)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
                
    return len(order), edges
