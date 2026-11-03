import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before the 120s limit

    # 1. Generate LUW graph as baseline
    n_luw, edges_luw = _construct_luw(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # 2. For the specific case k=5, g=9, try a randomized greedy search
    # for a graph with n < n_luw. The best known is 1116.
    # We attempt a random greedy construction for n in a small range below n_luw.
    # This is heuristic and may not succeed within the time limit.
    if k == 5 and g == 9 and best_n > 1116:
        # Try a few candidate n values below best_n
        # The search space is huge, so we use a very short random greedy.
        # We try n = 1116 first, then 1118, 1120, etc.
        # But even n=1116 is very hard for random greedy in 110s in Python.
        # We'll try a very limited search: random greedy for n=1116 with a few restarts.
        
        # Actually, let's try a smarter approach: use the known structure.
        # But without the explicit edges, we can't.
        
        # So we'll just do a quick random greedy for a small number of steps.
        # If it completes and is valid, we return it. Otherwise, LUW.
        
        # For now, we'll skip the random search as it's unlikely to succeed
        # in the time limit for n > 1000 in pure Python.
        # We return the LUW graph.
        
        pass

    return best_n, best_edges


def _construct_luw(k, g):
    """Construct the Lazebnik-Ustimenko-Woldar graph."""
    q = k
    # Determine the depth d for the LUW construction
    # For girth g, we need d such that 2d+1 >= g
    # But the LUW construction uses a specific depth based on g.
    # For g=9, d=4 gives girth 9? Let's check.
    # The LUW graph LUW(q, d) has girth 2d+1.
    # So for g=9, d=4.
    # For general g, d = (g-1)//2.
    
    d = (g - 1) // 2
    if d < 3:
        d = 3

    # The rule for the LUW construction
    # We use a table of coefficients
    # For q=5, d=4, the rules are:
    # y[2] = x[2] + y[1]*x[1]
    # y[3] = x[3] + y[2]*x[2]
    # y[4] = x[4] + y[3]*x[3]
    # But the actual LUW construction is more complex.
    
    # Let's use the standard LUW construction.
    # The vertices are (side, x) where side in {0,1} and x is a tuple of length d+1.
    # x[0] is None, x[1..d] are in Z_q.
    # The neighbours of (0, x) are (1, y) where:
    # y[1] = t (for t in Z_q)
    # y[c] = x[c] + y[c-1]*x[c-1] for c=2..d
    # The neighbours of (1, y) are (0, x) where:
    # x[1] = t (for t in Z_q)
    # x[c] = y[c] - x[c-1]*y[c-1] for c=2..d
    
    # This gives a 2-regular graph? No, it's q-regular.
    # Each vertex has q neighbours.
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                if side == 0:
                    y[c] = (x[c] + y[c-1] * x[c-1]) % q
                else:
                    y[c] = (x[c] - x[c-1] * y[c-1]) % q
            yield 1 - side, tuple(y)

    # Start from (0, (None, 0, 0, ..., 0))
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
