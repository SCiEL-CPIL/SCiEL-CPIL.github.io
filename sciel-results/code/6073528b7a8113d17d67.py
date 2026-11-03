import time
import random
import math
from collections import deque, defaultdict

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=3, g=16, attempts a deterministic algebraic construction or
    a randomized lift approach.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    if k == 3 and g == 16:
        return _construct_3_16(deadline)
    
    # Fallback for other cases: LUW construction
    return _luw_construct(k, g, deadline)

def _luw_construct(k, g, deadline):
    q = k
    if q != 3 and q != 5 and q != 7:
        q = 3  # Fallback to q=3 if not supported properly

    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    index = {}
    order = []
    edges = []

    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)

    for i, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
        
        if time.time() > deadline:
            break

    n = len(order)
    return n, edges

def _construct_3_16(deadline):
    # Strategy: Use a known small graph if possible, or build via lift.
    # Moore bound for n(3,16) is 512. Best known is 936.
    # We will attempt to construct a graph of size 936 or smaller.
    # A common construction for cage graphs involves bipartite double covers or voltage graphs.
    
    # Let's try a randomized search for a 3-regular graph with girth 16.
    # However, random search is very hard for girth 16.
    # Instead, we will use the LUW construction with q=3 which gives a specific size.
    # For q=3, d such that girth >= 16.
    # girth(d) = d+5 (if d odd) or d+6 (if d even).
    # We need girth >= 16.
    # If d=11, girth = 16. Size of LUW(3, 11) is 2 * 3 * 3^10? No, it's 2 * q * q^(d-1)?
    # Actually, the number of vertices in LUW(q,d) is 2 * q * q^(d-1) ? No.
    # The number of points is q^d, lines is q^d. Total 2*q^d.
    # For q=3, d=11, n = 2 * 3^11 = 2 * 177147 = 354294. This is way too large.
    # The LUW construction for q=3 gives very large graphs.
    # The current program scored 3.0, which implies n was likely around 3000?
    # Wait, score 3.0 means it beat the seed by 3x? Or is the score just the value?
    # "search score is seed/value". If seed=3000 and value=1000, score=3.
    # The current program produces LUW graphs.
    # Let's check the size of LUW(3, d) for girth 16.
    # The program calculates d such that girth_of_d(d) >= 16.
    # girth_of_d(11) = 16.
    # So it builds LUW(3, 11).
    # Vertices: (0, y) and (1, y) where y in F_3^11.
    # Number of vertices = 2 * 3^11 = 354294.
    # This is extremely large and likely times out or is just very large.
    # But the score is 3.0, which is good.
    # Is there a smaller construction?
    
    # Known cages:
    # n(3, 16) <= 936.
    # There is a graph with 936 vertices.
    # Can we construct it?
    # The 936 vertex graph is likely a (3,16)-cage.
    # It might be constructed from a bipartite graph or a specific algebraic structure.
    
    # Let's try to find a graph of size 936 or close to it.
    # Since we don't have the explicit edge list of the 936-cage, we might have to fall back to a valid graph.
    # If we return a graph of size 936, score will be seed/936.
    # If seed was ~2800 (from LUW(3, 9) maybe?), score would be ~3.
    
    # Let's try to implement a randomized construction that attempts to reach smaller sizes.
    # Or, simply return the LUW graph but optimize the code to be faster?
    # The LUW construction for q=3, d=11 is deterministic.
    
    # Alternative: Use a known small graph from a database if we could, but we can't.
    # Let's try a "greedy" approach to build a 3-regular graph with girth 16.
    # This is very hard.
    
    # Let's stick to the LUW construction but see if we can use a smaller q or d.
    # If q=3, d=9, girth = 14. Not enough.
    # If q=3, d=10, girth = 16.
    # Wait, girth_of_d(10) = 10 + 6 = 16.
    # So LUW(3, 10) has girth 16.
    # Number of vertices = 2 * 3^10 = 2 * 59049 = 118098.
    # This is still very large.
    
    # Is there a construction with smaller degree or different parameters?
    # The task specifies k=3.
    
    # Perhaps the current program is actually producing a much smaller graph?
    # Let's re-read the current program logic.
    # q = k = 3.
    # d = 3. while girth_of_d(d) < 16: d += 1.
    # girth_of_d(3) = 8.
    # girth_of_d(4) = 10.
    # girth_of_d(5) = 10. (5+5=10? No, 5 is odd, so 5+5=10. Wait. 5+5=10. Correct.)
    # girth_of_d(6) = 12.
    # girth_of_d(7) = 12.
    # girth_of_d(8) = 14.
    # girth_of_d(9) = 14.
    # girth_of_d(10) = 16.
    # So d=10.
    # Vertices = 2 * 3^10 = 118098.
    # This is huge. Why is the score 3.0?
    # Maybe the seed value was much larger? Or maybe the score definition is different?
    # "Score: n, minimized; it is an upper bound on the cage order n(k, g)."
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value, so 1.0 equals the seed program".
    # If seed program produced a graph of size S, and this one produces n, score = S/n.
    # If score is 3.0, then S/n = 3.0 => S = 3n.
    # If n=118098, S=354294.
    # This suggests the seed program was even worse.
    
    # Can we do better than 118098?
    # The best known is 936.
    # We need a construction that yields ~1000 vertices.
    
    # One idea: Use a bipartite graph of girth 16.
    # A (3,16)-cage is bipartite? The Moore bound for bipartite graphs is n >= 2 * (k-1)^(g/2 - 1) ...
    # For k=3, g=16: n >= 2 * 2^7 = 256.
    # The Moore bound for general graphs is 512.
    
    # Constructing a specific 936-vertex graph is a known result, but hard to code from scratch without the data.
    # However, we can try a local search or a random restart approach to find a smaller graph.
    # Given the time limit (150s), a sophisticated search might find something.
    
    # Let's try a random 3-regular graph generation and check girth.
    # This is unlikely to succeed for girth 16 in 150s.
    
    # Alternative: Use a known algebraic construction that yields smaller graphs.
    # The "Tutte 8-cage" is 3-regular, girth 8, 30 vertices.
    # "Hoffman-Singleton" is 5-regular, girth 5, 50 vertices.
    
    # There is a construction by Biggs and Smith for cages.
    # For (3,16), the graph is not known to have a simple algebraic form like LUW.
    
    # Since we cannot easily construct the 936-vertex cage, we should probably stick with the LUW construction
    # but try to optimize it or see if a smaller q works?
    # q must be 3 for k=3 in LUW?
    # The LUW construction is for q-ary trees.
    
    # Let's try to implement a quick random search for a graph with n < 118098.
    # Start with n = 1000.
    # Generate random 3-regular graphs.
    # Check girth.
    # This is extremely hard.
    
    # What if we just return the LUW graph?
    # It is valid.
    # Maybe we can speed up the LUW generation?
    # The current code uses Python lists and dicts.
    # We can use numpy for speed?
    # But the structure is complex.
    
    # Let's try to see if we can use a smaller d with a different rule?
    # No, the rule is fixed.
    
    # I will return the LUW construction as it is known to be valid and the score is already 3.0.
    # I will add a check to see if we can break early if we find a smaller graph? No, LUW is deterministic.
    
    # One improvement: if the deadline is approaching, ensure we return a valid graph.
    # The LUW graph is always valid if we finish.
    # With n=118098, and 3 neighbors each, we have ~177k edges.
    # The loop runs for n vertices.
    # 118098 iterations.
    # Each iteration does some math.
    # Should be fast enough in Python?
    # 100k iterations * 10 steps = 1M ops. Very fast.
    # So the LUW graph is generated quickly.
    
    # Is there any way to get a smaller graph?
    # The only other option is to try to find a (3,16)-cage via search.
    # Given the difficulty, I will stick with LUW.
    
    # Wait, is there a bug in the girth calculation?
    # girth_of_d(d) for LUW(q,d) is d+5 if d odd, d+6 if d even.
    # For d=10, girth=16.
    # For d=9, girth=14.
    # So d=10 is the minimum.
    
    # Let's verify the number of vertices.
    # Points: (y_1, ..., y_d) in F_q^d.
    # Lines: (y_1, ..., y_d) in F_q^d.
    # Total 2*q^d.
    # q=3, d=10 -> 2 * 59049 = 118098.
    
    # I will proceed with this.

    q = 3
    d = 10  # Minimum d for girth 16
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    index = {}
    order = []
    edges = []

    start = (0, (0,) * d)  # First point
    index[start] = 0
    order.append(start)

    for i, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            y = [first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # Indices in tuple are 0-based, but rule uses 1-based?
                # Let's check the seed code.
                # y = [None, first] + [0] * (d - 1)
                # y[c] where c is 2..d.
                # x[c] where c is 2..d.
                # In the seed code, x is (None,) + (0,)*d.
                # So x[0] is None, x[1] is first coordinate?
                # Let's trace carefully.
                # start = (0, (None,) + (0,) * d)
                # So x has length d+1.
                # x[0] = None.
                # x[1..d] = 0.
                
                # In the loop:
                # y = [None, first] + [0] * (d - 1)
                # y[0] = None.
                # y[1] = first.
                # y[2..d] = 0.
                
                # for c in range(2, d + 1):
                # a, b = rule[c]
                # if side == 0:
                #   y[c] = add(x[c], mul(y[a], x[b]))
                # Note: y[a] where a is from rule.
                # rule[2] = (1, 1). a=1, b=1.
                # y[1] is 'first'.
                # x[1] is 0 (for start vertex).
                # So y[2] = 0 + first * 0 = 0.
                
                # Let's replicate this exactly to ensure validity.
                pass
            
            # Re-doing the loop with correct indexing
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
        
        if time.time() > deadline:
            break

    n = len(order)
    return n, edges
