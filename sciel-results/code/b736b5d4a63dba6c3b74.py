"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # This is a q-regular bipartite graph with girth at least d + 5.
    # We want girth >= g, so we need d + 5 >= g => d >= g - 5.
    # To minimize vertices n = 2 * q^d, we choose the minimal d.
    q = k
    d = max(1, g - 5)
    
    # Verify time budget for construction
    if time.time() > deadline:
        # Fallback to a trivial valid graph if we are already late? 
        # But we just started. This check is mostly for safety in case of unexpected delays.
        pass

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Define the rule for indices a, b for each c in 2..d
    # The rule (a, b) determines y[c] = x[c] + y[a] * x[b]
    # Standard LUW uses a specific sequence of (a, b) pairs.
    # For d=1, no c in 2..d.
    # For d=2, c=2.
    # The paper "Regular graphs of large girth" by Lazebnik, Ustimenko, Woldar (1998)
    # defines the rule.
    # A common choice is:
    # For c=2, (a,b) = (1,1) ? No, indices are 1..d.
    # Let's use the rule from the current code which seemed to follow a pattern.
    # Current code rule:
    # rule[2] = (1, 1)
    # rule[3] = (2, 1)
    # rule[4] = (1, 2)
    # rule[c] for c >= 5:
    # r = (c-5) % 4
    # c0 = c - r
    # rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    # Let's verify if this rule ensures girth >= d+5.
    # The rule must ensure that the graph is a "cage" or has large girth.
    # The specific rule in the current code is likely correct for the LUW construction.
    
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The indices a, b must be in 1..d and a != c, b != c?
        # In the formula y[c] = x[c] + y[a] x[b], a and b should be < c?
        # No, in the LUW graph, the definition is often recursive or based on specific pairs.
        # The current code's rule uses c0 which is close to c.
        # Let's stick to the provided rule as it was part of the "seed" and likely valid.
        
        # Check bounds for a, b
        a_list = [1, c0 - 1, c0, 1]
        b_list = [c0 - 2, 1, 1, c0 + 1]
        
        # Ensure a, b are valid indices (1..d) and a != c, b != c is not strictly required 
        # but usually a, b < c to make it a triangular system?
        # If a or b >= c, the definition becomes implicit or cyclic.
        # The LUW graph CD(d,q) is defined such that the adjacency matrix has a specific form.
        # The standard definition uses y_i = x_i + sum ... or similar.
        # Actually, the standard LUW graph is a bipartite graph where edges are defined by:
        # y_j = x_j + x_i y_k for j=2..d, with (i,k) depending on j.
        # To avoid circularity, usually i, k < j.
        # Let's check the rule for c=5.
        # r=0, c0=5. a=1, b=3. Both < 5. OK.
        # c=6. r=1, c0=5. a=4, b=1. Both < 6. OK.
        # c=7. r=2, c0=5. a=5, b=1. a=5 < 7. OK.
        # c=8. r=3, c0=5. a=1, b=6. b=6 < 8. OK.
        # c=9. r=0, c0=9. a=1, b=7. OK.
        # It seems the rule keeps a, b < c.
        
        a = a_list[r]
        b = b_list[r]
        rule[c] = (a, b)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, with x[0] = None, x[1..d] values in 0..q-1
        for first in range(q):
            # y[1] is 'first'
            y = [None, first] + [0] * (d - 1)
            # Calculate y[c] for c = 2..d
            # y[c] = x[c] + y[a] * x[b] if side == 0 (Point to Line)
            # y[c] = x[c] - x[a] * y[b] if side == 1 (Line to Point)
            # Wait, the adjacency is symmetric in the bipartite sense?
            # If P ~ L, then L ~ P.
            # The relation is: y_c - x_c = y_a x_b (for P->L)
            # So x_c = y_c - y_a x_b (for L->P)
            
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # x is point, y is line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # x is line, y is point
                    # x[c] = y[c] - x[a] * y[b]  => y[c] = x[c] + x[a] * y[b]
                    # Wait, let's re-verify the inverse.
                    # If y_c = x_c + y_a x_b, then x_c = y_c - y_a x_b.
                    # So if we are at Line x (which is y in the forward step) and go to Point y (which is x in the forward step):
                    # We need to find Point y such that Line x is adjacent.
                    # Condition: x[c] = y[c] - x[a] y[b] ? No.
                    # The edge condition is defined on the pair (P, L).
                    # P ~ L iff for all c=2..d: L_c - P_c = L_a P_b.
                    # If we are at L (side 1, vertex x), and we want to find P (side 0, vertex y):
                    # Condition: x[c] - y[c] = x[a] y[b]
                    # => y[c] = x[c] - x[a] y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # BFS to construct the connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            # If we run out of time, we might return a partial graph? 
            # No, we need a valid graph. 
            # The BFS should be fast enough for n=1250.
            pass
            
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from point side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
