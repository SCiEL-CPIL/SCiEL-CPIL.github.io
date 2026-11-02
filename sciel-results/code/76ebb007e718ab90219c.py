"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least 2d + 2. We select the smallest d such that 2d + 2 >= g.
    # This minimizes the number of vertices n ~ 2 * q^d.
    
    # Determine minimal d
    # Girth lower bound is 2d + 2. We need 2d + 2 >= g => d >= (g - 2) / 2.
    # d must be at least 3 for the construction to be valid/interesting in this context.
    d = max(3, (g - 2) // 2 + 1 if (g - 2) % 2 != 0 else (g - 2) // 2)
    # Simplified: d = max(3, ceil((g-2)/2))
    # Let's just use integer math carefully
    # If g=9, (9-2)=7, 7//2=3, need 4? 
    # 2*3+2 = 8 < 9. So d=3 is not enough.
    # 2*4+2 = 10 >= 9. So d=4 is enough.
    # Formula: d = max(3, (g - 2 + 1) // 2) ?
    # g=9: (7+1)//2 = 4. Correct.
    # g=10: (8+1)//2 = 4. 2*4+2=10 >= 10. Correct.
    # g=11: (9+1)//2 = 5. 2*5+2=12 >= 11. Correct.
    # g=8: (6+1)//2 = 3. 2*3+2=8 >= 8. Correct.
    
    d = max(3, (g - 1) // 2)
    # Wait, (g-1)//2 for g=9 is 4. For g=10 is 4. For g=11 is 5.
    # Let's verify g=7: (6)//2 = 3. 2*3+2=8 >= 7. Correct.
    # So d = max(3, (g - 1) // 2) seems to be the ceiling of (g-2)/2?
    # ceil((g-2)/2) = floor((g-2+1)/2) = floor((g-1)/2).
    # Yes.
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # The rule for CD(d, q)
    # The code provided in the prompt uses a specific rule mapping.
    # We keep the rule generation from the original code as it is likely correct for the standard LUW construction.
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
                
    return len(order), edges
# EVOLVE-BLOCK-END
