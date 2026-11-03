"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth at least g on as few vertices as possible."""
    start_wall = time.time()
    deadline = start_wall + 140.0  # safety margin before 150s kill

    # The task is specifically about n(5,9): a 5-regular graph of girth >= 9.
    # Best known: 1116 vertices. The Lazebnik-Ustimenko-Woldar (LUW) construction CD(d,q)
    # with q=5, d=4 gives a 5-regular bipartite graph of girth 9 on 2*5^4 = 1250 vertices.
    # We can try to find smaller constructions or verify the LUW one.

    # First, try the LUW construction which is known to work for (k=5, g=9)
    # CD(d, q) with q=5 (prime), d=4: girth = d+5 = 9, order = 2*q^d = 1250
    # But we can also try other approaches.

    # Let's implement the LUW construction directly for k=5, g=9
    # q = 5, d = 4
    q = k
    d = g - 5  # = 4 for g=9

    # For LUW CD(d,q): vertices are (side, x) where side in {0,1}, x in GF(q)^{d+1}
    # x = (x_1, ..., x_{d+1}), with x_1 = None (or 0)
    # Edge between point p=(0, x) and line l=(1, y) iff for all c=2..d+1:
    #   y[c] - x[c] = y[a_c] * x[b_c]
    # where (a_c, b_c) is determined by the rule.

    # The rule for CD(d,q) with q prime:
    # rule[2] = (1,1)
    # For c >= 3: the rule depends on c mod 4
    # Standard LUW rule:
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    # c=5: (1,3)
    # c=6: (2,3)
    # c=7: (3,2)
    # c=8: (1,4)
    # etc.

    # Actually, let me use the standard LUW rule table.
    # The rule is: for position c (2-indexed, c from 2 to d), 
    # rule[c] = (a, b) where a+b is odd and max(a,b) < c.
    # The standard choice:
    # c=2: (1,1)
    # c=3: (2,1)  
    # c=4: (1,2)
    # c=5: (1,3)
    # c=6: (2,3)
    # c=7: (3,2)
    # c=8: (1,4)
    # c=9: (2,4)
    # c=10: (4,3)
    # ...

    # Let me just hardcode the rule for d up to maybe 10
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    rule[5] = (1, 3)
    rule[6] = (2, 3)
    rule[7] = (3, 2)
    rule[8] = (1, 4)
    rule[9] = (2, 4)
    rule[10] = (4, 3)
    rule[11] = (1, 5)
    rule[12] = (2, 5)
    rule[13] = (5, 4)

    # GF(5) operations
    def add(x, y):
        return (x + y) % q

    def sub(x, y):
        return (x - y) % q

    def mul(x, y):
        return (x * y) % q

    # Build the graph using BFS from the zero point
    # Vertex representation: (side, x_tuple) where x_tuple has length d+1
    # x[0] is index 1 (always 0 for points, or we can use it)
    # Actually in the LUW construction, x_1 is fixed to 0 for both points and lines
    # Let me re-read: vertices are (side, x) where x = (x_1, ..., x_{d+1})
    # x_1 = 0 always. So effectively x has d free coordinates.

    # For efficiency, let's represent vertices as integers
    # side * q^d + x_2 * q^(d-1) + x_3 * q^(d-2) + ... + x_{d+1}
    # where x_2, ..., x_{d+1} are in 0..q-1

    def vertex_to_id(side, x):
        """x is a tuple of length d+1, x[0] should be 0"""
        vid = side * (q ** d)
        for i in range(1, d + 1):
            vid += x[i] * (q ** (d - i))
        return vid

    def id_to_vertex(vid):
        side = vid // (q ** d)
        rem = vid % (q ** d)
        x = [0] * (d + 1)
        for i in range(d, 0, -1):
            x[i] = rem % q
            rem //= q
        return side, tuple(x)

    # Build adjacency by iterating over all point vertices and finding their line neighbors
    n_vertices = 2 * (q ** d)
    adj = [[] for _ in range(n_vertices)]

    # For each point (0, x), find all lines (1, y) such that the edge condition holds
    # y[c] = x[c] + y[a_c] * x[b_c] for c = 2..d
    # This determines y uniquely given x, so each point has exactly q neighbors (one for each y_2 value)
    # Wait, y_2 is free? Let me check.
    # For c=2: y[2] = x[2] + y[a_2] * x[b_2] = x[2] + y[1] * x[1] = x[2] + 0 * 0 = x[2]
    # So y[2] is determined by x. Then for c=3: y[3] = x[3] + y[a_3] * x[b_3]
    # a_3=2, b_3=1: y[3] = x[3] + y[2] * x[1] = x[3] + y[2] * 0 = x[3]
    # So actually y is completely determined by x! That means each point has exactly 1 neighbor?
    # That can't be right for a k-regular graph.

    # Let me reconsider. In the LUW construction, the edge condition is:
    # For point p=(0, x) and line l=(1, y), they are adjacent iff:
    # y[c] - x[c] = y[a_c] * x[b_c] for all c = 2, ..., d
    # Here x = (x_1, ..., x_{d+1}) and y = (y_1, ..., y_{d+1}), with x_1 = y_1 = 0.
    # The free variable is y_2 (or x_2 for the point side).
    # Actually, I think I'm confusing the indexing. Let me look at this more carefully.

    # The standard LUW CD(d,q) construction:
    # Vertices: V = {0,1} x GF(q)^d
    # For (s, x) in V, where x = (x_1, ..., x_d), the neighbors are determined by a linear recurrence.
    # 
    # Let me just use a simpler approach: enumerate all vertices and check adjacency.
    # For q=5, d=4: n = 2*625 = 1250 vertices. Checking all pairs is O(n^2) = 1.56M, which is fine.

    # Actually, let me just generate the edges by iterating over each vertex and computing its neighbors.
    # For a point (0, x) where x = (x_1, x_2, x_3, x_4) in GF(5)^4:
    # A line (1, y) is adjacent iff y[c] - x[c] = y[a_c] * x[b_c] for c=2,3,4
    # (using 1-indexed, so x_1 is the first component)
    # 
    # With rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2):
    # c=2: y[2] - x[2] = y[1] * x[1] = 0 * x[1] = 0, so y[2] = x[2]
    # c=3: y[3] - x[3] = y[2] * x[1] = x[2] * x[1]
    # c=4: y[4] - x[4] = y[1] * x[2] = 0 * x[2] = 0, so y[4] = x[4]
    # 
    # So y is uniquely determined by x: y[1]=0, y[2]=x[2], y[3]=x[3]+x[2]*x[1], y[4]=x[4]
    # That gives only 1 neighbor per point, which is wrong.

    # I think the issue is that the LUW construction has a different structure. Let me look at the original paper's description more carefully.
    # 
    # In Lazebnik-Ustimenko-Woldar, the construction is:
    # Let q be a prime power. Let d >= 2. 
    # V = {(0, x) : x in GF(q)^d} union {(1, y) : y in GF(q)^d}
    # (0, x) ~ (1, y) iff y_j = x_j + sum_{i=1}^{j-1} x_i * y_{j-i} for j = 2, ..., d
    # with y_1 = x_1.
    # 
    # Wait, that's a different formulation. Let me try this.
    # For (0, x) and (1, y) to be adjacent:
    # y_1 = x_1
    # y_j = x_j + x_1 * y_{j-1} + x_2 * y_{j-2} + ... + x_{j-1} * y_1 for j = 2, ..., d
    # 
    # This is a recurrence where y is determined by x. So again each point has exactly 1 neighbor?
    # That still doesn't give a regular graph.

    # I think I'm misunderstanding the construction. Let me just use a different approach.
    # 
    # Alternative: Use the known fact that the incidence graph of the generalized quadrangle or hexagon
    # gives regular graphs of high girth. But for (5,9), the best known is 1116 vertices.
    # 
    # Let me try a different strategy: use a random regular graph and check girth, or use
    # the configuration method. But that's complex.
    # 
    # Actually, let me just return the LUW construction as described in the original code,
    # but fix it to work correctly. The original code uses a BFS approach which should work.

    # Let me re-examine the original code's neighbour function:
    # For vertex (side, x) where x has d+1 components (x[0] is None/0):
    # For each "first" in range(q):
    #   y = [None, first] + [0]*(d-1)
    #   for c in range(2, d+1):
    #     a, b = rule[c]
    #     y[c] = x[c] + y[a]*x[b] if side==0 else x[c] - x[a]*y[b]
    #   yield (1-side, y)
    # 
    # So for a point (0, x), we iterate over y[1] = first (0 to q-1), and then compute y[2..d].
    # This gives q neighbors per point, which is correct for a q-regular graph.
    # 
    # The edge condition is: for side=0 (point), y[c] = x[c] + y[a_c] * x[b_c]
    # for side=1 (line), y[c] = x[c] - x[a_c] * y[b_c]  (but here x is the line and y is the point)
    # 
    # Wait, in the original code, when side=1, it computes y[c] = x[c] - x[a]*y[b]
    # where x is the line vertex and y is the point being computed.
    # So the edge condition for a line (1, x) and point (0, y) is:
    # y[c] = x[c] - x[a_c] * y[b_c] for c = 2..d
    # 
    # This is consistent: if (0, x) ~ (1, y), then y[c] = x[c] + y[a_c] * x[b_c] (from point side)
    # and also y[c] = x[c] - x[a_c] * y[b_c] (from line side)
    # These should be equivalent due to the specific rule choices.

    # OK so the original code's construction is correct. Let me just use it but with optimizations.
    # For q=5, d=4: n = 2*5^4 = 1250 vertices, 5*1250/2 = 3125 edges.

    # Let me just run the original LUW construction and return it.
    # The original code does BFS which is fine for 1250 vertices.

    # Actually, let me just copy the original approach but make it more efficient.
    q = k
    d = g - 5  # for g=9, d=4

    # Rule table for LUW
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    rule[5] = (1, 3)
    rule[6] = (2, 3)
    rule[7] = (3, 2)
    rule[8] = (1, 4)
    rule[9] = (2, 4)
    rule[10] = (4, 3)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0  # placeholder
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    start = (0, tuple([0] * (d + 1)))
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

    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
