import time
import math

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for explicit,
    verifiable high-girth regular graphs.
    """
    deadline = time.time() + 110  # Safety margin: 150s total, leave 40s for validation/return
    
    # LUW construction CD(d, q):
    # - q is a prime or 4
    # - d is odd, >= 3
    # - Graph is q-regular, bipartite
    # - Girth is at least d + 5 (if d is odd)
    # - Number of vertices: 2 * q^d (but we only build the connected component of (0, (0,...,0)))
    #   Actually, the full graph has 2*q^d vertices. The connected component might be smaller.
    #   For q=5, d=4: 2*5^4 = 1250 vertices.
    #   Wait, d must be odd for the girth guarantee d+5.
    #   For g=9, we need girth >= 9.
    #   If d is odd, girth >= d+5.
    #   d+5 >= 9 => d >= 4. But d must be odd, so d >= 5.
    #   If d=5, girth >= 10. This is valid for g=9.
    #   Number of vertices: 2 * 5^5 = 6250.
    #   The Moore bound for n(5,9) is 1 + 5 + 5*4 + 5*4^2 + ... 
    #   For bipartite, girth 10, Moore bound is 2 * sum_{i=0}^{3} 5^i = 2*(1+5+25+125) = 312.
    #   6250 is much larger, but valid.
    #   
    #   Can we do better?
    #   q must be >= k=5. Smallest prime >= 5 is 5.
    #   d=5 gives 6250 vertices.
    #
    #   What if we use q=5, d=4? d is even.
    #   For even d, the girth is at least d+4? Or is it not guaranteed?
    #   The LUW paper states: for d odd, girth >= d+5.
    #   For d even, the construction still works but the girth bound is different or not stated.
    #   Let's stick to odd d to be safe.
    #
    #   Is there a better q? No, q must be >= k.
    #   So q=5, d=5 is the standard LUW choice for k=5, g=9 (actually g=10).
    #
    #   Let's verify the parameters.
    #   k=5, g=9.
    #   q=5.
    #   d=5 (odd).
    #   Girth >= 5+5=10 >= 9. Correct.
    #   Vertices: 2 * 5^5 = 6250.
    #   Edges: 5 * 6250 / 2 = 15625.
    #
    #   This is valid. Is it fast enough?
    #   Building 6250 vertices and 15625 edges should be very fast in Python.
    #   The BFS/DFS to find the connected component of (0, 0) will traverse all of them if connected.
    #   LUW graphs are connected for prime q and d>=3? Yes, typically.
    
    q = k  # 5
    # Determine d
    # We need girth >= g.
    # LUW with odd d has girth >= d + 5.
    # So d + 5 >= g => d >= g - 5.
    # d must be odd.
    min_d = g - 5
    if min_d < 3:
        d = 3
    else:
        # Next odd number >= min_d
        if min_d % 2 == 0:
            d = min_d + 1
        else:
            d = min_d
    
    # For k=5, g=9: d >= 4. Next odd is 5.
    
    # Field operations for GF(q). q=5 is prime.
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q
    def neg(x): return (-x) % q
    
    # The rule for LUW CD(d, q).
    # Variables are x_1, ..., x_d and y_1, ..., y_d.
    # Equations:
    # y_1 = x_1
    # For i >= 2:
    #   If i is even: y_i = x_i + y_{i-1} * x_{i-1}  (Wait, let me check the standard definition)
    #   Actually, the LUW construction CD(d, q) is defined as follows:
    #   Bipartite graph with vertices (x, y) in F_q^d x F_q^d.
    #   (x, y) ~ (x', y') iff:
    #     y_1 = x'_1
    #     y_i = x'_i + x'_{i-1} y_{i-1}  for i=2..d (if side 0 to 1)
    #     OR
    #     x_1 = y'_1
    #     x_i = y'_i + y'_{i-1} x_{i-1}  for i=2..d (if side 1 to 0)
    #   This is a recursive definition.
    #   Let's implement the forward direction from a vertex (side, vector).
    #   If side=0 (vertex is x), neighbors are y vectors.
    #   For each choice of y_1 in F_q:
    #     Compute y_2, ..., y_d from x and y_1.
    #     (x, y_1..y_d) is a neighbor.
    #   If side=1 (vertex is y), neighbors are x vectors.
    #   For each choice of x_1 in F_q:
    #     Compute x_2, ..., x_d from y and x_1.
    #     (x_1..x_d, y) is a neighbor.
    
    # Let's use BFS from (0, (0,0,0,0,0)) to build the graph.
    # Vertex representation: (side, tuple of d integers)
    # side=0: x-vertex, side=1: y-vertex
    
    start = (0, tuple([0]*d))
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, vec = vertex
        for first_val in range(q):
            new_vec = [0] * d
            if side == 0:
                # x -> y
                # y_1 = first_val (which is x'_1? No, the neighbor is y)
                # Let's re-read the adjacency.
                # (x, y) adjacent to (x', y') if y = f(x, x'_1) and x' = g(y, y'_1)?
                # Standard LUW:
                # (x, y) ~ (x', y') iff:
                #   y_1 = x'_1
                #   y_i = x'_i + x'_{i-1} * y_{i-1}
                # This means given x and x'_1, we can determine y? No.
                # Given x and x'_1, we can compute y?
                # y_1 = x'_1.
                # y_2 = x'_2 + x'_1 * y_1. But x'_2 is unknown.
                # So we cannot compute y from x and x'_1 alone.
                
                # Let's look at the other direction:
                # Given y and y'_1 (which is x_1), can we compute x?
                # x_1 = y'_1.
                # x_2 = y'_2 + y'_1 * x_1? No.
                
                # The standard form is:
                # (x, y) ~ (x', y') iff:
                #   y_1 = x'_1
                #   x_i = y'_i + y'_{i-1} * x_{i-1}  (for i=2..d)  <-- This determines x' from y and x'_1?
                #   Wait, if we know y (from the first vertex) and x'_1, we can compute x'_2, x'_3...
                #   So given a vertex (x, y) [side 1? No, let's say we are at (x, y) where x is the "first" part],
                #   and we pick a neighbor's "first" coordinate x'_1, we can compute the rest of x'.
                #   Then the neighbor is (x', y).
                #   But we also need the neighbor's "second" part to be consistent?
                #   Actually, in the LUW graph, the vertices are pairs (x, y).
                #   The adjacency is defined by:
                #   (x, y) ~ (x', y') iff:
                #     y_1 = x'_1
                #     x_i = y'_i + y'_{i-1} x_{i-1}  (i=2..d)
                #   This equation involves x, y', and x'.
                #   It does NOT involve y directly, except through the first condition?
                #   No, it involves y'_i.
                #   This looks like it defines x' in terms of y' and x.
                #   But we don't know y'.
                
                # Let's use the formulation from the original LUW paper or standard references.
                # The graph CD(d, q) has vertex set V = F_q^d x F_q^d.
                # Two vertices (x, y) and (x', y') are adjacent iff:
                #   y_1 = x'_1
                #   x_i = y'_i + y'_{i-1} * x_{i-1}  for i = 2, ..., d
                #   (Here x, y, x', y' are in F_q^d)
                #   
                # Let's check if this is k-regular.
                # Fix (x, y). How many (x', y') are adjacent?
                # We can choose y' freely?
                # If we choose y' freely, then x'_1 = y_1 is fixed.
                # And x'_i is determined by x_{i-1}, y'_{i-1}, y'_i.
                # Specifically, x'_2 = y'_2 + y'_1 * x_1.
                # x'_3 = y'_3 + y'_2 * x_2.
                # ...
                # So for any choice of y' in F_q^d, there is a unique x' such that (x, y) ~ (x', y').
                # Thus, the degree is q^d.
                # We want degree k=5.
                # So q^d = 5?
                # If q=5, d=1, degree=5. But d must be >=3 for high girth.
                # If d=1, girth is small.
                # 
                # Wait, the standard LUW graph CD(d, q) has degree q.
                # How?
                # Let's re-read.
                # Maybe the adjacency is:
                # (x, y) ~ (x', y') iff:
                #   x_1 = y'_1
                #   y_i = x'_i + x'_{i-1} * y_{i-1}
                #   If we fix (x, y) and choose x'_1, then y_1 = x'_1? No.
                #   
                # Correct LUW adjacency (from Lazebnik, Ustimenko, Woldar 1998):
                # Vertices are (x, y) in F_q^d x F_q^d.
                # (x, y) ~ (x', y') iff:
                #   y_1 = x'_1
                #   x_i = y'_i + y'_{i-1} * x_{i-1}  for i=2..d
                #   AND
                #   x_1 = y'_1  ?? No, that would be symmetric.
                #
                # Let's look at the "incidence" version.
                # Actually, the graph is bipartite with parts X and Y.
                # X = F_q^d, Y = F_q^d.
                # x in X is adjacent to y in Y iff:
                #   y_1 = x_1
                #   x_i = y_i + y_{i-1} * x_{i-1}  for i=2..d
                #   Let's check degree.
                #   Fix x in X.
                #   Choose y_1 freely? No, y_1 = x_1 is fixed.
                #   Choose y_2 freely?
                #   x_2 = y_2 + y_1 * x_1 => y_2 = x_2 - y_1 * x_1. Fixed.
                #   So y is uniquely determined by x?
                #   Then degree is 1. That's not right.
                #
                # Let's try:
                # x ~ y iff:
                #   x_1 = y_1
                #   y_i = x_i + x_{i-1} * y_{i-1}  for i=2..d
                #   Fix x.
                #   y_1 = x_1.
                #   y_2 = x_2 + x_1 * y_1. Fixed.
                #   Degree 1.
                #
                # Okay, the standard LUW graph CD(d, q) is q-regular.
                # The adjacency is:
                # (x, y) ~ (x', y') iff:
                #   y_1 = x'_1
                #   x'_i = x_i + y_{i-1} * y'_i  ??
                #
                # Let's use the explicit construction from the seed code provided in the prompt.
                # The seed code uses:
                #   rule[c] = (a, b)
                #   y_list[c] = add(x[c], mul(y_list[a], x[b]))  # for side 0
                #   y_list[c] = sub(x[c], mul(x[a], y_list[b]))  # for side 1
                #
                # This implies:
                # Side 0 (x-vertex) to Side 1 (y-vertex):
                #   y_1 = first (chosen)
                #   y_c = x_c + y_a * x_b
                # Side 1 (y-vertex) to Side 0 (x-vertex):
                #   x_1 = first (chosen)
                #   x_c = y_c - x_a * y_b
                #
                # The rule (a, b) depends on c.
                # In the seed:
                #   for c in range(5, d+1):
                #       r, c0 = (c-5)%4, c-(c-5)%4
                #       rule.append([(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r])
                #
                # For c=2,3,4 the rule is fixed:
                #   rule[2] = (1, 1)
                #   rule[3] = (2, 1)
                #   rule[4] = (1, 2)
                #
                # Let's verify degree.
                # For side 0, we choose y_1 (q choices).
                # Then y_2, y_3... are determined.
                # So each x-vertex has q neighbors.
                # Thus the graph is q-regular.
                # We want k=5, so q=5.
                #
                # Girth: The seed says girth >= d+5 for odd d.
                # For k=5, g=9, we need d such that d+5 >= 9 => d >= 4.
                # d must be odd, so d=5.
                #
                # So we use the seed's logic exactly, but ensure it's robust.
                
                # The seed's rule generation:
                # rule[1] = None
                # rule[2] = (1, 1)
                # rule[3] = (2, 1)
                # rule[4] = (1, 2)
                # for c in 5..d:
                #   r = (c-5) % 4
                #   c0 = c - r
                #   rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
                
                rule = [None, None, (1, 1), (2, 1), (1, 2)]
                for c in range(5, d + 1):
                    r = (c - 5) % 4
                    c0 = c - r
                    if r == 0:
                        rule.append((1, c0 - 2))
                    elif r == 1:
                        rule.append((c0 - 1, 1))
                    elif r == 2:
                        rule.append((c0, 1))
                    else:
                        rule.append((1, c0 + 1))
                
                # Now compute neighbors
                new_vec_list = [0] * d
                new_vec_list[0] = first_val  # y_1
                for c in range(2, d + 1):
                    a, b = rule[c]
                    new_vec_list[c - 1] = add(vec[c - 1], mul(new_vec_list[a - 1], vec[b - 1]))
                neighbor = (1, tuple(new_vec_list))
                
            else:
                # side 1 (y-vertex) to side 0 (x-vertex)
                # x_1 = first_val
                # x_c = y_c - x_a * y_b
                new_vec_list = [0] * d
                new_vec_list[0] = first_val
                for c in range(2, d + 1):
                    a, b = rule[c]
                    new_vec_list[c - 1] = sub(vec[c - 1], mul(new_vec_list[a - 1], vec[b - 1]))
                neighbor = (0, tuple(new_vec_list))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            edges.append((i, index[neighbor]))
    
    n = len(order)
    # The graph might not be connected if we only BFS from start.
    # But LUW graphs are connected for prime q and d>=3.
    # To be safe, we should check if all 2*q^d vertices are reached.
    # If not, we might have a smaller component. But the task asks for a k-regular graph.
    # A disconnected graph is not k-regular if some components have different sizes?
    # No, if each component is k-regular, the whole graph is k-regular.
    # But the girth is the minimum girth over all components.
    # If we only include the connected component of start, it is k-regular (if BFS completes) and has the correct girth.
    # However, if the BFS doesn't reach all vertices, the resulting graph is smaller, which is better!
    # But we must ensure it is k-regular.
    # In the BFS, each vertex we visit has exactly q neighbors added (or already present).
    # When we finish, do all vertices have degree q?
    # Yes, because for each vertex, we generated all q neighbors.
    # If a neighbor was already visited, we added an edge to it.
    # So the degree of each visited vertex is exactly q.
    # Thus, the induced subgraph on the visited vertices is q-regular.
    #
    # So returning the BFS component is valid and potentially smaller than 2*q^d.
    
    return n, edges
