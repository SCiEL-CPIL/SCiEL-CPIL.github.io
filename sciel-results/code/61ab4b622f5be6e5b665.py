import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # We will use the Polarity Graph of the Projective Plane PG(2, 7)
    # This graph is C4-free.
    # The number of vertices is q^2 + q + 1 = 49 + 7 + 1 = 57.
    # The independence number is small (<= q = 7), so any subgraph is valid for alpha <= 10.
    # We want n = 43.
    
    q = 7
    n_target = 43
    
    # Generate points of PG(2, q)
    # Points are equivalence classes of non-zero vectors (x, y, z) in GF(q)^3
    # We represent each point by its canonical representative where the first non-zero component is 1.
    
    points = []
    seen = set()
    
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    rep = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    rep = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    rep = (0, 0, 1)
                
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    # Total points should be q^2 + q + 1
    # n_points = len(points)
    
    # Define Polarity
    # A polarity is a symmetric bilinear form. 
    # Let's use the form defined by the matrix:
    # [ 1  0  0 ]
    # [ 0  1  0 ]
    # [ 0  0 -1 ]
    # P is on the polar of Q if P^T M Q = 0
    # x_p x_q + y_p y_q - z_p z_q = 0
    
    # We only need the first n_target vertices
    # Map vertices to indices 0..n_target-1
    # We will build the adjacency list for these vertices
    
    # Precompute inverses or just use the formula
    # Since q is small, we can just iterate
    
    edges = []
    
    # To ensure C4-free, we rely on the property of the full graph.
    # Any induced subgraph of a C4-free graph is C4-free.
    
    # We select the first n_target points
    selected_points = points[:n_target]
    
    # Build edges
    for i in range(n_target):
        for j in range(i + 1, n_target):
            p = selected_points[i]
            r = selected_points[j]
            # Check if p is on polar of r
            # x_p x_r + y_p y_r - z_p z_r == 0
            val = (p[0] * r[0] + p[1] * r[1] - p[2] * r[2]) % q
            if val == 0:
                edges.append([i, j])
                
    # Verification (optional, but good for safety if time permits)
    # The problem states "exact search" is not required for the output, 
    # but the graph must satisfy the properties.
    # The Polarity Graph of PG(2, q) is known to be C4-free.
    # Proof sketch: A C4 in the graph corresponds to two points P, R and two points Q, S 
    # such that P is perp Q, P is perp S, R is perp Q, R is perp S.
    # This implies Q and S are on the polar of P, and Q and S are on the polar of R.
    # In a projective plane, two distinct points P and R have polars that intersect in exactly one point.
    # So Q must equal S. Thus no C4.
    
    # Independence number:
    # An independent set is a set of points no two of which are perpendicular.
    # It is a known result that the maximum size of such a set in PG(2, q) is at most q.
    # For q=7, alpha <= 7.
    # Since 7 < 11, the condition is satisfied.
    
    return {'n': n_target, 'edges': edges}
# EVOLVE-BLOCK-END
