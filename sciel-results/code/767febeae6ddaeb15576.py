"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _count_c4(adj):
    """Count 4-cycles in a graph given its adjacency matrix."""
    # Number of C4s = (1/8) * sum over all pairs (u,v) of C(common_neighbors(u,v), 2)
    # But more efficiently: C4 count = (1/8) * sum_{u,v} (A^2)_{uv}^2 - ... 
    # Actually: number of 4-cycles = (1/4) * sum_{u,v} C((A^2)_{uv}, 2)
    # Wait, let me think again.
    # A 4-cycle u-v-w-x-u is counted by pairs of vertices (u,w) that share 2 common neighbors (v,x).
    # For each pair (u,w), if they share k common neighbors, they contribute C(k,2) to the count.
    # Each 4-cycle is counted exactly once this way (the two opposite pairs).
    # So total C4s = sum_{u<w} C(common(u,w), 2)
    n = adj.shape[0]
    common = adj @ adj  # (A^2)_{uv} = number of common neighbors of u and v
    total = 0
    for u in range(n):
        for w in range(u + 1, n):
            k = common[u, w]
            if k >= 2:
                total += k * (k - 1) // 2
    return total

def _build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5).
    
    PG(2,5) has 31 points. The polarity graph has vertices as points of PG(2,5),
    with an edge between two points if and only if they are perpendicular with respect
    to a suitable polarity.
    
    For q=5 (odd), the polarity graph of PG(2,q) is C4-free and has q(q+1)/2 + (q-1)/2 = 90 edges.
    Actually, the number of edges is q(q+1)/2 = 15... no.
    
    Let me reconsider. The polarity graph of PG(2,q) for odd q has:
    - n = q^2 + q + 1 = 31 vertices
    - Each vertex has degree q = 5 (since each point is perpendicular to q+1 points including itself,
      so q other points are perpendicular to it... wait, for a correlation polarity in PG(2,q),
      each point is self-perpendicular and perpendicular to q other points, so degree q)
    - Total edges = 31 * 5 / 2 = 77.5... that's not right.
    
    Actually for a polarity in PG(2,q) with q odd: each point is absolute (self-perpendicular),
    and each point is perpendicular to exactly q other points. So degree = q, total edges = 31*5/2 = 77.5.
    That's not an integer, so something is off.
    
    Let me look at this differently. The polarity graph of PG(2,q) for q odd:
    - n = q^2 + q + 1 vertices
    - Each vertex has degree q
    - Total edges = n*q/2 = (q^2+q+1)*q/2
    
    For q=5: 31*5/2 = 77.5. Still not integer.
    
    Hmm, let me reconsider. For a non-singular polarity in PG(2,q) with q odd:
    - The number of absolute points is q+1
    - Each absolute point is perpendicular to q-1 other points (plus itself)
    - Each non-absolute point is perpendicular to q other points
    - So edges from absolute points: (q+1) * (q-1) / 2
    - Edges from non-absolute points: (q^2 - q) * q / 2
    - Total = (q+1)(q-1)/2 + (q^2-q)q/2 = (q^2-1)/2 + q^3-q^2/2 = ...
    
    This is getting complicated. Let me just construct it directly.
    
    For PG(2,5), we can represent points as 1D subspaces of F_5^3.
    A polarity can be defined by a symmetric matrix M. Two points [x] and [y] are
    perpendicular if x^T M y = 0.
    
    For q=5, a standard choice is the Hermitian polarity, but since 5 is not a square
    in F_5, let's use a symmetric bilinear form.
    
    Actually, let me use a simpler approach. The orthogonal polarity in PG(2,q) for q odd
    can be defined using the symmetric matrix:
    M = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    
    But I need to verify this gives a C4-free graph with 90 edges.
    
    Let me try a different approach: use the known structure. The polarity graph of
    PG(2,5) is a (31, 5, 0, 1) or similar design... 
    
    Actually, I recall that for the polarity graph of PG(2,q) with q odd:
    - It's a regular graph of degree q
    - It has n = q^2 + q + 1 vertices
    - Number of edges = n*q/2
    
    For this to be an integer, n*q must be even. For q=5, n=31, 31*5=155, which is odd.
    So degree can't be exactly 5 for all vertices.
    
    Let me just construct it properly. In PG(2,5), there are 31 points. I'll enumerate them
    as canonical representatives and define a polarity.
    """
    q = 5
    # Enumerate all 1D subspaces of F_q^3
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Canonical form: first non-zero coordinate is 1
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)  # modular inverse
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    # Define a polarity using a symmetric matrix
    # Use M = [[1, 0, 0], [0, 1, 0], [0, 0, -1]] = [[1,0,0],[0,1,0],[0,0,4]]
    # Two points x, y are perpendicular if x^T M y = 0
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int)
    
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(points[i], dtype=int)
            y = np.array(points[j], dtype=int)
            val = (x @ M @ y) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
    
    return edges, adj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Build the polarity graph (C4-free, 90 edges)
    edges_90, adj = _build_polarity_graph()
    
    # Count current C4s (should be 0)
    c4_base = _count_c4(adj)
    
    # Find the best edge to add (the one that creates the fewest C4s)
    best_edge = None
    best_c4 = float('inf')
    
    n = 31
    # Try all possible edges not in the graph
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                # Adding edge (i,j) creates C4s for each pair of common neighbors of i and j
                # that are also connected... actually, adding edge (i,j) creates a C4
                # for each path of length 3 between i and j that doesn't use the direct edge.
                # A C4 containing the new edge (i,j) has the form i-a-b-j-i where a,b are
                # common neighbors of i and j... no, a is neighbor of i, b is neighbor of j,
                # and a-b is an edge.
                # So C4s created = number of edges (a,b) where a in N(i), b in N(j), a != b,
                # and (a,b) is an edge.
                
                ni = np.where(adj[i] == 1)[0]
                nj = np.where(adj[j] == 1)[0]
                
                c4s_created = 0
                for a in ni:
                    for b in nj:
                        if a != b and adj[a, b] == 1:
                            c4s_created += 1
                
                # Each C4 is counted once
                total_c4 = c4_base + c4s_created
                
                if total_c4 < best_c4:
                    best_c4 = total_c4
                    best_edge = [i, j]
                
                # Check time
                if time.time() > deadline - 5.0:
                    break
        if time.time() > deadline - 5.0:
            break
    
    # If we found a good edge, add it
    if best_edge is not None and time.time() < deadline - 2.0:
        edges_90.append(best_edge)
        return edges_90
    else:
        # Fallback: add the first available edge
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    edges_90.append([i, j])
                    return edges_90
    
    return edges_90

# EVOLVE-BLOCK-END
