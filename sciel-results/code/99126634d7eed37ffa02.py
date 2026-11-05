import time
import math

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    # d is chosen such that girth is at least d + 5 (for odd d) or d + 6?
    # The seed uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # This ensures girth >= g.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

def _cycle_construct(g):
    """
    Construct a cycle graph C_g (2-regular, girth g).
    """
    n = g
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, edges

def _tensor_product(n1, edges1, n2, edges2):
    """
    Compute the tensor (Kronecker) product of two graphs.
    G1 is n1 vertices, G2 is n2 vertices.
    Result has n1 * n2 vertices.
    Edge (u, u') ~ (v, v') iff (u, v) in E1 and (u', v') in E2.
    """
    n = n1 * n2
    # Map (u, u') to u * n2 + u'
    edges = []
    # For efficiency, iterate over edges of G1 and G2
    # But we need to generate all edges in the product.
    # E_product = { ((u, u'), (v, v')) | (u, v) in E1, (u', v') in E2 }
    # We can iterate over all pairs of edges? No, that's |E1| * |E2|.
    # For k1 * k2 regular, |E1| = n1 * k1 / 2, |E2| = n2 * k2 / 2.
    # Total edges = n1 * n2 * k1 * k2 / 2.
    
    # Construct adjacency lists for G1 and G2 for faster lookup?
    # Or just nested loops over edges.
    # Since we need the list of edges, we generate them.
    
    # Optimization: 
    # For each edge (u, v) in G1, and (u', v') in G2, add ((u*n2+u'), (v*n2+v'))
    # But we must avoid duplicates and ensure simple graph.
    # Tensor product of simple graphs is simple.
    
    # To speed up, we can pre-compute the edge list.
    for u, v in edges1:
        for u_prime, v_prime in edges2:
            node1 = u * n2 + u_prime
            node2 = v * n2 + v_prime
            # Ensure unique edge representation (min, max)
            if node1 < node2:
                edges.append((node1, node2))
            else:
                edges.append((node2, node1))
                
    return n, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Check if k is composite and allows a better tensor product
    # Strategy: Find factors k1, k2 such that k1 * k2 = k.
    # We want to minimize n = n1 * n2.
    # n1 ~ q1^d1, n2 ~ q2^d2.
    # For prime k, only 1 * k.
    # For composite k, we can split.
    
    best_n = float('inf')
    best_edges = []
    
    # Candidate factor pairs
    factors = []
    for i in range(1, int(math.isqrt(k)) + 1):
        if k % i == 0:
            factors.append((i, k // i))
            if i != k // i:
                factors.append((k // i, i))
    
    # Try each factorization
    # Note: For k=5, factors are (1,5) and (5,1).
    # k1=1 is not a valid graph degree for LUW (q=1 invalid).
    # We require k1 >= 2 for LUW.
    # If k1=1, we can use a matching? No, matching is 1-regular.
    # Tensor product of 1-regular (matching) and k-regular is k-regular.
    # But 1-regular graph is just disjoint edges. Girth is infinite (no cycles).
    # Tensor product with a graph of girth g:
    # If G1 is matching (no cycles), G1 x G2 has girth = girth(G2).
    # Vertices: n1 * n2.
    # If G1 is a single edge (2 vertices), n1=2.
    # Then n = 2 * n2.
    # This is just two copies of G2.
    # This doesn't help reduce n.
    
    # So we need k1, k2 >= 2.
    valid_factors = [(k1, k2) for k1, k2 in factors if k1 >= 2 and k2 >= 2]
    
    if not valid_factors:
        # Prime k or k=1,2,3
        # Use LUW directly
        n, edges = _luw_construct(k, g)
        return n, edges
    
    # Try all valid factorizations
    for k1, k2 in valid_factors:
        if time.time() > deadline:
            break
            
        # Construct G1
        n1, edges1 = _luw_construct(k1, g)
        
        # Construct G2
        # If k2 == 2, use Cycle C_g
        # If k2 > 2, use LUW
        if k2 == 2:
            n2, edges2 = _cycle_construct(g)
        else:
            n2, edges2 = _luw_construct(k2, g)
            
        # Tensor Product
        n, edges = _tensor_product(n1, edges1, n2, edges2)
        
        if n < best_n:
            best_n = n
            best_edges = edges
            
    # Fallback to direct LUW if tensor product didn't yield a better result or failed
    # Actually, direct LUW is always a candidate.
    n_direct, edges_direct = _luw_construct(k, g)
    if n_direct < best_n:
        best_n = n_direct
        best_edges = edges_direct
        
    return best_n, best_edges
