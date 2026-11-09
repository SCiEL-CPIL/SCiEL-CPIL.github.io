import time
import random

# EVOLVE-BLOCK-START

def _mod_sqrt(a, p):
    """Compute modular square root of a mod p (p prime), return smallest root >= 0 or None."""
    if a == 0:
        return 0
    # For prime p, if p % 4 == 3: sqrt = a^((p+1)/4) % p
    # Else use Tonelli-Shanks
    p_minus_1 = p - 1
    if p_minus_1 % 4 == 3:
        r = pow(a, (p_minus_1 + 1) // 4, p)
        if (r * r) % p == a:
            return min(r, p - r)
        else:
            return None
    # Tonelli-Shanks
    def legendre_symbol(a, p):
        ls = pow(a, (p - 1) // 2, p)
        return -1 if ls == p - 1 else ls

    if legendre_symbol(a, p) != 1:
        return None

    q = p_minus_1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = (t2 * t2) % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c = i, (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    return min(r, p - r)


def _build_polarity_graph(q):
    """Build polarity graph of PG(2, q). Vertices are projective points (q^2+q+1).
    Two points are adjacent iff one is on the polar line of the other.
    Returns list of edges and vertex count."""
    p = q  # q is prime (here 5)
    # Points in PG(2,q): represented as triples (x, y, z) not all zero, up to scalar
    # Canonical form: scale so that first non-zero coord is 1
    points = []
    seen = set()
    for x in range(p):
        for y in range(p):
            for z in range(p):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize: multiply by inverse of first nonzero coord
                if x != 0:
                    inv = pow(x, p - 2, p)
                    key = (1, (y * inv) % p, (z * inv) % p)
                elif y != 0:
                    inv = pow(y, p - 2, p)
                    key = (0, 1, (z * inv) % p)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n = len(points)  # should be q^2 + q + 1 = 31 for q=5
    # Point index map
    pmap = {pt: i for i, pt in enumerate(points)}
    
    # Polarity: for point (x, y, z), polar line is x*X + y*Y + z*Z = 0 (symmetric bilinear form)
    # Actually for PG(2,q), a polarity maps point (x,y,z) to line x*X + y*Y + z*Z = 0
    # A point (X,Y,Z) is on the polar line of (x,y,z) iff x*X + y*Y + z*Z = 0 (mod p)
    # Two points are adjacent iff each is on the other's polar line, which for symmetric form is:
    # x*X + y*Y + z*Z = 0
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            a = points[i]
            b = points[j]
            # Check if a is on polar of b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2] == 0 mod p
            dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) % p
            if dot == 0:
                edges.append([i, j])
    
    return edges, n, points


def _count_c4s(edges):
    """Count number of C4 subgraphs in a graph given as edge list."""
    n = 31
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    c4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            c4_count += len(common)
    # Each C4 is counted twice (once for each pair of opposite vertices)
    # Wait, let me reconsider. A C4 has 4 vertices forming a cycle.
    # For a C4 with vertices a-b-c-d-a, the pairs of opposite vertices are (a,c) and (b,d).
    # So each C4 contributes 2 to the sum (one for each diagonal pair).
    # Therefore c4_count = sum over all pairs of |common neighbors| / 2
    return c4_count // 2


def construct():
    deadline = time.time() + 140  # safety margin
    
    # Build polarity graph of PG(2,5)
    edges_pg, n, points = _build_polarity_graph(5)
    
    # The polarity graph should be C4-free
    # For q=5, it has q^2+q+1 = 31 vertices
    # Number of edges: each vertex has degree (q+1)/2 * ... let me just count
    # Actually, in the polarity graph of PG(2,q) with q odd, the number of edges is
    # (q^2+q+1) * q / 2 = 31 * 5 / 2 = 77.5 ... that's not integer.
    # Let me reconsider. The degree of each vertex in the polarity graph is (q-1)/2 + 1 = (q+1)/2 for absolute points
    # Actually, this is getting complicated. Let me just use the edges we computed.
    
    # edges_pg should have some number of edges. For q=5, I expect around 77 or 80 edges.
    # But we need 91 edges total. The task says ex(31;C4) = 90, meaning max C4-free edges is 90.
    # So the polarity graph might have fewer than 90 edges, or maybe it's a different construction.
    
    # Let me try a different approach: use the known C4-free construction with 90 edges,
    # then add one edge that creates 0 C4s.
    
    # Alternative: Use a random greedy approach to build a C4-free graph with as many edges as possible,
    # then add the 91st edge.
    
    # Actually, let me just try the polarity graph and see how many edges it has.
    # If it has >= 91 edges, we can take a 91-edge subset (C4-free subgraph of C4-free graph is C4-free).
    # If it has < 91 edges, we need to add edges.
    
    num_edges_pg = len(edges_pg)
    
    if num_edges_pg >= 91:
        # Take first 91 edges - still C4-free
        return edges_pg[:91]
    
    # Otherwise, we need to add edges to reach 91
    # Start with the polarity graph edges
    current_edges = list(edges_pg)
    adj = [set() for _ in range(n)]
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Try to add edges greedily, preferring edges that create 0 C4s
    remaining_pairs = []
    edge_set = set()
    for u, v in current_edges:
        edge_set.add((min(u,v), max(u,v)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                remaining_pairs.append((i, j))
    
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(remaining_pairs)
    
    # For each candidate edge, compute how many C4s it would create
    # Adding edge (u,v) creates a C4 for each pair (a,b) where a is neighbor of u, b is neighbor of v,
    # a != v, b != u, a != b, and a-b is not an edge... wait no.
    # Actually, adding edge (u,v) creates a C4 if there exist a, b such that:
    # a is adjacent to u, b is adjacent to v, a != v, b != u, a != b,
    # and a is adjacent to b. Then u-a-b-v-u is a C4.
    # Wait, that's not quite right either. A C4 containing the new edge (u,v) would be
    # u - a - b - v - u where a is a common... no.
    # The C4s created by adding edge (u,v) are: for each pair (a,b) where a~u, b~v,
    # a != v, b != u, a != b, and a~b. Then u,a,b,v form a C4: u-a-b-v-u.
    
    def c4s_created(u, v):
        count = 0
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[v]:
                if b == u or b == a:
                    continue
                if a in adj[b]:
                    count += 1
        return count
    
    # Sort remaining pairs by number of C4s they create (ascending)
    scored_pairs = []
    for (i, j) in remaining_pairs:
        c = c4s_created(i, j)
        scored_pairs.append((c, i, j))
    
    scored_pairs.sort(key=lambda x: x[0])
    
    for c, i, j in scored_pairs:
        if len(current_edges) >= 91:
            break
        if time.time() > deadline - 10:
            break
        current_edges.append([i, j])
        adj[i].add(j)
        adj[j].add(i)
        edge_set.add((i, j))
    
    # If we still don't have 91 edges, just add any remaining pairs
    if len(current_edges) < 91:
        for (i, j) in remaining_pairs:
            if len(current_edges) >= 91:
                break
            if (i, j) not in edge_set:
                current_edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                edge_set.add((i, j))
    
    return current_edges[:91]

# EVOLVE-BLOCK-END
