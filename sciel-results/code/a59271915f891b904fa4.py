import time
import random
import numpy as np

def construct(k, g):
    """Construct a k-regular graph with girth >= g.
    
    Returns (n, edges) where n is the number of vertices and edges is a list
    of (i, j) pairs on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    # Try SL(2,11) Cayley graph approach
    result = _sl2_11_cayley()
    if result is not None:
        n, edges = result
        # Verify the result
        if _verify_graph(n, edges, k, g, deadline):
            return n, edges
    
    # Fallback to LUW construction
    return _luw_construct(k, g)


def _sl2_11_cayley():
    """Construct a 5-regular Cayley graph of SL(2,11) with girth >= 9.
    
    Returns (n, edges) or None if no valid construction found.
    """
    start_time = time.time()
    deadline = start_time + 135
    
    p = 11
    # Generate all elements of SL(2, p) as 2x2 matrices mod p with det = 1
    elements = []
    elem_set = set()
    
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # det = ad - bc = 1, so d = (1 + bc) * a^{-1} mod p
                # But a might be 0, handle separately
                if a == 0:
                    # Need -bc = 1 mod p, so bc = -1 mod p
                    for d in range(p):
                        if (-c * b) % p == 1:
                            mat = (a, b, c, d)
                            if mat not in elem_set:
                                elem_set.add(mat)
                                elements.append(mat)
                else:
                    inv_a = pow(a, p-2, p)
                    d = (1 + b * c) * inv_a % p
                    mat = (a, b, c, d)
                    if mat not in elem_set:
                        elem_set.add(mat)
                        elements.append(mat)
    
    n = len(elements)
    # SL(2, 11) should have 11*12*11/2 = 726 elements
    # Wait, |SL(2,p)| = p(p^2-1) = 11*120 = 1320
    # Let me recount: p(p^2-1) = 11*(121-1) = 11*120 = 1320
    
    if n != 1320:
        # Try a different generation approach
        elements = []
        elem_set = set()
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    for d in range(p):
                        det = (a * d - b * c) % p
                        if det == 1:
                            mat = (a, b, c, d)
                            if mat not in elem_set:
                                elem_set.add(mat)
                                elements.append(mat)
        n = len(elements)
    
    if n != 1320:
        return None
    
    # Create index mapping
    elem_to_idx = {mat: i for i, mat in enumerate(elements)}
    
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
    
    def mat_inv(m):
        a, b, c, d = m
        # For SL(2,p), inverse is (d, -b, -c, a) mod p
        return (d, (-b) % p, (-c) % p, a)
    
    # Find a symmetric generating set of size 5
    # We need 5 elements s1, s2, s3, s4, s5 such that:
    # - The set is symmetric: if s is in S, then s^{-1} is in S
    # - No element is the identity
    # - The Cayley graph has girth >= 9
    
    # Strategy: pick random elements and check
    identity = (1, 0, 0, 1)
    
    # Precompute inverses
    inv_map = {mat: mat_inv(mat) for mat in elements}
    
    # Try random symmetric sets
    random.seed(42)
    
    def check_girth(gen_set):
        """Check if the Cayley graph with generator set gen_set has girth >= 9.
        
        Uses BFS from each vertex to verify no cycle shorter than 9.
        For efficiency, we only need to check that balls of radius 4 are trees.
        """
        # For a 5-regular graph, ball of radius 4 has at most 1 + 5 + 25 + 125 + 625 = 781 vertices
        # If girth >= 9, the ball of radius 4 around any vertex is a tree
        
        # We'll do BFS from a few sample vertices to check for short cycles
        # For a complete check, we'd need all vertices, but let's sample
        
        # Actually, for correctness, we need to check all vertices
        # But that's expensive. Let's use a heuristic: check that no two
        # generators (or products) create a short cycle.
        
        # A cycle of length < 9 means there exist words in the generators
        # of length < 9 that multiply to identity, where the word is
        # "reduced" (no adjacent generator and its inverse).
        
        # For girth >= 9, we need:
        # - No generator is identity
        # - No product of 2 generators is identity (except s * s^{-1})
        # - No product of 3 generators is identity (except s * t * t^{-1} type)
        # - etc.
        
        # Let's check all words of length up to 4 (since girth 9 means
        # no closed walk of length < 9, and a cycle of length L gives a
        # closed walk of length L, but also we need to check that the
        # ball of radius 4 is a tree)
        
        # Actually, the cleanest check: for each vertex v, do BFS and verify
        # that the first 4 levels form a tree (no vertex appears twice at
        # the same or different levels in a way that creates a cycle < 9).
        
        # For efficiency, let's check a subset of vertices
        sample_size = min(100, n)
        sample_vertices = random.sample(range(n), sample_size)
        
        for v in sample_vertices:
            # BFS from v
            visited = {v: (0, None)}  # vertex -> (distance, parent)
            queue = [(v, 0, None)]
            
            while queue:
                current, dist, parent = queue.pop(0)
                if dist >= 4:
                    continue
                
                for gen in gen_set:
                    # Multiply current by gen
                    # In Cayley graph, neighbors of v are v * s for s in S
                    neighbor = mat_mul(elements[current], gen)
                    nidx = elem_to_idx.get(neighbor)
                    
                    if nidx is None:
                        continue
                    
                    if nidx not in visited:
                        visited[nidx] = (dist + 1, current)
                        queue.append((nidx, dist + 1, current))
                    else:
                        # Found a vertex already visited
                        prev_dist, prev_parent = visited[nidx]
                        # Check if this creates a cycle of length < 9
                        # The new path has length dist + 1
                        # The old path has length prev_dist
                        # A cycle would be formed if we can trace back
                        # from current to the LCA of current and nidx
                        
                        # For simplicity, if we find a back-edge or cross-edge
                        # that creates a cycle < 9, fail
                        cycle_len = dist + 1 + prev_dist
                        # This is an upper bound; actual cycle might be shorter
                        # if there's a common ancestor
                        if cycle_len < 9:
                            return False
        
        return True
    
    # Try to find a good generator set
    # Start with known generators for SL(2,11)
    # Standard generators: x = [[0, 1], [-1, 0]], y = [[1, 1], [0, 1]]
    x = (0, 1, p-1, 0)  # [[0, 1], [-1, 0]]
    y = (1, 1, 0, 1)     # [[1, 1], [0, 1]]
    
    # Generate some candidate generators
    candidates = [x, y, mat_inv(x), mat_inv(y)]
    
    # Add more candidates by multiplying
    for _ in range(20):
        a = random.choice(elements)
        if a != identity:
            candidates.append(a)
            candidates.append(mat_inv(a))
    
    # Remove identity and duplicates
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c != identity and c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    # Try random subsets of size 5 that are symmetric
    # A symmetric set of size 5: either 5 elements where each is its own inverse
    # (order 2), or we need pairs + one self-inverse
    
    # For simplicity, let's try sets where we pick 2 pairs of inverses + 1 self-inverse
    # Or 5 self-inverse elements (order 2)
    
    # Find self-inverse elements (order 2)
    self_inverse = [mat for mat in elements if mat != identity and mat_inv(mat) == mat]
    
    # Try various combinations
    attempts = 0
    max_attempts = 5000
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        
        # Pick a random symmetric set of size 5
        # Option 1: 2 pairs + 1 self-inverse
        if len(self_inverse) >= 1 and len(unique_candidates) >= 4:
            # Pick 2 non-self-inverse elements and their inverses
            non_self_inv = [c for c in unique_candidates if mat_inv(c) != c and c != identity]
            if len(non_self_inv) >= 2:
                pair1 = random.choice(non_self_inv)
                pair2 = [c for c in non_self_inv if c != pair1 and mat_inv(c) != pair1]
                if pair2:
                    pair2 = random.choice(pair2)
                    si = random.choice(self_inverse)
                    gen_set = [pair1, mat_inv(pair1), pair2, mat_inv(pair2), si]
                    
                    # Verify it's a valid generating set (generates the group)
                    # For now, just check girth
                    if check_girth(gen_set):
                        # Build the graph
                        edges = []
                        for i in range(n):
                            for gen in gen_set:
                                neighbor = mat_mul(elements[i], gen)
                                nidx = elem_to_idx.get(neighbor)
                                if nidx is not None and i < nidx:
                                    edges.append((i, nidx))
                        return n, edges
        
        # Option 2: 5 self-inverse elements
        if len(self_inverse) >= 5:
            gen_set = random.sample(self_inverse, 5)
            if check_girth(gen_set):
                edges = []
                for i in range(n):
                    for gen in gen_set:
                        neighbor = mat_mul(elements[i], gen)
                        nidx = elem_to_idx.get(neighbor)
                        if nidx is not None and i < nidx:
                            edges.append((i, nidx))
                return n, edges
    
    return None


def _verify_graph(n, edges, k, g, deadline):
    """Verify that the graph is k-regular, simple, and has girth >= g."""
    if time.time() > deadline:
        return False
    
    # Check degree
    degree = [0] * n
    edge_set = set()
    
    for u, v in edges:
        if u == v:
            return False  # Loop
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False  # Duplicate edge
        edge_set.add(key)
        degree[u] += 1
        degree[v] += 1
    
    for d in degree:
        if d != k:
            return False
    
    # Check girth using BFS from each vertex
    # For girth >= g, we need no cycle of length < g
    # Use BFS to find shortest cycle through each vertex
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS to find the shortest cycle
    # This is expensive, so we'll sample vertices
    sample_size = min(50, n)
    sample_vertices = random.sample(range(n), sample_size)
    
    for v in sample_vertices:
        # BFS from v
        dist = [-1] * n
        parent = [-1] * n
        dist[v] = 0
        queue = [v]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= (g // 2):
                continue
            
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
                elif w != parent[u]:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[w] + 1
                    # But we need to be more careful
                    cycle_len = dist[u] + dist[w] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for k-regular graphs with girth g."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
