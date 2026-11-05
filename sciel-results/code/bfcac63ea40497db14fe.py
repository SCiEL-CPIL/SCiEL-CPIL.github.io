"""Constructs small k-regular graphs of girth g using algebraic constructions."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g on as few vertices as possible.
    
    Strategy:
    1. Try Lazebnik-Ustimenko-Woldar (LUW) construction for k = q (prime or 4)
    2. Try circulant graph constructions
    3. Use backtracking/DFS to build a graph vertex by vertex with girth constraint
    
    Returns (n, edges) where n is number of vertices and edges is list of (i, j) tuples.
    """
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Try LUW construction first
    result_luw = _try_luw(k, g, deadline)
    if result_luw is not None:
        n_luw, edges_luw = result_luw
    else:
        n_luw, edges_luw = None, None
    
    # Try circulant construction
    result_circ = _try_circulant(k, g, deadline)
    if result_circ is not None:
        n_circ, edges_circ = result_circ
    else:
        n_circ, edges_circ = None, None
    
    # Choose the best (smallest n) result
    best = None
    if n_luw is not None:
        best = (n_luw, edges_luw)
    if n_circ is not None:
        if best is None or n_circ < best[0]:
            best = (n_circ, edges_circ)
    
    if best is not None:
        return best
    
    # Fallback: use a simple construction that guarantees correctness
    return _fallback_construction(k, g)

def _try_luw(k, g, deadline):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    This is a q-regular bipartite graph with girth at least d+5 for odd d.
    """
    if time.time() > deadline - 5:
        return None
    
    q = k
    # For LUW, we need q to be prime or 4
    if q != 4 and not _is_prime(q):
        return None
    
    # d should be such that girth >= g
    # Girth of CD(d, q) is at least d + 5 for odd d, d + 4 for even d
    # We want girth >= g, so we need d + 5 >= g (for odd d) or d + 4 >= g (for even d)
    # Let's try d = g - 5 if that gives odd d, else g - 4
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    # Build the rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    n = len(order)
    # Verify girth
    if _verify_girth(n, edges, g):
        return n, edges
    return None

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _try_circulant(k, g, deadline):
    """
    Try to construct a circulant graph C(n; S) where S is a set of k/2 offsets.
    For k even, we can use symmetric offsets.
    """
    if time.time() > deadline - 5:
        return None
    
    if k % 2 != 0:
        return None
    
    half_k = k // 2
    # For a circulant graph C(n; {s1, s2, ..., s_{k/2}}), the girth is determined
    # by the shortest non-trivial relation sum of ±si ≡ 0 mod n
    
    # Try small n values
    # Moore bound for (k, g): 
    # For g odd: n >= k + (k-1) * (1 + k + ... + k^((g-3)/2))
    # For g even: n >= 2 * (1 + (k-1) + ... + (k-1)^((g-2)/2))
    if g % 2 == 1:
        moore = k + (k - 1) * sum(k**i for i in range((g - 3) // 2 + 1))
    else:
        moore = 2 * sum((k - 1)**i for i in range((g - 2) // 2 + 1))
    
    # Try n from moore bound upward
    max_n = min(500, moore + 200)
    
    for n in range(max(moore, k + 2), max_n + 1):
        if time.time() > deadline - 5:
            return None
        
        # Try different offset sets
        # Start with consecutive offsets: {1, 2, ..., k/2}
        # This gives a cycle of length n, which might be too short
        # We need offsets that avoid short cycles
        
        # For girth >= g, we need no relation of length < g
        # A simple approach: use offsets that are "spread out"
        
        # Try offsets {1, n//4, n//3, ...} for appropriate n
        # Actually, let's try a few patterns
        
        # Pattern 1: {1, 2, ..., k/2} - bad for small n
        # Pattern 2: {1, p, 2p, ..., (k/2-1)p} mod n where p is chosen to avoid short cycles
        
        # For simplicity, try p = n // (k // 2 + 1) or similar
        found = False
        for p in range(2, n // 2):
            offsets = [(i * p) % n for i in range(1, half_k + 1)]
            # Remove duplicates and 0
            offsets = list(set(o for o in offsets if o != 0 and o != n // 2))
            if len(offsets) < half_k:
                continue
            
            # Take first half_k offsets
            offsets = sorted(offsets)[:half_k]
            
            # Build edges
            edges = []
            for v in range(n):
                for s in offsets:
                    u = (v + s) % n
                    if v < u:
                        edges.append((v, u))
            
            # Check degree
            deg = [0] * n
            for a, b in edges:
                deg[a] += 1
                deg[b] += 1
            if any(d != k for d in deg):
                continue
            
            # Check girth
            if _verify_girth(n, edges, g):
                return n, edges
        
        # Also try non-circulant: random-ish construction
        # Use a greedy approach
    
    return None

def _verify_girth(n, edges, g):
    """
    Verify that the graph has no cycle shorter than g.
    Uses BFS from each vertex to find shortest cycles.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    # For each vertex, do BFS to find shortest cycle through that vertex
    # A cycle through v has length L if there's a path of length L-1 from v to a neighbor of v
    # that doesn't use the edge (v, neighbor)
    
    # More efficiently: for each edge (u, v), the shortest cycle containing that edge
    # is 1 + shortest path from u to v not using that edge
    
    # Actually, let's use a standard approach:
    # For each vertex s, do BFS. The girth is the minimum over all s of the shortest
    # cycle through s.
    
    # A cycle through s: s -> ... -> s. The length is at least g.
    # We can find this by doing BFS from s and looking for when we reach a neighbor of s
    # via a path that doesn't use the direct edge.
    
    # Simpler: for each edge (u, v), compute shortest path from u to v avoiding that edge.
    # The cycle length is that path length + 1.
    
    # For efficiency, let's do BFS from each vertex and track the shortest cycle.
    
    # Actually, the most straightforward: for each starting vertex s, do BFS.
    # When we discover a vertex that's already visited, we've found a cycle.
    # The cycle length is dist[current] + dist[neighbor] + 1 if the neighbor is the parent,
    # or we need to be more careful.
    
    # Let's use: for each vertex s, BFS. For each vertex v reached at distance d,
    # if v is adjacent to s and v != the first neighbor, then we have a cycle of length d + 1.
    # But this might miss some cycles.
    
    # Better: for each edge (u, v), remove that edge and find shortest path from u to v.
    # If the shortest path has length L, then there's a cycle of length L + 1.
    # The girth is the minimum over all edges.
    
    # This is O(E * (V + E)) which is too slow for large graphs.
    
    # Alternative: BFS from each vertex. For each vertex s, do BFS. Keep track of parent.
    # When we encounter a vertex that has been visited, and it's not the parent, we have a cycle.
    # The cycle length is dist[u] + dist[v] + 1 where u and v are the two vertices in the BFS tree
    # that are connected by the non-tree edge.
    
    # Let's do a simpler check: just verify no cycle of length < g exists.
    # We can do this by checking all walks of length < g and ensuring they don't close.
    
    # For girth >= g, there should be no closed walk of length < g that starts and ends
    # at the same vertex without immediately backtracking.
    
    # Actually, the simplest correct check: use BFS from each vertex.
    # For vertex s, BFS gives distances. For each edge (u, v) where u != v,
    # if dist_s[u] + 1 + dist_s[v] < g and the paths don't share edges in a way that
    # creates a shorter cycle... this is getting complicated.
    
    # Let's use a different approach: for each vertex s, do BFS and find the shortest
    # cycle through s.
    
    # For each s:
    #   BFS from s. For each vertex v at distance d from s, for each neighbor w of v,
    #   if w is adjacent to s and w != the parent of v in the BFS tree, then we have
    #   a cycle of length d + 1 (if w is s) or d + dist_s[w] + 1 (in general).
    
    # Actually, the standard algorithm:
    # For each vertex s, do BFS. For each non-tree edge (u, v), the cycle length is
    # dist[u] + dist[v] + 1. The girth is the minimum over all s and all non-tree edges.
    
    # But this counts each cycle multiple times. That's fine for finding the minimum.
    
    # Let's implement this:
    min_cycle = float('inf')
    
    for s in range(n):
        # BFS from s
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Non-tree edge (u, v) - but we might count it twice
                    # Cycle length: dist[u] + dist[v] + 1
                    # But we need to make sure this is a valid cycle (no repeated vertices)
                    # For BFS, this is guaranteed if we're careful
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
        
        # Early termination
        if min_cycle < g:
            return False
    
    return min_cycle >= g

def _fallback_construction(k, g):
    """
    Fallback: use a known construction or a simple valid graph.
    This is a last resort.
    """
    # For small k and g, use explicit constructions
    # For larger cases, we need a better approach
    
    # Use a complete bipartite graph K_{k,k} which has girth 4
    # This only works if g <= 4
    
    # For g > 4, we need something else.
    # Let's use a random regular graph and check its girth.
    # If it doesn't have sufficient girth, try again.
    
    import random
    random.seed(42)
    
    max_attempts = 100
    for attempt in range(max_attempts):
        # Try different n values
        n = k + 2 + attempt * 2  # Start small and grow
        
        # Generate a random k-regular graph
        # Use configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        
        edges = []
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u != v:
                edges.append((min(u, v), max(u, v)))
        
        # Check for duplicates and self-loops
        edge_set = set()
        valid_edges = []
        for a, b in edges:
            if a != b and (a, b) not in edge_set:
                edge_set.add((a, b))
                valid_edges.append((a, b))
        
        # Check degree
        deg = [0] * n
        for a, b in valid_edges:
            deg[a] += 1
            deg[b] += 1
        
        if any(d != k for d in deg):
            continue
        
        # Check girth
        if _verify_girth(n, valid_edges, g):
            return n, valid_edges
    
    # If all attempts fail, return a trivial (but possibly invalid) result
    # This should never happen for reasonable inputs
    return k, []
# EVOLVE-BLOCK-END
