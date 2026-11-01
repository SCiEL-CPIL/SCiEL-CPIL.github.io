"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible.
    
    Uses a combination of known small cage constructions, a randomized search for 
    generic cases, and a LUW fallback.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # 1. Known small cages
    if (k, g) in known_cages:
        return known_cages[(k, g)]
    
    # 2. Calculate bounds
    if g % 2 == 0:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
    else:
        half = (g - 1) // 2
        moore = k * sum((k - 1) ** i for i in range(half))
    
    # 3. Try Randomized Search for n in [moore, upper_bound]
    # The LUW construction gives an upper bound.
    # We estimate LUW size to set the search limit.
    n_luw_est = _estimate_luw_size(k, g)
    
    # We search up to min(n_luw_est, moore * 5, 2000)
    # For (5,9), moore=426, n_luw_est ~ 6000. We search up to 2000.
    search_limit = min(n_luw_est, int(moore * 4) + 100, 2000)
    
    # If moore is very small, search a bit higher
    if search_limit < moore + 100:
        search_limit = moore + 100
        
    # Ensure we don't search below Moore bound
    n_start = max(moore, 2 * k)
    
    # Randomized search
    for n in range(n_start, search_limit + 1):
        if time.time() > deadline:
            break
            
        # Try to find a valid random k-regular graph on n vertices
        # Number of attempts depends on time remaining
        attempts = 0
        max_attempts = 50 # Adjust based on n
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            graph = _generate_random_k_regular(n, k)
            if graph is None:
                continue
            
            # Check girth
            if _verify_girth(n, graph, k, g):
                return n, graph
                
        # If we found a valid graph, we return it immediately above.
        # If not, we try the next n.
        
    # 4. Fallback: LUW Construction
    # LUW is guaranteed to work for prime k (or k=4)
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        # Verify just in case, though LUW is theoretically sound
        if _verify_girth(n, edges, k, g):
            return n, edges
            
    # 5. Last Resort: Generic Random (might fail girth, but returns something)
    # This should rarely be reached if LUW works
    n_final = max(n_start, 2 * k)
    edges = _generate_random_k_regular(n_final, k)
    if edges:
        return n_final, edges
        
    # Absolute fallback: Complete graph (invalid girth, but valid structure)
    # This will score 0, but prevents crash
    n = k + 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return n, edges


def _estimate_luw_size(k, g):
    """Estimate the number of vertices in the LUW construction for given k, g."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # LUW(d, q) has 2 * q^d vertices
    return 2 * (q ** d)


def _generate_random_k_regular(n, k):
    """Generate a random k-regular simple graph on n vertices using configuration model."""
    if n * k < 2 or n * k % 2 != 0:
        return None
    if k >= n:
        return None
        
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None # Loop
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            return None # Multi-edge
        edges.add((u, v))
        
    return list(edges)


def _verify_girth(n, edges, k, g):
    """Verify that the graph has girth at least g and is k-regular."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # BFS from each vertex to find shortest cycle
    # We only need to check cycles up to length g-1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2: 
                # If we've gone deep enough, we might have found a cycle, 
                # but we specifically look for back-edges creating short cycles
                pass
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar type construction."""
    q = k
    if q < 2:
        return None
    # For LUW, we need q to be prime or 4
    # The girth of LUW(d, q) is at least d + 5 for odd d
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if d < 3:
        return None
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Check if q is prime
        if not _is_prime(q):
            return None
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
    # Check if the graph is k-regular
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return None
    
    return n, edges


def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


# Known cages dictionary
known_cages = {}

# Petersen graph: 3-regular, girth 5, 10 vertices
def _petersen():
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges
known_cages[(3, 5)] = _petersen()

# Heawood graph: 3-regular, girth 6, 14 vertices
def _heawood():
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6),
        (2, 4, 5)
    ]
    edges = []
    for i, line in enumerate(lines):
        for p in line:
            edges.append((i, 7 + p))
    return 14, edges
known_cages[(3, 6)] = _heawood()

# (3,7)-cage: 24 vertices
def _cage_3_7():
    # Foster graph is (3,9). (3,7) cage is 24 vertices.
    # We can use a known edge list or generate via random search if not hardcoded.
    # For now, let random search handle it, but we can hardcode a valid one if known.
    # The (3,7)-cage is the unique 3-regular graph on 24 vertices with girth 7.
    # Hardcoding is error-prone. Let's rely on random search for this if it fails.
    # But random search for n=24 is very fast.
    return None
# We don't put (3,7) in known_cages to let the random search find it quickly, 
# unless we are sure. Actually, random search for small n is very reliable.

# (4,5)-cage: 19 vertices
# (4,6)-cage: 26 vertices
# (5,5)-cage: 30 vertices (Complement of Petersen? No. 5-regular on 30 vertices? 
# The (5,5)-cage is the complement of the 4-regular graph? 
# Actually, the (5,5)-cage is the 5-regular graph on 30 vertices? 
# Moore bound for (5,5): 1 + 5 + 20 = 26. 
# The unique (5,5)-cage has 30 vertices? 
# Let's let random search handle these small cases.

# EVOLVE-BLOCK-END
