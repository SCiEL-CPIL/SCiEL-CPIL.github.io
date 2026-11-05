import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i,j) pairs.
    """
    # Strategy 1: Cayley Graph Search on Abelian Groups
    # This is a new approach compared to LUW/Greedy.
    
    # Moore Bound for odd girth g=2r+1: n >= k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = k * sum((k - 1) ** i for i in range(r))
    
    # We want to minimize n. Start from moore_bound.
    # Since k=5 is odd, for cyclic groups Z_n, we need n to be even to have an involution c.
    # If n is odd, we can't form a symmetric generating set of odd size in Z_n using pairs +/- and one involution.
    # So we start from the first even number >= moore_bound.
    
    start_n = moore_bound
    if start_n % 2 != 0:
        start_n += 1
        
    # Time limit management
    start_time = time.time()
    time_limit = 110  # Leave 10s for verification and overhead
    
    for n in range(start_n, 1200):  # Hard cap to prevent infinite loop, can be adjusted
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a Cayley graph on Z_n with girth >= g
        # Generators: {a, -a, b, -b, c} where c = n/2 (involution)
        # Conditions: a, b, c distinct modulo n. a != 0, b != 0.
        # a != b, a != -b, etc.
        
        # Random attempts
        found = False
        attempts = 50
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Choose random generators
            # Ensure a, b are not 0, n/2, and distinct from each other and their negatives
            valid_gen = False
            while not valid_gen:
                a = random.randint(1, n - 1)
                b = random.randint(1, n - 1)
                c = n // 2
                
                if a == b or a == -a % n or b == -b % n:
                    continue
                if a == b % n or a == (-b) % n:
                    continue
                if a == c or b == c:
                    continue
                valid_gen = True
                
            # Construct edges
            edges = []
            # For each vertex v, connect to v+a, v-a, v+b, v-b, v+c
            # To avoid duplicates, we only add edge (u, v) if u < v
            # Or just generate all and deduplicate later? 
            # Better: iterate v, add edges to neighbors, keep set.
            
            edge_set = set()
            for v in range(n):
                neighbors = [(v + a) % n, (v - a) % n, (v + b) % n, (v - b) % n, (v + c) % n]
                for u in neighbors:
                    if u == v:
                        return None, None # Should not happen if a,b,c != 0
                    if u < v:
                        edge_set.add((u, v))
                    elif u > v:
                        edge_set.add((v, u))
            
            # Check for simplicity (no loops/multiedges) - already handled by set and u!=v check
            # Check girth
            if _check_girth(n, list(edge_set), k, g):
                return n, list(edge_set)
        
        # If not found for this n, try next n
    
    # Fallback: Use LUW if Cayley search fails
    # We reuse the logic from the original code but wrapped in a try/except
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        if n_luw is not None:
            return n_luw, edges_luw
    except Exception:
        pass
        
    # Final Fallback: Random Greedy (from original)
    try:
        n_rand, edges_rand = _random_greedy(k, g, time_limit=10)
        if n_rand is not None:
            return n_rand, edges_rand
    except Exception:
        pass
        
    # If all else fails, return a trivial invalid or best effort?
    # The problem requires valid output.
    # We must ensure we return something valid.
    # If time runs out, we might return the last best candidate?
    # For safety, let's try to return a LUW graph which is guaranteed valid (if it terminates).
    
    return _luw_construct(k, g)


def _check_girth(n, edges, k, g):
    """
    Verify girth >= g and regularity.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # BFS for girth
    half = (g - 1) // 2
    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if visited[u] >= half:
                continue
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    queue.append(v)
                elif v != u and visited[v] <= visited[u]:
                    # Cycle detected
                    # Length is visited[u] + visited[v] + 1
                    # If this is < g, fail
                    if visited[u] + visited[v] + 1 < g:
                        return False
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
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
    return len(order), edges


def _random_greedy(k, g, time_limit=60):
    """
    Fallback random greedy.
    """
    start_time = time.time()
    r = (g - 1) // 2
    moore_bound = k * sum((k - 1) ** i for i in range(r))
    
    for start_n in range(moore_bound, moore_bound + 20):
        if time.time() - start_time > time_limit:
            break
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            edges, success = _try_greedy_build(k, g, start_n, seed=attempt)
            if success and edges is not None:
                if _check_girth(k, g, start_n, edges):
                    return start_n, edges
    return None, None


def _try_greedy_build(k, g, n, seed=0):
    rng = random.Random(seed)
    degree = [0] * n
    edges = []
    edge_set = set()
    
    stubs = [v for v in range(n) for _ in range(k)]
    rng.shuffle(stubs)
    
    candidate_edges = []
    valid = True
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            valid = False
            break
        if (min(u, v), max(u, v)) in edge_set:
            valid = False
            break
        edge_set.add((min(u, v), max(u, v)))
        candidate_edges.append((u, v))
    
    if not valid:
        return None, False
    return candidate_edges, True
