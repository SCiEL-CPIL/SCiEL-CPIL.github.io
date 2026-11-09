"""Construct 3-regular graph of girth 16 via random regular graph search + NB-BFS girth check."""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Moore bound for (3,16): 2 * (2^7 - 1) = 254, but actual lower bound is 512
    # Start searching from a reasonable size
    if k != 3:
        # Fallback for other k: use LUW-like construction
        return _luw_construct(k, g)
    
    # For (3,16), search for small 3-regular graphs with girth >= 16
    # The known best is 936, but we'll try to find something smaller or equal
    # Start from Moore bound and increase
    
    moore = 2 * ((1 << (g // 2 - 1)) - 1)  # 2*(2^7-1)=254 for g=16
    # But girth 16 means no cycles < 16, so shortest cycle >= 16
    # Moore bound: n >= 2*(2^(g/2 - 1) - 1) = 2*(2^7-1) = 254 for even girth
    
    # Try sizes starting from a minimum that's feasible
    # For girth 16, we need enough vertices
    # Let's try: 256, 384, 512, 640, 768, 936, ...
    
    sizes_to_try = []
    # Start from Moore bound rounded up to multiple of 2 (for 3-reg, n must be even)
    min_n = max(moore, 2 * k)  # at least 2k vertices
    # Make even
    if min_n % 2 != 0:
        min_n += 1
    
    # Try a range of sizes
    for n in range(min_n, 2000, 2):
        sizes_to_try.append(n)
    
    random.seed(42)
    
    for n in sizes_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple random graphs of this size
        max_attempts = 20 if n < 500 else 10 if n < 1000 else 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            edges = _random_3_regular(n)
            if edges is None:
                continue
            
            # Check girth >= g using non-backtracking BFS
            if _has_girth_at_least(edges, n, g):
                return n, edges
    
    # Fallback: use LUW construction if random search fails
    return _luw_construct(k, g)

def _random_3_regular(n):
    """Generate a random 3-regular graph on n vertices using configuration model + dedup."""
    if n < 6 or n % 2 != 0:
        return None
    
    # Configuration model: each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    # Pair up stubs
    edges = []
    seen = set()
    for i in range(0, n * 3, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None  # loop
        key = (min(u, v), max(u, v))
        if key in seen:
            return None  # multi-edge
        seen.add(key)
        edges.append(key)
    
    # Verify degree
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if any(d != 3 for d in degree):
        return None
    
    return edges

def _has_girth_at_least(edges, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking BFS from each vertex
    # A non-backtracking walk of length L from vertex s means:
    # the sequence of vertices v0=s, v1, ..., vL where vi != vi-2 for all i>=2
    # If we find vL = v0 with L < g, then there's a cycle of length < g
    
    for s in range(n):
        # BFS state: (current_vertex, previous_vertex, length)
        # We track the shortest non-backtracking walk back to s
        # For each vertex, we track the minimum length of a non-backtracking walk from s that ends there
        # But we need to avoid backtracking, so we track (prev, curr) pairs
        
        # dist[prev][curr] = shortest non-backtracking walk length from s to curr with prev as the previous vertex
        # This is O(n^2) space per start vertex, which is too much for large n
        # Alternative: just do BFS on the directed line graph (states are directed edges)
        
        # Directed edge (u, v) means we arrived at v from u
        # From (u, v), we can go to (v, w) for w in adj[v] if w != u
        # We want to find the shortest path from any (s, v) to any (v, s) where the path length + 1 < g
        
        # Actually, a cycle of length L corresponds to a non-backtracking walk of length L
        # starting and ending at the same vertex.
        
        # Let's do BFS on directed edges:
        # State: (prev, curr)
        # Start: all (s, v) for v in adj[s], length 1
        # Goal: find (v, s) for any v in adj[s], with total length < g
        
        # For efficiency, we only need to check lengths up to g-1
        max_len = g - 1  # if we find a cycle of length < g, girth < g
        
        # BFS
        visited = set()
        queue = deque()
        
        for v in adj[s]:
            state = (s, v)
            queue.append((state, 1))
            visited.add(state)
        
        while queue:
            (prev, curr), length = queue.popleft()
            
            if length >= max_len:
                continue
            
            for w in adj[curr]:
                if w == prev:
                    continue  # no backtracking
                
                new_state = (curr, w)
                
                if new_state in visited:
                    continue
                
                visited.add(new_state)
                
                if w == s:
                    # Found a cycle of length length + 1
                    if length + 1 < g:
                        return False
                
                queue.append((new_state, length + 1))
        
        # Also need to check: can we return to s via a non-backtracking walk of length < g?
        # The BFS above checks for cycles that go through s.
        # But what about cycles not through s? Those will be found when we start BFS from a vertex on that cycle.
    
    return True

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
    start_time = time.time()
    deadline = start_time + 140
    
    q = k
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(d, 3)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q
        
    def get_digit(val, pos):
        return (val // powers[pos-1]) % q

    def get_point_neighbours(p_id):
        for l1 in range(q):
            current_l_id = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                p_c = get_digit(p_id, c)
                l_a = get_digit(current_l_id, a)
                p_b = get_digit(p_id, b)
                l_c = add(p_c, mul(l_a, p_b))
                current_l_id += l_c * powers[c-1]
            yield (1, current_l_id)

    def get_line_neighbours(l_id):
        for p1 in range(q):
            current_p_id = p1
            for c in range(2, d + 1):
                a, b = rule[c]
                l_c = get_digit(l_id, c)
                l_a = get_digit(l_id, a)
                p_b = get_digit(current_p_id, b)
                p_c = sub(l_c, mul(l_a, p_b))
                current_p_id += p_c * powers[c-1]
            yield (0, current_p_id)

    start_vertex = (0, 0)
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, vid = vertex
        if side == 0:
            neighbors = get_point_neighbours(vid)
        else:
            neighbors = get_line_neighbours(vid)
            
        for n_side, n_vid in neighbors:
            n_vertex = (n_side, n_vid)
            if n_vertex not in index:
                index[n_vertex] = len(order)
                order.append(n_vertex)
            if side == 0:
                edges.append((index[vertex], index[n_vertex]))
                
    n = len(order)
    return n, edges
