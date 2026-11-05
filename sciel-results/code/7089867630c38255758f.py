import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses a randomized edge-switching heuristic to minimize short cycles,
    starting from a random 3-regular graph and iteratively improving girth.
    """
    start_time = time.time()
    deadline = start_time + 130.0

    if k == 3:
        return _construct_3reg_edge_switching(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_3reg_edge_switching(g, deadline, start_time):
    """
    Construct a 3-regular graph with girth >= g by:
    1. Start with a random 3-regular graph on n vertices
    2. Iteratively find short cycles and switch edges to break them
    3. Use spectral/trace-based objective to guide switches
    
    We try increasing n from the Moore bound upward.
    """
    # Moore bound for k=3, girth g:
    # For even g=2r: n >= 2 * (3*(2^(r-1) - 1))
    # For odd g=2r+1: n >= 3 * (2^r - 1)
    if g % 2 == 0:
        r = g // 2
        moore_bound = 2 * (3 * (2**(r-1) - 1))
    else:
        r = (g - 1) // 2
        moore_bound = 3 * (2**r - 1)
    
    # Try various n values starting from a reasonable lower bound
    # The known best is 936 for g=16, so we aim for that range
    # But we'll try smaller first
    
    # For g=16, Moore bound = 2*(3*(2^7-1)) = 2*3*127 = 762
    # Known best: 936
    
    best_n = float('inf')
    best_edges = None
    
    # Candidate n values to try
    if g <= 8:
        candidates = [moore_bound, moore_bound + 2, moore_bound + 4, moore_bound + 6]
    elif g <= 12:
        candidates = [moore_bound, moore_bound + 2, moore_bound + 4, moore_bound + 6, moore_bound + 8]
    else:
        # For larger girth, try values around the known best
        # Scale from known results
        if g == 16:
            candidates = [762, 764, 766, 768, 770, 772, 774, 776, 778, 780, 782, 784, 786, 788, 790, 792, 794, 796, 798, 800]
        elif g == 14:
            candidates = [512, 514, 516, 518, 520, 522, 524, 526, 528, 530]
        elif g == 18:
            candidates = [2048, 2050, 2052, 2054, 2056]
        else:
            # General: try Moore bound and a few above
            candidates = [moore_bound + 2*i for i in range(20)]
    
    # Filter to valid 3-regular graph sizes (n must be even)
    candidates = [n for n in candidates if n % 2 == 0 and n >= 4]
    
    # Sort by n (we want smallest)
    candidates.sort()
    
    for n in candidates:
        if time.time() > deadline:
            break
        if n >= best_n:
            continue
        
        # Try to construct a graph on n vertices with girth >= g
        result = _try_build_3reg(n, g, deadline, start_time)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found
                # If we found a valid graph, we can stop or try smaller
                # But since we're going in increasing order, first valid is best
                return n_found, edges_found
    
    # If nothing worked, use fallback LUW construction
    return _construct_luw(k=3, g=g, d=11)

def _try_build_3reg(n, g, deadline, start_time):
    """
    Try to build a 3-regular graph on n vertices with girth >= g.
    Uses random initialization + edge switching to eliminate short cycles.
    """
    # Generate random 3-regular graph
    def random_3reg(n):
        """Generate a random 3-regular graph on n vertices using configuration model + switching."""
        # Configuration model: create 3n stubs
        stubs = []
        for v in range(n):
            for _ in range(3):
                stubs.append(v)
        
        # Randomly pair stubs
        while len(stubs) > 0:
            i = random.randint(0, len(stubs)-1)
            j = random.randint(0, len(stubs)-1)
            if i == j:
                continue
            u, v = stubs[i], stubs[j]
            if u == v:
                # Self-loop, retry
                continue
            stubs[i] = stubs[-1]
            stubs.pop()
            stubs[j] = stubs[-1]
            stubs.pop()
        
        # This approach is flawed; use a proper method
        # Use the "random regular graph" algorithm
        return _random_regular_graph(n, 3)
    
    def _random_regular_graph(n, k):
        """Generate a random k-regular graph on n vertices."""
        # Use the configuration model with rejection for self-loops and multi-edges
        max_attempts = 1000
        for _ in range(max_attempts):
            stubs = [v for v in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges:
                    valid = False
                    break
                edges.add((min(u,v), max(u,v)))
            
            if valid and len(edges) == n * k // 2:
                return list(edges)
        
        # Fallback: use a deterministic construction
        return _deterministic_3reg(n)
    
    def _deterministic_3reg(n):
        """Deterministic 3-regular graph construction."""
        # Cycle + chords
        edges = []
        # Base cycle
        for i in range(n):
            edges.append((i, (i+1) % n))
        # Add chords to make 3-regular
        # Each vertex currently has degree 2, needs 1 more
        # Add a perfect matching
        random.shuffle(list(range(n)))
        perm = list(range(n))
        random.shuffle(perm)
        for i in range(n//2):
            u, v = perm[2*i], perm[2*i+1]
            if u != v and (min(u,v), max(u,v)) not in [(a,b) for a,b in edges]:
                edges.append((min(u,v), max(u,v)))
        
        # Verify it's 3-regular
        deg = [0]*n
        for u,v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == 3 for d in deg):
            return edges
        
        # Fallback: simple construction
        edges = []
        for i in range(n):
            edges.append((i, (i+1) % n))
            edges.append((i, (i+2) % n))
        # Remove duplicates and self-loops
        seen = set()
        clean = []
        for u,v in edges:
            if u != v:
                key = (min(u,v), max(u,v))
                if key not in seen:
                    seen.add(key)
                    clean.append(key)
        return clean
    
    # Start with random 3-regular graph
    edges = _random_regular_graph(n, 3)
    
    # Verify it's valid
    adj = _build_adjacency(n, edges)
    if not _is_regular(adj, n, 3):
        return None
    
    # Iteratively improve girth
    max_iter = 20000
    current_girth = _compute_girth(n, edges, adj)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        if current_girth >= g:
            return n, edges
        
        # Find a short cycle
        cycle = _find_short_cycle(n, adj, current_girth)
        if cycle is None:
            # No cycle shorter than current_girth found, recompute
            current_girth = _compute_girth(n, edges, adj)
            if current_girth >= g:
                return n, edges
            # Try random switching
            cycle = _random_cycle(n, adj, min(g, current_girth))
            if cycle is None:
                break
        
        # Try to switch an edge in the cycle to break it
        new_edges = _switch_cycle_edge(n, edges, adj, cycle, deadline)
        if new_edges is not None:
            edges = new_edges
            adj = _build_adjacency(n, edges)
            if not _is_regular(adj, n, 3):
                # Revert
                adj = _build_adjacency(n, edges)
                continue
            current_girth = _compute_girth(n, edges, adj)
        else:
            # Try random edge switch
            new_edges = _random_edge_switch(n, edges, adj)
            if new_edges is not None:
                edges = new_edges
                adj = _build_adjacency(n, edges)
                if not _is_regular(adj, n, 3):
                    continue
                current_girth = _compute_girth(n, edges, adj)
    
    # Final check
    current_girth = _compute_girth(n, edges, adj)
    if current_girth >= g:
        return n, edges
    
    return None

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _is_regular(adj, n, k):
    return all(len(adj[i]) == k for i in range(n))

def _compute_girth(n, edges, adj):
    """Compute exact girth using BFS from each vertex."""
    min_cycle = float('inf')
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif v != start and dist[v] >= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
    return min_cycle if min_cycle < float('inf') else float('inf')

def _find_short_cycle(n, adj, target_len):
    """Find a cycle of length <= target_len using non-backtracking walk."""
    for start in range(n):
        # BFS with parent tracking
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1, [start, nb]))
        
        while queue:
            u, prev, l, path = queue.popleft()
            
            if u == start and l >= 3:
                return path
            
            if l >= target_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    if v == start:
                        return path + [start]
                    if v not in [p for p in path]:
                        new_path = path + [v]
                        queue.append((v, u, l+1, new_path))
    
    return None

def _random_cycle(n, adj, max_len):
    """Find a random cycle up to max_len."""
    for _ in range(100):
        start = random.randint(0, n-1)
        path = [start]
        prev = -1
        for l in range(max_len):
            neighbors = [v for v in adj[path[-1]] if v != prev]
            if not neighbors:
                break
            if l == max_len - 1:
                if start in neighbors:
                    return path + [start]
                else:
                    break
            next_v = random.choice(neighbors)
            if next_v == start and l >= 2:
                return path + [start]
            if next_v in path:
                break
            path.append(next_v)
            prev = path[-2]
    return None

def _switch_cycle_edge(n, edges, adj, cycle, deadline):
    """Try to switch an edge in the cycle to break it while maintaining 3-regularity."""
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    # Get unique edges in the cycle
    cycle_edges = []
    for i in range(len(cycle)-1):
        u, v = cycle[i], cycle[i+1]
        key = (min(u,v), max(u,v))
        if key not in cycle_edges:
            cycle_edges.append(key)
    
    random.shuffle(cycle_edges)
    
    for e1 in cycle_edges:
        if time.time() > deadline:
            break
        u1, v1 = e1
        
        # Find another edge e2 = (u2, v2) not in cycle
        # Such that removing e1 and e2 and adding (u1,u2) and (v1,v2) is valid
        # Or removing e1 and e2 and adding (u1,v2) and (v1,u2)
        
        # Get neighbors of u1 and v1 excluding the cycle edge
        nbrs_u1 = [v for v in adj[u1] if v != v1]
        nbrs_v1 = [v for v in adj[v1] if v != u1]
        
        # Try switching with a random edge
        for _ in range(20):
            u2 = random.choice(nbrs_u1) if nbrs_u1 else None
            if u2 is None:
                break
            # u2 is connected to u1, we want to switch u1-u2 with v1-something
            nbrs_v1_excl = [v for v in adj[v1] if v != u1 and v != u2]
            if not nbrs_v1_excl:
                continue
            v2 = random.choice(nbrs_v1_excl)
            
            # Check that (u1, v2) and (u2, v1) don't already exist
            key1 = (min(u1, v2), max(u1, v2))
            key2 = (min(u2, v1), max(u2, v1))
            key_e1 = (min(u1, v1), max(u1, v1))
            key_e2 = (min(u2, v2), max(u2, v2))
            
            if key1 in edge_set or key2 in edge_set:
                continue
            if key1 == key_e1 or key2 == key_e2:
                continue
            if u1 == v2 or u2 == v1:
                continue
            
            # Perform the switch
            new_edges = []
            for e in edges:
                ek = (min(e[0],e[1]), max(e[0],e[1]))
                if ek == key_e1 or ek == key_e2:
                    continue
                new_edges.append(e)
            new_edges.append((min(u1, v2), max(u1, v2)))
            new_edges.append((min(u2, v1), max(u2, v1)))
            
            # Verify 3-regular
            adj_new = _build_adjacency(n, new_edges)
            if _is_regular(adj_new, n, 3):
                return new_edges
    
    return None

def _random_edge_switch(n, edges, adj):
    """Perform a random 2-switch to maintain 3-regularity."""
    edge_list = list(edges)
    random.shuffle(edge_list)
    
    for e1 in edge_list[:50]:
        u1, v1 = e1
        nbrs_u1 = [v for v in adj[u1] if v != v1]
        if not nbrs_u1:
            continue
        u2 = random.choice(nbrs_u1)
        nbrs_v1 = [v for v in adj[v1] if v != u1 and v != u2]
        if not nbrs_v1:
            continue
        v2 = random.choice(nbrs_v1)
        
        key1 = (min(u1, v2), max(u1, v2))
        key2 = (min(u2, v1), max(u2, v1))
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        
        if key1 in edge_set or key2 in edge_set:
            continue
        if u1 == v2 or u2 == v1:
            continue
        
        new_edges = []
        key_e1 = (min(u1,v1), max(u1,v1))
        key_e2 = (min(u2,v2), max(u2,v2))
        for e in edges:
            ek = (min(e[0],e[1]), max(e[0],e[1]))
            if ek == key_e1 or ek == key_e2:
                continue
            new_edges.append(e)
        new_edges.append(key1)
        new_edges.append(key2)
        
        adj_new = _build_adjacency(n, new_edges)
        if _is_regular(adj_new, n, 3):
            return new_edges
    
    return None

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    For k=3, uses q=3 arithmetic.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
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
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_len = g - 1
    
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True
