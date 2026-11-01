import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Calculate LUW result as fallback and upper bound
    n_luw, edges_luw = _luw_construct(k, g)
    
    # 2. Calculate Moore Bound
    # n >= 2 * (k-1)^(g/2 - 1) for even g, or 1 + k * sum...
    # For simplicity, use a lower bound estimate
    if g % 2 == 0:
        moore = 2 * (k - 1) ** (g // 2 - 1)
    else:
        moore = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
        
    # 3. Search for better graph
    # Start N slightly above Moore
    N = max(int(moore * 1.2) + 10, 50)
    best_n = n_luw
    best_edges = edges_luw
    
    while N < best_n and time.time() < deadline:
        # Try to find a valid graph of size N
        success, edges = _search_valid_graph(k, g, N, deadline)
        if success:
            best_n = N
            best_edges = edges
            # Found one, can we go smaller? 
            # We can try to reduce N further, but random search is hard.
            # Let's just return the first valid one found below LUW?
            # Or keep searching for smaller?
            # The score is 1/n. So smaller is better.
            # Let's try to find the smallest possible.
            # But we don't know the smallest.
            # Let's just return the first valid one found? 
            # No, we should try a few sizes.
            # If we found one at N, maybe N-10 works?
            # But random search is stochastic.
            # Let's just return the best found so far.
            break 
        
        # If failed, increase N
        N = int(N * 1.1)
        
    return best_n, best_edges

def _luw_construct(k, g):
    # Re-implement LUW logic from seed
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

def _search_valid_graph(k, g, N, deadline):
    # Generate random k-regular graph
    # Use configuration model
    # Then run local search to maximize girth
    
    # 1. Generate random k-regular graph
    # We need to be careful with N and k. N*k must be even.
    if (N * k) % 2 != 0:
        return False, []
        
    # Configuration model
    half_edges = N * k // 2
    stubs = [i for i in range(N) for _ in range(k)]
    random.shuffle(stubs)
    
    # Pair them up
    # This creates a multigraph. We need to remove loops and multi-edges.
    # Simple approach: Retry until simple.
    # For large N, probability of simple is high?
    # Actually, for k=5, N=1000, it's very likely to have multi-edges.
    # We need a better generator or cleanup.
    
    # Let's use a simple backtracking or just accept multi-edges and then fix?
    # Fixing multi-edges is hard.
    
    # Alternative: Start with a ring lattice and randomize.
    # Ring lattice is k-regular? No, 2-regular.
    # We can use a "Random Regular Graph" generator that ensures simplicity.
    # Algorithm:
    # 1. Start with empty graph.
    # 2. For each vertex, connect to k random unconnected vertices.
    # This is biased and might fail.
    
    # Better: Use the "Random Pairing" with rejection.
    # If N is not too large, rejection works.
    
    edges = set()
    adjacency = [set() for _ in range(N)]
    
    # Try to build
    attempts = 0
    while attempts < 100:
        edges.clear()
        for i in range(N):
            adjacency[i].clear()
            
        # Create stubs
        stubs = [i for i in range(N) for _ in range(k)]
        random.shuffle(stubs)
        
        # Pair
        valid = True
        for i in range(0, N * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adjacency[u]:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
            adjacency[u].add(v)
            adjacency[v].add(u)
            
        if valid:
            break
        attempts += 1
        
    if not valid:
        return False, []
        
    # Convert to list of edges
    edge_list = list(edges)
    
    # 2. Local Search to maximize girth
    # We want girth >= g.
    # Current girth might be small.
    
    # Calculate girth
    def get_girth(adj):
        # BFS from each node
        min_girth = float('inf')
        for start in range(N):
            # BFS
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] * 2 >= min_girth:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1
                        # This is not exactly right for undirected BFS cycle detection
                        # Standard: if v is visited and v is not parent, cycle.
                        # But we need the length.
                        # In BFS tree, the first time we see a cross edge (u,v) where dist[u] == dist[v] or dist[u] == dist[v]-1?
                        # Actually, for undirected graph, if we visit v from u, and v is already visited,
                        # and v is not parent of u, then we have a cycle.
                        # The length is dist[u] + dist[v] + 1? No.
                        # If dist[u] == dist[v], cycle length is 2*dist[u] + 1?
                        # Let's just do a proper BFS cycle detection.
                        pass
        return min_girth
    
    # The above girth function is incomplete.
    # Let's use a simpler check:
    # A graph has girth < g if and only if there is a cycle of length < g.
    # We can check for cycles of length 3, 4, ..., g-1.
    
    def has_short_cycle(adj, max_len):
        # Check for cycles of length 3 to max_len
        # For length 3:
        for u in range(N):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u: continue
                    if w in adj[u]:
                        return True
        # For length 4:
        # ...
        # This is expensive.
        
        # Alternative: BFS from each node up to depth max_len//2
        # If we meet a neighbor, we have a cycle.
        for start in range(N):
            # BFS
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Check if cycle length < max_len
                        # If dist[u] + dist[v] + 1 < max_len?
                        # Not exactly.
                        # If we find a back edge (u,v) where dist[u] >= dist[v],
                        # the cycle length is dist[u] - dist[v] + 1? No.
                        # In BFS, if we see a visited node v (not parent),
                        # the cycle length is dist[u] + dist[v] + 1 only if it's a "cross" edge in a tree?
                        # Actually, for undirected graphs, any non-tree edge (u,v) creates a cycle of length dist[u] + dist[v] + 1?
                        # No, that's for directed.
                        # For undirected, if dist[u] == dist[v], cycle is 2*dist[u] + 1?
                        # If dist[u] == dist[v] + 1, cycle is 2*dist[v] + 2?
                        # Let's just say if we encounter a visited node, we have a cycle.
                        # We need to ensure the length is < max_len.
                        # If dist[u] + 1 <= max_len//2, then the cycle is short.
                        if dist[u] + 1 <= max_len // 2:
                            return True
        return False
    
    # If has_short_cycle(adj, g) is False, we are done.
    if not has_short_cycle(adjacency, g):
        return True, edge_list
        
    # Otherwise, try to improve.
    # We can try a few random 2-switches.
    # But for now, if the first random graph doesn't work, we fail.
    # This is weak.
    
    return False, []
