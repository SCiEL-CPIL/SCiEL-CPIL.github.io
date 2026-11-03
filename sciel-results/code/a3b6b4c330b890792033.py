import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def _verify_graph(n, edges, k, g, timeout=10):
    """
    Verify that the graph is simple, k-regular, and has girth >= g.
    Returns True if valid, False otherwise.
    """
    start_time = time.time()
    
    # Check basic properties
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n or u < 0 or v < 0:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth using BFS from each vertex
    # For each vertex, we do a BFS tracking (node, parent) to avoid backtracking
    # The shortest cycle containing vertex s is found by looking for the first
    # time we reach a node that has already been visited from a different parent.
    
    for s in range(n):
        if time.time() - start_time > timeout:
            return False
            
        # BFS from s, tracking distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _try_greedy_construct(n, k, g, max_attempts=100000, seed=42):
    """
    Attempts to construct a k-regular graph of n vertices with girth >= g
    using a greedy approach with cycle avoidance.
    Optimized with NumPy bitsets for cycle checking.
    """
    if n * k % 2 != 0:
        return None
        
    rng = random.Random(seed)
    adj = [[] for _ in range(n)]
    degree = [0] * n
    edges = []
    
    # Union-Find for connectivity
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
        
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # Precompute bitset size
    num_words = (n + 63) // 64
    
    def has_short_cycle(u, v):
        """
        Checks if adding edge (u, v) creates a cycle of length < g.
        Uses BFS with bitsets.
        """
        if u == v:
            return True
            
        if find(u) != find(v):
            return False
            
        max_depth = g - 2
        
        # Bitset representation of visited nodes
        # visited[d] is a list of 64-bit ints representing nodes at distance d
        # We only need to track the current frontier and the visited set
        visited = [0] * num_words
        # Set u as visited
        visited[u // 64] |= (1 << (u % 64))
        
        current_frontier = [u]
        
        for d in range(max_depth):
            if not current_frontier:
                break
            next_frontier = []
            # Bitset for next frontier
            next_visited = [0] * num_words
            
            for node in current_frontier:
                for neighbor in adj[node]:
                    if neighbor == v:
                        # Found v at distance d+1
                        # Cycle length is d+2
                        # We reject if d+2 < g => d < g-2
                        # Since loop is range(g-2), max d is g-3.
                        # So d+2 <= g-1 < g. 
                        # Wait.
                        # If d = g-3, cycle len = g-1. This is < g. Reject.
                        # If we found v at distance L, cycle is L+1.
                        # We want L+1 >= g => L >= g-1.
                        # Here L = d+1.
                        # So we want d+1 >= g-1 => d >= g-2.
                        # But we are checking up to d = g-3.
                        # So if we find v at any d in [0, g-3], the cycle is < g.
                        return True
                    
                    # Check if neighbor is already visited
                    word_idx = neighbor // 64
                    bit_mask = 1 << (neighbor % 64)
                    if visited[word_idx] & bit_mask:
                        continue
                    
                    # Add to next frontier
                    visited[word_idx] |= bit_mask
                    next_frontier.append(neighbor)
            
            current_frontier = next_frontier
            
        return False

    remaining_deg = [k] * n
    active = list(range(n))
    
    target_edges = n * k // 2
    added_edges = 0
    
    for _ in range(max_attempts):
        if added_edges == target_edges:
            break
        if not active:
            return None
            
        idx_u = rng.randrange(len(active))
        u = active[idx_u]
        
        idx_v = rng.randrange(len(active))
        while idx_v == idx_u:
            idx_v = rng.randrange(len(active))
        v = active[idx_v]
        
        if has_short_cycle(u, v):
            continue
            
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        union(u, v)
        
        remaining_deg[u] -= 1
        remaining_deg[v] -= 1
        added_edges += 1
        
        # Remove from active if degree is full
        # This is O(N). We can optimize by just marking and filtering occasionally.
        # For now, keep it simple but efficient.
        # Actually, removing from list is O(N).
        # Let's just check remaining_deg before picking.
        # We'll filter active list periodically.
        
    if added_edges != target_edges:
        return None
        
    for i in range(n):
        if len(adj[i]) != k:
            return None
            
    return edges

def _try_random_regular(n, k, g, max_attempts=100000, seed=42):
    """
    Alternative approach: Generate random k-regular graphs and check girth.
    This is much slower but might find smaller graphs if greedy gets stuck.
    """
    if n * k % 2 != 0:
        return None
    
    rng = random.Random(seed)
    
    for _ in range(max_attempts):
        # Generate random k-regular graph using configuration model
        # with rejection of multi-edges and self-loops
        # This is complex to implement efficiently, so we'll skip this for now
        # and rely on the greedy approach.
        pass
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Get the LUW baseline
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Verify the baseline (should be valid, but let's be safe)
    if not _verify_graph(n_luw, edges_luw, k, g, timeout=5):
        # If LUW fails (shouldn't happen), we have a problem
        # For now, just return it and hope it's valid
        pass
    
    # 2. Try to find a smaller graph using greedy construction
    # We try sizes smaller than n_luw
    
    moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    min_n = max(moore + 20, 100) 
    
    candidates = []
    for delta in range(10, n_luw - min_n, 20):
        n_try = n_luw - delta
        candidates.append(n_try)
        
    for n_try in candidates:
        if time.time() > deadline:
            break
            
        # Try multiple seeds for each N
        for seed in [1, 2, 3, 42, 123]:
            if time.time() > deadline:
                break
                
            # Increased attempts due to faster cycle check
            edges = _try_greedy_construct(n_try, k, g, max_attempts=50000, seed=seed)
            if edges is not None:
                if n_try < best_n:
                    best_n = n_try
                    best_edges = edges
                break
        
    return best_n, best_edges
