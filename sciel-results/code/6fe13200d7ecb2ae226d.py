import random
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    # Try to find a smaller graph using local search
    # Only attempt if k is small (e.g., 3) and g is not too small
    if k == 3 and g >= 8:
        start_time = time.time()
        deadline = start_time + 100.0 # 100 seconds budget
        
        # Try different sizes
        # Moore bound for k=3 is approx 2 * 2^(g/2 - 1)
        # We try sizes significantly larger than Moore bound but smaller than LUW
        # LUW for g=16 is ~350k. We try up to 10k.
        sizes_to_try = [1000, 2000, 4000, 8000]
        
        best_result = None
        
        for N in sizes_to_try:
            if time.time() > deadline:
                break
            
            # Generate random 3-regular graph
            # Use a simple configuration model or random swaps
            # For simplicity, generate a random 3-regular graph using a known method
            # or just random edges and fix degrees (hard).
            # Let's use a simple random regular graph generator.
            
            def generate_random_regular_graph(n, k):
                if n * k % 2 != 0:
                    return None
                # Use a simple algorithm: create a list of half-edges and pair them
                # This might create multi-edges or self-loops, which we need to fix.
                # For k=3, n must be even.
                if n % 2 != 0:
                    return None
                
                # Simple method: start with a cycle, then add edges
                # Or use the "random swaps" method on a fixed structure.
                # Let's start with a 3-regular graph: a "ladder" or similar.
                # Actually, for n=1000, we can just generate random edges and check.
                
                # Better: Use the "randomized" approach with a base graph.
                # Base: 2-regular (cycle). Add a perfect matching.
                # Cycle: 0-1-2-...-n-1-0
                # Matching: 0-n/2, 1-n/2+1, ...
                edges = []
                for i in range(n):
                    edges.append((i, (i + 1) % n))
                for i in range(n // 2):
                    edges.append((i, i + n // 2))
                
                # Shuffle edges to randomize
                random.shuffle(edges)
                
                # Now perform random swaps to randomize further
                # This ensures we don't start with a structured graph
                for _ in range(n * 10):
                    if time.time() > deadline:
                        break
                    # Pick two random edges
                    e1_idx = random.randint(0, len(edges) - 1)
                    e2_idx = random.randint(0, len(edges) - 1)
                    if e1_idx == e2_idx: continue
                    
                    u1, v1 = edges[e1_idx]
                    u2, v2 = edges[e2_idx]
                    
                    # Check if swap is valid (no multi-edges, no self-loops)
                    # New edges: (u1, u2) and (v1, v2) OR (u1, v2) and (v1, u2)
                    # Try both
                    for new_e1, new_e2 in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
                        a, b = new_e1
                        c, d = new_e2
                        if a == b or c == d: continue
                        # Check for existing edges
                        # This is O(N) per check, too slow.
                        # We need a fast check. Use a set of edges.
                        pass
                    
                    # Simplified: Just shuffle the matching part
                    # This is getting complex. Let's use a simpler random graph gen.
                    
                # Let's use a standard random 3-regular graph generator
                # Reference: "Generating Random Regular Graphs"
                # For small k, we can use the following:
                # 1. Create a list of n * k / 2 pairs of half-edges.
                # 2. Shuffle.
                # 3. Pair them.
                # 4. Check for self-loops and multi-edges. If found, retry.
                # For n=1000, k=3, probability of bad graph is low.
                
                def gen_k_regular(n, k):
                    if n * k % 2 != 0: return None
                    half_edges = []
                    for i in range(n):
                        for _ in range(k):
                            half_edges.append(i)
                    random.shuffle(half_edges)
                    edges = []
                    for i in range(len(half_edges) // 2):
                        u = half_edges[2*i]
                        v = half_edges[2*i+1]
                        if u == v:
                            return None
                        # Check for duplicate
                        if (u, v) in edges or (v, u) in edges:
                            return None
                        edges.append((u, v))
                    return edges
                
                graph_edges = gen_k_regular(N, k)
                if graph_edges is None:
                    continue
                
                # Compute girth
                # Build adjacency list
                adj = [[] for _ in range(N)]
                for u, v in graph_edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # BFS from each vertex to find shortest cycle
                # This is O(N * E) = O(N^2). For N=1000, 10^6 ops. OK.
                # For N=8000, 64 * 10^6 ops. Might be slow in Python.
                # We will limit N to 4000 for safety.
                
                def compute_girth(adj, n):
                    min_girth = n + 1
                    for start in range(n):
                        # BFS
                        dist = [-1] * n
                        parent = [-1] * n
                        dist[start] = 0
                        queue = deque([start])
                        found = False
                        while queue and not found:
                            u = queue.popleft()
                            for v in adj[u]:
                                if dist[v] == -1:
                                    dist[v] = dist[u] + 1
                                    parent[v] = u
                                    queue.append(v)
                                elif v != parent[u]:
                                    # Found a cycle
                                    # Length is dist[u] + dist[v] + 1
                                    # Wait, this finds *a* cycle, not necessarily the shortest one involving start.
                                    # To find the shortest cycle in the graph, we need to be careful.
                                    # Standard BFS from each vertex finds the shortest cycle *passing through* that vertex?
                                    # No, BFS from start finds the shortest path from start to any node.
                                    # If we hit a visited node, we found a cycle.
                                    # The length is dist[u] + dist[v] + 1.
                                    # This is the length of the cycle formed by the two paths.
                                    # It is guaranteed to be a simple cycle.
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < min_girth:
                                        min_girth = cycle_len
                                    # We can break if we found a very short cycle?
                                    # No, we want the minimum.
                                    # But if we found a cycle of length L, we don't need to explore deeper for this start?
                                    # Actually, BFS explores in increasing distance.
                                    # The first back-edge found gives the shortest cycle *for this start*.
                                    # So we can break the inner loop?
                                    # No, there might be a shorter cycle involving a different neighbor?
                                    # BFS ensures dist is minimal.
                                    # So the first back-edge found for a given start gives the shortest cycle containing start.
                                    # So we can break the BFS for this start.
                                    found = True
                                    break
                        # Optimization: If min_girth is already <= 3, break
                        if min_girth <= 3:
                            return min_girth
                    return min_girth
                
                girth = compute_girth(adj, N)
                
                if girth >= g:
                    return N, graph_edges
                
                # If not, try to improve via local search
                # We will try a few swaps
                for _ in range(100): # Limit steps per size
                    if time.time() > deadline:
                        break
                    
                    # Find shortest cycle
                    # We need the actual cycle to break it
                    # Let's find one
                    cycle = None
                    for start in range(N):
                        dist = [-1] * N
                        parent = [-1] * N
                        dist[start] = 0
                        queue = deque([start])
                        while queue:
                            u = queue.popleft()
                            for v in adj[u]:
                                if dist[v] == -1:
                                    dist[v] = dist[u] + 1
                                    parent[v] = u
                                    queue.append(v)
                                elif v != parent[u]:
                                    # Reconstruct cycle
                                    # Path 1: start -> ... -> u
                                    # Path 2: start -> ... -> v
                                    # Cycle: u -> ... -> start -> ... -> v -> u
                                    # We need the path from u to start and v to start
                                    path_u = []
                                    curr = u
                                    while curr != start:
                                        path_u.append(curr)
                                        curr = parent[curr]
                                    path_u.append(start)
                                    
                                    path_v = []
                                    curr = v
                                    while curr != start:
                                        path_v.append(curr)
                                        curr = parent[curr]
                                    path_v.append(start)
                                    
                                    # Cycle is path_u + path_v reversed (excluding start)
                                    # Actually, the cycle is u -> parent[u] -> ... -> start -> parent[v] -> ... -> v -> u
                                    # Let's just store the nodes
                                    cycle_nodes = path_u + path_v[::-1]
                                    cycle = cycle_nodes
                                    break
                            if cycle: break
                        if cycle: break
                        if min_girth <= 3: break
                    
                    if not cycle:
                        break
                    
                    # Pick a random edge in the cycle to swap
                    # Cycle nodes: c0, c1, ..., cm
                    # Edges: (c0, c1), (c1, c2), ...
                    edge_idx = random.randint(0, len(cycle) - 1)
                    u = cycle[edge_idx]
                    v = cycle[(edge_idx + 1) % len(cycle)]
                    
                    # Find a random non-adjacent edge to swap with
                    # We want to break the cycle.
                    # Swap (u, v) with (x, y) -> (u, x) and (v, y)
                    # We need x, y such that (u, x) and (v, y) are not existing edges
                    # and not self-loops.
                    # Also, x, y should not be in the cycle (to avoid creating new short cycles)
                    
                    # Pick random x, y
                    swapped = False
                    for _ in range(10):
                        x = random.randint(0, N - 1)
                        y = random.randint(0, N - 1)
                        if x == y or x == u or x == v or y == u or y == v:
                            continue
                        # Check if (x, y) is an edge
                        if y not in adj[x]:
                            continue
                        # Check if (u, x) and (v, y) are not edges
                        if x in adj[u] or y in adj[v]:
                            continue
                        # Check if (u, y) and (v, x) are not edges (for the other swap option)
                        # Let's try (u, x) and (v, y)
                        
                        # Perform swap
                        # Remove (u, v) and (x, y)
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
                        # Add (u, x) and (v, y)
                        adj[u].append(x)
                        adj[x].append(u)
                        adj[v].append(y)
                        adj[y].append(v)
                        
                        # Update graph_edges
                        # This is hard to maintain. We'll rebuild at the end if successful.
                        # For now, just check girth.
                        
                        new_girth = compute_girth(adj, N)
                        if new_girth > girth:
                            girth = new_girth
                            if girth >= g:
                                # Rebuild edges
                                final_edges = []
                                for i in range(N):
                                    for j in adj[i]:
                                        if i < j:
                                            final_edges.append((i, j))
                                return N, final_edges
                            swapped = True
                            break
                        else:
                            # Revert
                            adj[u].remove(x)
                            adj[x].remove(u)
                            adj[v].remove(y)
                            adj[y].remove(v)
                            
                            adj[u].append(v)
                            adj[v].append(u)
                            adj[x].append(y)
                            adj[y].append(x)
                            
                    if not swapped:
                        continue
                        
    # Fallback to LUW
    return luw_construct(k, g)

# EVOLVE-BLOCK-END
