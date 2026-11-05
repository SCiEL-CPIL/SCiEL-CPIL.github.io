import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve_one(edges, n, deadline):
    if not edges:
        return []
    
    m = len(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # 1. Find T: vertices with odd degree
    degrees = [0] * n
    for u, v in edge_set:
        degrees[u] += 1
        degrees[v] += 1
        
    T = [i for i in range(n) if degrees[i] % 2 != 0]
    
    # 2. If T is not empty, find Min T-Join
    single_edges = set()
    if T:
        # Compute all-pairs shortest paths
        # Use Dijkstra or BFS since unweighted
        dist = [[-1] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                # Need adjacency list
                # Build adj from edge_set
                # This is done inside the loop for each i, which is inefficient.
                # Let's build adj once.
                pass
        
    # Build adjacency list once
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
        
    # Re-do shortest paths with adj
    dist = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    queue.append(v)
                    
    # Min Weight Perfect Matching on T
    # T size is small. Use DP or recursive.
    # Let's use a simple recursive matching with memoization.
    # Key: frozenset of remaining vertices in T
    # Since T is a list, we can map indices.
    
    if len(T) == 0:
        single_edges = set()
    else:
        # Map T vertices to indices 0..k-1
        t_index = {v: i for i, v in enumerate(T)}
        k = len(T)
        
        # dp[mask] = min cost to match vertices in mask
        # mask is a bitmask of size k
        # Initialize with infinity
        INF = 10**9
        dp = [INF] * (1 << k)
        dp[0] = 0
        
        # Precompute weights between T vertices
        w = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(i+1, k):
                w[i][j] = dist[T[i]][T[j]]
                w[j][i] = w[i][j]
                
        for mask in range(1, 1 << k):
            # Find the first set bit
            # If mask is 0, skip
            # We want to pick the first vertex i in mask, and match it with j in mask
            # i = lowest set bit
            i = (mask & -mask).bit_length() - 1
            # Try matching i with every other j in mask
            rest = mask & ~(1 << i)
            for j in range(k):
                if rest & (1 << j):
                    new_mask = rest & ~(1 << j)
                    if dp[new_mask] + w[i][j] < dp[mask]:
                        dp[mask] = dp[new_mask] + w[i][j]
                        
        # Now we need to reconstruct the matching to get the paths
        # This is harder. We can just use the cost to verify, but we need the edges.
        # Alternative: Just find the matching recursively.
        
        def find_matching(mask):
            if mask == 0:
                return []
            i = (mask & -mask).bit_length() - 1
            rest = mask & ~(1 << i)
            best_cost = INF
            best_j = -1
            for j in range(k):
                if rest & (1 << j):
                    new_mask = rest & ~(1 << j)
                    cost = dp[new_mask] + w[i][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_j = j
            return [(T[i], T[best_j])] + find_matching(rest & ~(1 << best_j))
            
        matching = find_matching((1 << k) - 1)
        
        # Find shortest paths for each pair
        # We need the actual paths, not just distances.
        # Re-run BFS to store parent pointers.
        
        # For each pair (u, v) in matching, find a shortest path
        # We can do BFS from u to v.
        
        # To get the symmetric difference, we just count how many times each edge is used.
        # Edge usage count: +1 for each path that uses it.
        # If count is odd, edge is in T-join.
        
        edge_usage = defaultdict(int)
        
        for u, v in matching:
            # BFS to find path from u to v
            parent = [-1] * n
            parent[u] = u
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                if curr == v:
                    break
                for nb in adj[curr]:
                    if parent[nb] == -1:
                        parent[nb] = curr
                        queue.append(nb)
                        
            # Reconstruct path
            path = []
            curr = v
            while curr != u:
                path.append((curr, parent[curr]))
                curr = parent[curr]
                
            for a, b in path:
                e = tuple(sorted((a, b)))
                edge_usage[e] += 1
                
        # T-join edges are those with odd usage
        single_edges = {e for e, cnt in edge_usage.items() if cnt % 2 == 1}
        
    # 3. Remaining edges form Eulerian graph
    cycle_edges = edge_set - single_edges
    
    # 4. Decompose cycle_edges into simple cycles
    # Greedy longest cycle
    adj_cyc = [set() for _ in range(n)]
    for u, v in cycle_edges:
        adj_cyc[u].add(v)
        adj_cyc[v].add(u)
        
    pieces = [list(e) for e in single_edges]
    
    rng = random.Random(42)
    
    # Time check
    start_time = time.time()
    
    while cycle_edges:
        if time.time() > deadline - 2.0:
            # Fallback: add remaining as single edges
            pieces.extend([list(e) for e in cycle_edges])
            break
            
        # Find longest cycle in adj_cyc
        best_cycle = None
        best_len = 0
        
        # Try a few random starts
        for _ in range(10):
            # Pick a random vertex with degree > 0
            candidates = [i for i in range(n) if adj_cyc[i]]
            if not candidates:
                break
            start = rng.choice(candidates)
            
            visited = [False] * n
            path = []
            
            def dfs(u):
                nonlocal best_len, best_cycle
                visited[u] = True
                path.append(u)
                
                for v in adj_cyc[u]:
                    if v == start and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    elif not visited[v]:
                        dfs(v)
                        
                path.pop()
                visited[u] = False
                
            dfs(start)
            
            if best_len == n:
                break
                
        if not best_cycle:
            # No cycle found? Should not happen if Eulerian and connected components have cycles
            # If it's a tree (which it isn't, it's Eulerian), or just edges?
            # Eulerian graphs with edges must have cycles.
            # If we fail to find one, we might have a bug or it's a single edge? No, min cycle 3.
            # If adj_cyc has edges but no cycle, it's a forest. But Eulerian implies even degrees.
            # A forest with even degrees is a set of isolated vertices.
            # So if cycle_edges is not empty, there must be a cycle.
            # If we fail, we break and add remaining as singles.
            pieces.extend([list(e) for e in cycle_edges])
            break
            
        # Remove best_cycle from graph
        L = len(best_cycle)
        cycle_edges_to_remove = set()
        for i in range(L):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % L]
            e = tuple(sorted((u, v)))
            cycle_edges_to_remove.add(e)
            
        cycle_edges -= cycle_edges_to_remove
        for u, v in cycle_edges_to_remove:
            adj_cyc[u].remove(v)
            adj_cyc[v].remove(u)
            
        pieces.append(best_cycle)
        
    return pieces

def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
