"""Structural rewrite: max edge-disjoint Hamiltonian cycles via 2-factor (bipartite matching)."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _hopcroft_karp(adj, n):
    # left nodes 0..n-1, right nodes 0..n-1, edge i->j if j in adj[i]
    pairU = [-1] * n
    pairV = [-1] * n
    dist = [0] * n
    from collections import deque

    def bfs():
        q = deque()
        for u in range(n):
            if pairU[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pairV[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pairU[u] == -1 and dfs(u):
                matching += 1
    return pairU, matching

def _find_2factor(adj, n, seed):
    # Find a 2-regular spanning subgraph (cycle cover) via bipartite matching.
    # Each vertex chooses 2 outgoing neighbors; each vertex receives exactly 2.
    # Build left adjacency: for vertex u, candidates = adj[u] \ {u}
    # We need a 2-regular directed subgraph. Model: each left node picks 2 right nodes,
    # each right node gets exactly 2. This is a 2-regular bipartite subgraph.
    # We solve by finding a perfect matching in an expanded graph:
    # left side: 2 copies of each vertex (u,0),(u,1); right side: 2 copies (v,0),(v,1)
    # Edge (u,i)->(v,j) if v in adj[u].
    # Actually simpler: find a 2-regular spanning subgraph by solving a 2-assignment.
    # Use the fact that a 2-regular spanning subgraph exists iff the graph has a 2-factor.
    # We'll use a reduction: find a perfect matching in the "double" graph.
    # Left: 2n nodes (u,0),(u,1) for u in 0..n-1
    # Right: 2n nodes (v,0),(v,1)
    # Edge (u,i) -> (v,j) if v in adj[u]
    # A perfect matching gives each left node one right neighbor and vice versa.
    # But we need each vertex to have degree 2 in the resulting undirected subgraph.
    # This approach finds a 1-regular directed subgraph (permutation), not 2-regular.
    # 
    # Better approach: find a 2-factor by decomposing into a perfect matching twice,
    # or use the fact that in a regular graph we can find a 2-factor.
    # 
    # Simplest reliable approach for 21 vertices: use networkx-free matching.
    # We'll find a 2-regular spanning subgraph by:
    # 1. Find a perfect matching M1 in the graph (each vertex matched once).
    # 2. Remove M1 edges, find another perfect matching M2.
    # 3. Union M1 + M2 is a 2-regular spanning subgraph (union of two perfect matchings = 2-factor).
    # This works if the graph has at least two edge-disjoint perfect matchings.

    # Find perfect matching 1
    adj1 = [set(adj[u]) for u in range(n)]
    # Remove self-loops (shouldn't exist)
    for u in range(n):
        adj1[u].discard(u)
    pairU1, m1 = _hopcroft_karp(adj1, n)
    if m1 < n:
        return None  # no perfect matching, no 2-factor

    # Record matched edges and remove them
    edges_used = set()
    for u in range(n):
        v = pairU1[u]
        if v != u:
            edges_used.add((min(u, v), max(u, v)))

    # Build adj2 without edges used in M1
    adj2 = [set(adj[u]) for u in range(n)]
    for u in range(n):
        adj2[u].discard(u)
        for (a, b) in edges_used:
            if u == a:
                adj2[u].discard(b)
            elif u == b:
                adj2[u].discard(a)

    pairU2, m2 = _hopcroft_karp(adj2, n)
    if m2 < n:
        return None  # no second perfect matching

    # Union of the two matchings gives a 2-regular spanning subgraph
    # Collect all edges
    all_edges = list(edges_used)
    for u in range(n):
        v = pairU2[u]
        if v != u:
            e = (min(u, v), max(u, v))
            all_edges.append(e)

    # Build degree-2 graph and extract cycles
    deg_adj = [set() for _ in range(n)]
    for a, b in all_edges:
        deg_adj[a].add(b)
        deg_adj[b].add(a)

    # Verify 2-regular
    for u in range(n):
        if len(deg_adj[u]) != 2:
            return None

    # Extract cycles
    visited = [False] * n
    cycles = []
    for start in range(n):
        if visited[start]:
            continue
        # Walk the cycle
        cycle = [start]
        visited[start] = True
        prev = -1
        cur = start
        while True:
            # Find next unvisited neighbor
            nxt = None
            for nb in deg_adj[cur]:
                if not visited[nb]:
                    nxt = nb
                    break
            if nxt is None:
                # All neighbors visited; check if we're back to start
                break
            prev = cur
            cur = nxt
            cycle.append(cur)
            visited[cur] = True
            if cur == start:
                break
        if len(cycle) >= 3:
            cycles.append(cycle)

    return cycles, all_edges

def _find_hamiltonian_cycle(adj, n, seed):
    """Find a Hamiltonian cycle using randomized DFS with Warnsdorff heuristic."""
    rng = random.Random(seed)
    visited = [False] * n
    path = []

    def dfs(u, depth):
        visited[u] = True
        path.append(u)
        if depth == n:
            # Check if we can close the cycle
            if 0 in adj[u]:
                return True
            return False
        # Sort neighbors by Warnsdorff: prefer those with fewer unvisited neighbors
        neighbors = [v for v in adj[u] if not visited[v]]
        if not neighbors:
            # Try to close
            if depth == n - 1 and 0 in adj[u]:
                visited[u] = False
                path.pop()
                return True
            visited[u] = False
            path.pop()
            return False
        neighbors.sort(key=lambda v: sum(1 for w in adj[v] if not visited[w]))
        rng.shuffle(neighbors)  # add randomness among ties
        for v in neighbors:
            if dfs(v, depth + 1):
                return True
        visited[u] = False
        path.pop()
        return False

    # Start from vertex 0
    if dfs(0, 1):
        return list(path)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    n = P['n']
    adj = _build_adj()

    best_cycles = []
    best_edges_used = set()

    # Strategy: iteratively find Hamiltonian cycles (2-factors that are single cycles)
    # using randomized DFS, removing edges after each successful cycle.
    
    # First, try the 2-factor approach to get a baseline
    seed = 42
    cycles_2f, edges_2f = _find_2factor(adj, n, seed)
    if cycles_2f is not None:
        # Check if it's a single Hamiltonian cycle
        if len(cycles_2f) == 1 and len(cycles_2f[0]) == n:
            best_cycles.append(cycles_2f[0])
            best_edges_used.update(edges_2f)

    # Now iteratively find more Hamiltonian cycles
    remaining_adj = [set(adj[u]) for u in range(n)]
    for u in range(n):
        remaining_adj[u].discard(u)
        for (a, b) in best_edges_used:
            if u == a:
                remaining_adj[u].discard(b)
            elif u == b:
                remaining_adj[u].discard(a)

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Check if remaining graph can support a Hamiltonian cycle (min degree >= 2)
        min_deg = min(len(remaining_adj[u]) for u in range(n))
        if min_deg < 2:
            break

        # Try to find a Hamiltonian cycle in the remaining graph
        hc = _find_hamiltonian_cycle(remaining_adj, n, seed + attempts)
        if hc is not None:
            # Verify it's valid
            valid = True
            new_edges = set()
            for i in range(len(hc)):
                a = hc[i]
                b = hc[(i + 1) % len(hc)]
                e = (min(a, b), max(a, b))
                if e in best_edges_used or e in new_edges:
                    valid = False
                    break
                new_edges.add(e)
            if valid:
                best_cycles.append(hc)
                best_edges_used.update(new_edges)
                # Update remaining graph
                for u in range(n):
                    for (a, b) in new_edges:
                        if u == a:
                            remaining_adj[u].discard(b)
                        elif u == b:
                            remaining_adj[u].discard(a)
                    remaining_adj[u].discard(u)
                seed += 1000
                if len(best_cycles) >= 10:
                    break
        else:
            seed += 1

    # Ensure we have at least 1 cycle (the seed)
    if not best_cycles:
        best_cycles = [list(range(n))]

    # Return as list of lists of Python ints
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])

    return result

# EVOLVE-BLOCK-END
