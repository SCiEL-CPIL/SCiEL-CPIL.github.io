"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

_DEADLINE = 110.0  # safety margin under 120s wall-clock kill (task says 120s, prompt says 150s, use 110 to be safe)

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_objective(x, adj, n):
    """
    Compute J = V_sub - C_sub for the subgraph defined by x.
    x is a set of edges (u, v) that are 'on'.
    """
    if not x:
        return 0
    
    # Degrees in subgraph
    deg = [0] * n
    for u, v in x:
        deg[u] += 1
        deg[v] += 1
    
    # Vertices with degree > 0
    v_sub = sum(1 for d in deg if d > 0)
    
    # Connected components
    visited = [False] * n
    c_sub = 0
    for i in range(n):
        if deg[i] > 0 and not visited[i]:
            c_sub += 1
            # BFS
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if deg[nb] > 0 and not visited[nb] and (tuple(sorted((node, nb))) in x):
                        visited[nb] = True
                        queue.append(nb)
    
    return v_sub - c_sub

def _extract_cycles(x, adj, n):
    """
    Extract edge-disjoint cycles from the even subgraph x.
    Returns list of cycles (list of vertices).
    """
    if not x:
        return []
    
    remaining = set(x)
    cycles = []
    
    # Build adjacency for remaining
    while remaining:
        # Find a vertex with degree > 0 in remaining
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        start = None
        for v in deg:
            if deg[v] > 0:
                start = v
                break
        
        if start is None:
            break
        
        # Walk to find a cycle
        # Since all degrees are even, we can just follow edges until we return to start
        # But we need a simple cycle.
        # In an even subgraph, any edge is part of some cycle.
        # We can find a cycle by DFS.
        
        # Simple approach: pick an edge, then find a path back.
        # Or use Hierholzer's algorithm on the remaining subgraph?
        # Hierholzer gives an Eulerian circuit, which is a closed walk.
        # We can decompose the closed walk into simple cycles.
        
        # Let's just find one simple cycle using DFS.
        adj_rem = defaultdict(list)
        for u, v in remaining:
            adj_rem[u].append(v)
            adj_rem[v].append(u)
        
        # DFS to find a cycle
        visited = {start}
        path = [start]
        parent = {start: -1}
        cycle = None
        
        stack = [start]
        while stack:
            node = stack[-1]
            if adj_rem[node]:
                nb = adj_rem[node].pop()
                if nb == start and len(path) >= 3:
                    cycle = path[:]
                    break
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    parent[nb] = node
                    stack.append(nb)
                else:
                    # Back edge, but we only care about closing to start for simplicity in this extraction
                    # Actually, any back edge in DFS forms a cycle.
                    # But we want to remove edges from 'remaining'.
                    # If we find any cycle, we can remove it.
                    if nb != parent.get(node, -1):
                        # Found a cycle involving node and nb
                        # Reconstruct cycle from nb to node
                        cyc = [nb]
                        cur = node
                        while cur != nb:
                            cyc.append(cur)
                            cur = parent[cur]
                        cyc.reverse()
                        cycle = cyc
                        break
            else:
                stack.pop()
                path.pop()
                visited.discard(node)
        
        if not cycle:
            # Should not happen if deg > 0
            break
        
        # Remove cycle edges from remaining
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            remaining.discard(e)
        
        cycles.append(cycle)
    
    return cycles

def _decompose_one(edges, deadline):
    if not edges:
        return []
    
    n = max(max(u, v) for u, v in edges) + 1
    adj = _build_adj(edges)
    edge_list = [tuple(sorted(e)) for e in edges]
    edge_set = set(edge_list)
    
    # Precompute triangles for faster cycle generation
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        triangles.append((u, v, w))
    
    # Initial state: all edges off
    x = set()
    best_x = set()
    best_J = 0
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    steps = 50000
    
    current_J = 0
    
    start_time = time.time()
    
    for i in range(steps):
        if time.time() > deadline:
            break
        
        # Pick a random cycle
        if triangles and random.random() < 0.5:
            # Pick a random triangle
            tri = random.choice(triangles)
            cycle = [tri[0], tri[1], tri[2]]
        else:
            # Pick a random edge and try to find a cycle
            if not edge_list:
                break
            u, v = random.choice(edge_list)
            # Random walk from v to u
            visited = {v}
            path = [v]
            cur = v
            found = False
            # Limit walk length
            for _ in range(10):
                if not adj[cur]:
                    break
                nbs = list(adj[cur])
                random.shuffle(nbs)
                next_v = None
                for nb in nbs:
                    if nb == u and len(path) >= 3:
                        next_v = u
                        found = True
                        break
                    elif nb not in visited:
                        next_v = nb
                        break
                if next_v is None:
                    break
                path.append(next_v)
                visited.add(next_v)
                cur = next_v
            if found:
                cycle = path
            else:
                continue
        
        if not cycle or len(cycle) < 3:
            continue
            
        # Compute edges of this cycle
        cycle_edges = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e not in edge_set:
                # Invalid cycle (edge not in graph)
                continue
            cycle_edges.add(e)
        
        if not cycle_edges:
            continue
            
        # Flip
        new_x = set(x)
        for e in cycle_edges:
            if e in new_x:
                new_x.discard(e)
            else:
                new_x.add(e)
        
        new_J = _compute_objective(new_x, adj, n)
        
        # Acceptance
        if new_J > current_J or random.random() < pow(2.71828, (new_J - current_J) / T):
            x = new_x
            current_J = new_J
            if current_J > best_J:
                best_J = current_J
                best_x = set(x)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Extract cycles from best_x
    cycles = _extract_cycles(best_x, adj, n)
    
    # Remaining edges
    used_edges = set()
    for c in cycles:
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            used_edges.add(tuple(sorted((u, v))))
    
    remaining_edges = edge_set - used_edges
    
    pieces = cycles
    for e in sorted(remaining_edges):
        pieces.append(list(e))
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    # Allocate time budget
    time_per_graph = 30.0
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        deadline = time.time() + time_per_graph
        pieces = _decompose_one(edges, deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
