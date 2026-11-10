"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

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

def _hierarchy_decompose(edges_set, deadline):
    """
    Structural decomposition:
    1. Find the 2-core (vertices with degree >= 2, iteratively peel degree-1).
    2. Within the 2-core, find maximal even-degree subgraphs and use Hierholzer
       to decompose them into edge-disjoint cycles (optimal for that subgraph).
    3. For odd-degree vertices in the core, use randomized DFS to extract cycles.
    4. Remaining edges become single-edge pieces.
    """
    if not edges_set:
        return []
    
    pieces = []
    remaining = set(edges_set)
    
    # Step 1: Extract 2-core
    adj = _build_adj(remaining)
    # Iteratively remove vertices with degree < 2
    changed = True
    while changed:
        changed = False
        to_remove = []
        for v in list(adj.keys()):
            if len(adj[v]) < 2:
                to_remove.append(v)
        for v in to_remove:
            for nb in list(adj[v]):
                e = tuple(sorted((v, nb)))
                remaining.discard(e)
                adj[nb].discard(v)
            del adj[v]
            changed = True
    
    if not remaining:
        # All edges were in trees (no cycles possible)
        for e in edges_set:
            pieces.append([e[0], e[1]])
        return pieces
    
    # Step 2: Try to find even-degree subgraphs and decompose via Hierholzer
    # Group edges by connected components in the 2-core
    visited_v = set()
    components = []
    for v in list(adj.keys()):
        if v in visited_v:
            continue
        # BFS to find component
        comp = set()
        queue = deque([v])
        visited_v.add(v)
        while queue:
            cur = queue.popleft()
            comp.add(cur)
            for nb in adj[cur]:
                if nb not in visited_v:
                    visited_v.add(nb)
                    queue.append(nb)
        components.append(comp)
    
    # For each component, try Hierholzer if all degrees are even
    for comp in components:
        if time.time() > deadline:
            break
        # Check if all vertices in comp have even degree within remaining
        all_even = True
        for v in comp:
            if len(adj[v]) % 2 != 0:
                all_even = False
                break
        
        if all_even and len(comp) >= 3:
            # Use Hierholzer's algorithm to decompose into cycles
            cycles = _hierholzer_cycles(comp, adj, remaining)
            for cyc in cycles:
                if len(cyc) >= 3:
                    pieces.append(cyc)
            # Remove all edges in this component from remaining
            for v in comp:
                for nb in list(adj[v]):
                    e = tuple(sorted((v, nb)))
                    if e in remaining:
                        remaining.discard(e)
            # Rebuild adj for remaining
            adj = _build_adj(remaining)
            # Re-extract 2-core after removal
            changed = True
            while changed:
                changed = False
                to_remove = []
                for v in list(adj.keys()):
                    if len(adj[v]) < 2:
                        to_remove.append(v)
                for v in to_remove:
                    for nb in list(adj[v]):
                        e = tuple(sorted((v, nb)))
                        remaining.discard(e)
                        adj[nb].discard(v)
                    del adj[v]
                    changed = True
    
    # Step 3: For remaining edges, use randomized DFS cycle extraction
    if remaining:
        adj = _build_adj(remaining)
        # Re-extract 2-core
        changed = True
        while changed:
            changed = False
            to_remove = []
            for v in list(adj.keys()):
                if len(adj[v]) < 2:
                    to_remove.append(v)
            for v in to_remove:
                for nb in list(adj[v]):
                    e = tuple(sorted((v, nb)))
                    remaining.discard(e)
                    adj[nb].discard(v)
                del adj[v]
                changed = True
        
        # Randomized DFS to find cycles
        rng = random.Random(42)
        max_iter = 200
        for _ in range(max_iter):
            if not remaining or time.time() > deadline:
                break
            
            # Pick a random vertex with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                break
            
            start = rng.choice(candidates)
            cycle = _randomized_dfs_cycle(start, adj, remaining, rng)
            if cycle and len(cycle) >= 3:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((a, b)))
                    remaining.discard(e)
                    adj[a].discard(b)
                    adj[b].discard(a)
            
            # Periodically re-extract 2-core
            if _ % 20 == 0:
                changed = True
                while changed:
                    changed = False
                    to_remove = []
                    for v in list(adj.keys()):
                        if len(adj[v]) < 2:
                            to_remove.append(v)
                    for v in to_remove:
                        for nb in list(adj[v]):
                            e = tuple(sorted((v, nb)))
                            remaining.discard(e)
                            adj[nb].discard(v)
                        del adj[v]
                        changed = True
    
    # Step 4: Remaining edges as single-edge pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces

def _hierholzer_cycles(vertices, adj, remaining):
    """
    Apply Hierholzer's algorithm to find edge-disjoint cycles in an
    even-degree subgraph. Returns list of cycles (each a list of vertices).
    """
    cycles = []
    # Build edge list for this subgraph
    edge_set = set()
    for v in vertices:
        for nb in adj[v]:
            e = tuple(sorted((v, nb)))
            if e in remaining:
                edge_set.add(e)
    
    if not edge_set:
        return cycles
    
    # Build adjacency with edge tracking
    edge_adj = defaultdict(list)
    for u, v in edge_set:
        edge_adj[u].append(v)
        edge_adj[v].append(u)
    
    # Hierholzer's algorithm
    def find_cycle(start):
        stack = [start]
        path = []
        while stack:
            v = stack[-1]
            if edge_adj[v]:
                nb = edge_adj[v].pop()
                # Remove the edge
                if nb in edge_adj[v]:
                    edge_adj[v].remove(nb)
                else:
                    # Should not happen, but safety
                    pass
                stack.append(nb)
            else:
                path.append(stack.pop())
        return path[::-1]
    
    # Find all cycles
    processed_edges = set()
    for u, v in edge_set:
        if (u, v) in processed_edges or (v, u) in processed_edges:
            continue
        # Check if this edge is still available
        if not edge_adj[u] or v not in edge_adj[u]:
            continue
        cycle = find_cycle(u)
        if len(cycle) >= 3:
            # Ensure it's a proper cycle
            cycles.append(cycle)
        processed_edges.add((u, v))
        processed_edges.add((v, u))
    
    return cycles

def _randomized_dfs_cycle(start, adj, remaining, rng):
    """
    Randomized DFS to find a cycle starting from 'start'.
    Uses random neighbor ordering to explore diverse paths.
    """
    max_depth = 15
    for _attempt in range(5):
        path = [start]
        visited = {start}
        stack = [(start, 0)]
        
        while stack:
            node, depth = stack.pop()
            if depth >= max_depth:
                continue
            
            # Get neighbors sorted randomly, prefer high-degree
            neighbors = [nb for nb in adj[node] if nb != start and nb not in visited]
            if not neighbors:
                continue
            
            # Shuffle with bias toward high-degree nodes
            neighbors.sort(key=lambda nb: (-len(adj[nb]), rng.random()))
            
            for nb in neighbors[:3]:  # Try top 3
                e = tuple(sorted((node, nb)))
                if e not in remaining:
                    continue
                
                if nb == start and len(path) >= 3:
                    return path
                
                visited.add(nb)
                path.append(nb)
                stack.append((nb, depth + 1))
                
                # Check if we can close
                if start in adj[nb]:
                    e_close = tuple(sorted((nb, start)))
                    if e_close in remaining:
                        return path[:]
                
                # Backtrack
                path.pop()
                visited.discard(nb)
    
    return None

def _decompose_graph(edges, deadline):
    """Decompose edges into cycles and single edges using structural methods."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    return _hierarchy_decompose(edges_set, deadline)

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    # Evaluator kills at 150s, leave 10s margin
    deadline = time.time() + 140.0
    
    for i, spec in enumerate(specs):
        # Allocate time: ~45s per graph
        graph_deadline = deadline - (len(specs) - 1 - i) * 5.0
        
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
