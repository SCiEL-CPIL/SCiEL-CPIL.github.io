"""Combined approach: iterative cycle extraction with multi-strategy search and local refinement."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def _find_cycle_dfs(start, adj, remaining, rng, max_depth=40):
        """DFS to find a cycle back to start."""
        path = [start]
        visited = {start}
        
        def dfs(current, depth):
            if depth > max_depth:
                return None
            neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
            if not neighbors:
                return None
            # Try to close cycle
            if start in neighbors and len(path) >= 3:
                return path[:]
            unvisited = [nb for nb in neighbors if nb not in visited]
            if not unvisited:
                return None
            rng.shuffle(unvisited)
            for nb in unvisited:
                visited.add(nb)
                path.append(nb)
                result = dfs(nb, depth + 1)
                if result:
                    return result
                path.pop()
                visited.discard(nb)
            return None
        
        return dfs(start, 0)
    
    def _find_cycle_greedy(adj, remaining, rng):
        """Greedy cycle finding: start from high-degree vertex, extend greedily."""
        degs = {}
        for u in adj:
            degs[u] = len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining])
        
        if not degs:
            return None
        
        start_candidates = sorted(degs.keys(), key=lambda u: (-degs[u], rng.random()))
        best_cycle = None
        
        for start in start_candidates[:min(5, len(start_candidates))]:
            if degs[start] < 2:
                continue
            
            for _ in range(5):
                path = [start]
                visited = {start}
                current = start
                found = False
                
                for _ in range(50):
                    neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining]
                    if not neighbors:
                        break
                    unvisited = [nb for nb in neighbors if nb not in visited]
                    
                    if unvisited:
                        # Prefer higher-degree neighbors
                        unvisited_sorted = sorted(unvisited, key=lambda nb: (-degs.get(nb, 0), rng.random()))
                        next_node = unvisited_sorted[0]
                        visited.add(next_node)
                        path.append(next_node)
                        current = next_node
                    elif start in neighbors and len(path) >= 3:
                        found = True
                        break
                    else:
                        break
                
                if found and len(path) >= 3:
                    if best_cycle is None or len(path) > len(best_cycle):
                        best_cycle = path
                    if len(path) >= 10:
                        break
        
        return best_cycle
    
    def _find_any_cycle(adj, remaining):
        """Find any cycle using DFS."""
        visited = set()
        parent = {}
        
        def dfs(u):
            visited.add(u)
            for v in adj[u]:
                if (min(u, v), max(u, v)) not in remaining:
                    continue
                if v == u:
                    continue
                if v not in visited:
                    parent[v] = u
                    result = dfs(v)
                    if result:
                        return result
                elif v in visited and v != parent.get(u, -1) and u != parent.get(v, -1):
                    # Found a back edge, reconstruct cycle
                    cycle = [u, v]
                    cur = u
                    while cur != v:
                        cur = parent.get(cur)
                        if cur is None or cur == -1:
                            break
                        cycle.append(cur)
                    if len(cycle) >= 3:
                        return cycle
            return None
        
        for u in range(max(adj.keys()) + 1 if adj else 0):
            if u not in visited and (u in adj and len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining]) >= 1):
                parent[u] = -1
                result = dfs(u)
                if result:
                    return result
        return None
    
    def _local_search_merge(pieces, edge_set, adj, rng, iterations=50):
        """Try to merge single edges into cycles by finding augmenting paths."""
        pieces = [p[:] for p in pieces]
        single_edges = [(i, p) for i, p in enumerate(pieces) if len(p) == 2]
        
        for _ in range(iterations):
            if not single_edges:
                break
            
            # Pick a random single edge
            idx, edge = rng.choice(single_edges)
            u, v = edge[0], edge[1]
            
            # Try to find a path from u to v in the remaining graph that doesn't use this edge
            # and forms a cycle when combined
            # This is complex; simpler: try to replace 2+ single edges with a cycle
            # Find two single edges that share a vertex and form a path of length 2
            # Then check if the endpoints are connected
            
            # Simpler approach: try to find a 3-cycle involving one of the single edges
            found_merge = False
            for j, ej in single_edges:
                if j == idx:
                    continue
                uj, vj = ej[0], ej[1]
                # Check if we can form a cycle: u-v-ej-u or u-ej-v-u etc.
                # Cycle: u, v, uj, u (if v-uj and uj-u exist)
                e1 = tuple(sorted((v, uj)))
                e2 = tuple(sorted((uj, u)))
                if e1 in edge_set and e2 in edge_set:
                    # We can merge edge (u,v) and edge (uj,vj) into cycle u-v-uj-u if vj==uj... 
                    # Actually need to be more careful
                    pass
            
            # Alternative: just try to find any 3-cycle in the remaining edges
            # and replace 3 single edges with it
            remaining_edges = set()
            for i, p in enumerate(pieces):
                if len(p) == 2:
                    remaining_edges.add(tuple(sorted(p)))
            
            # Find a triangle in remaining edges
            for e in list(remaining_edges):
                a, b = e
                for c in range(len(adj)):
                    if c == a or c == b:
                        continue
                    e2 = tuple(sorted((a, c)))
                    e3 = tuple(sorted((b, c)))
                    if e2 in remaining_edges and e3 in remaining_edges:
                        # Found triangle a-b-c-a
                        # Find the indices of these three edges
                        indices = []
                        for i, p in enumerate(pieces):
                            if len(p) == 2:
                                pe = tuple(sorted(p))
                                if pe in (e, e2, e3):
                                    indices.append(i)
                        if len(indices) == 3:
                            # Replace these three single edges with the cycle
                            new_cycle = [a, b, c]
                            for i in sorted(indices, reverse=True):
                                del pieces[i]
                            pieces.append(new_cycle)
                            found_merge = True
                            break
                if found_merge:
                    break
            
            if not found_merge:
                break
        
        # Rebuild single_edges list after modifications
        return pieces
    
    def decompose_graph(edges, n, time_budget=40):
        """Decompose graph into minimum pieces using iterative improvement."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        best_pieces = None
        best_count = len(edge_set) + 1
        start_time = time.time()
        
        max_restarts = 100
        rng = random.Random(42)
        
        for restart in range(max_restarts):
            if time.time() > start_time + time_budget - 2:
                break
            
            # Build adjacency
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining = set(edge_set)
            pieces = []
            
            # Iteratively extract cycles
            while remaining:
                if time.time() > start_time + time_budget - 1:
                    break
                
                cycle = None
                
                # Strategy 1: Greedy cycle (fast, good for dense graphs)
                c1 = _find_cycle_greedy(adj, remaining, rng)
                if c1 and len(c1) >= 3:
                    cycle = c1
                
                # Strategy 2: Randomized DFS from high-degree vertices
                if not cycle or len(cycle) < 5:
                    degs = {}
                    for u in adj:
                        degs[u] = len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining])
                    
                    if degs:
                        verts = sorted(degs.keys(), key=lambda u: (-degs[u], rng.random()))
                        for start in verts[:min(6, len(verts))]:
                            if degs.get(start, 0) < 2:
                                continue
                            for _ in range(3):
                                c2 = _find_cycle_dfs(start, adj, remaining, rng, max_depth=30)
                                if c2 and len(c2) >= 3:
                                    if cycle is None or len(c2) > len(cycle):
                                        cycle = c2
                                    if len(c2) >= 8:
                                        break
                            if cycle and len(cycle) >= 8:
                                break
                
                # Strategy 3: Fallback - find any cycle
                if not cycle:
                    c3 = _find_any_cycle(adj, remaining)
                    if c3 and len(c3) >= 3:
                        cycle = c3
                
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        if e in remaining:
                            remaining.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
                    pieces.append(cycle)
                else:
                    # No cycle; take a single edge
                    e = tuple(sorted(rng.choice(list(remaining))))
                    remaining.discard(e)
                    adj[e[0]].discard(e[1])
                    adj[e[1]].discard(e[0])
                    pieces.append(list(e))
            
            # Local search: try to merge single edges into cycles
            pieces = _local_search_merge(pieces, edge_set, adj, rng, iterations=30)
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
                
                if best_count <= 5:
                    break
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
        
        return best_pieces
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline - 5:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        pieces = decompose_graph(edges, n, time_budget=40)
        results.append(pieces)
    
    return results

# EVOLVE-BLOCK-END
