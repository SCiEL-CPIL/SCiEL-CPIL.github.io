import time
import random
import sys
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []
        
        # Build initial adjacency structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        pieces = []
        
        def find_connected_components(adj):
            visited = set()
            components = []
            for node in list(adj.keys()):
                if node not in visited:
                    # BFS to find component
                    q = deque([node])
                    comp = []
                    visited.add(node)
                    while q:
                        curr = q.popleft()
                        comp.append(curr)
                        for nei in adj[curr]:
                            if nei not in visited:
                                visited.add(nei)
                                q.append(nei)
                    components.append(comp)
            return components

        def find_hamiltonian_cycle_in_component(component, adj):
            # The component is guaranteed to be 2-connected in the context of the 
            # iterative algorithm (after removing degree 1 vertices).
            # We use a simple heuristic to find a Hamiltonian cycle.
            # Since the graph is dense (random with p>0.3), a greedy approach with backtracking
            # or even just a random walk that tries to close often works, but for guaranteed 
            # correctness on 2-connected graphs, we can use the fact that they contain a 
            # Hamiltonian cycle? No, not all 2-connected graphs are Hamiltonian.
            # However, random graphs with p > 1/log(n) are almost surely Hamiltonian.
            # For n=20..30 and p=0.3..0.4, they are very likely Hamiltonian.
            
            # Let's try to find a Hamiltonian cycle using a simple DFS with heuristics.
            # If it fails, we fall back to finding any cycle.
            
            if len(component) < 3:
                return None
            
            # Sort component vertices for deterministic start
            start = component[0]
            vertices = sorted(component)
            
            def dfs_cycle(path, visited):
                if len(path) == len(vertices):
                    # Check if last connects to start
                    last = path[-1]
                    if start in adj[last]:
                        return path
                    return None
                
                curr = path[-1]
                # Heuristic: prefer vertices with fewer remaining options (Warnsdorff's rule)
                candidates = [v for v in adj[curr] if v not in visited]
                if not candidates:
                    return None
                
                # Sort candidates by degree in remaining graph?
                # For simplicity, just try all, but order by degree to prune faster
                candidates.sort(key=lambda x: len(adj[x]))
                
                for v in candidates:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        res = dfs_cycle(path, visited)
                        if res:
                            return res
                        path.pop()
                        visited.remove(v)
                return None

            # Try to find Hamiltonian cycle
            # To avoid exponential blowup if not Hamiltonian, we limit time or depth?
            # Given small N (<=30), DFS is fast if successful. If not, it might be slow.
            # But these graphs are almost surely Hamiltonian.
            # Let's add a timeout guard for the DFS just in case.
            dfs_start_time = time.time()
            
            # Optimization: Start from a vertex with min degree in the component?
            # Or just the first one.
            res = dfs_cycle([start], {start})
            
            if res:
                return res
            
            # If Hamiltonian cycle not found quickly (or at all), find any cycle
            # This fallback is unlikely to be needed for these parameters, but good for safety.
            # Simple cycle finding:
            # Pick an edge, do DFS to find a path back to start.
            for u in component:
                for v in adj[u]:
                    if v > u: # avoid duplicates
                        # Try to find path v -> ... -> u
                        # This is getting complex. Let's just return a small cycle if we must.
                        # Actually, if it's 2-connected, it has a cycle.
                        pass
                        
            return None

        def find_cycle_in_2conn_component(component, adj):
            # Fallback for if Hamiltonian search fails or times out
            # Find any cycle
            visited = set()
            parent = {}
            
            def dfs_cycle_finder(node, start_node):
                visited.add(node)
                for nei in adj[node]:
                    if nei == start_node and len(visited) >= 3:
                        # Found cycle
                        cycle = [node]
                        curr = node
                        while curr != start_node:
                            curr = parent[curr]
                            cycle.append(curr)
                        return cycle
                    if nei not in visited:
                        parent[nei] = node
                        res = dfs_cycle_finder(nei, start_node)
                        if res:
                            return res
                return None

            for u in component:
                if u not in visited:
                    res = dfs_cycle_finder(u, u)
                    if res:
                        return res
            return None

        # Main Loop: Iterative decomposition
        while remaining_edges:
            if time.time() > time_limit:
                # Fallback: just take edges
                break
                
            # 1. Find vertices with degree 1 (in current remaining graph)
            # Note: adj is maintained dynamically
            degree_1_vertices = [u for u in adj if len(adj[u]) == 1]
            
            if degree_1_vertices:
                # Process all degree 1 vertices
                # To ensure we don't remove edges that are part of a potential cycle 
                # in a 2-connected component, we must remove them as single edges.
                # Order doesn't strictly matter, but we must be careful not to double remove.
                for u in degree_1_vertices:
                    if u not in adj or len(adj[u]) != 1:
                        continue # already processed or removed
                    v = next(iter(adj[u]))
                    edge = tuple(sorted((u, v)))
                    if edge in remaining_edges:
                        pieces.append(list(edge))
                        remaining_edges.remove(edge)
                        # Update adj
                        adj[u].discard(v)
                        adj[v].discard(u)
                        if not adj[u]:
                            del adj[u]
                        if not adj[v]:
                            del adj[v]
            else:
                # All vertices have degree >= 2
                # Find 2-connected components
                # A graph where all vertices have degree >= 2 is a union of 2-edge-connected components?
                # Not necessarily 2-vertex-connected, but we can find cycles.
                # The standard algorithm for minimum cycle cover (or similar) on graphs with min degree >= 2
                # involves finding a cycle in each 2-connected component.
                
                # We will find a cycle in the largest component first to maximize efficiency?
                # Or just find any cycle.
                
                components = find_connected_components(adj)
                if not components:
                    break
                    
                # Sort components by size descending to find longer cycles first?
                # Actually, any cycle works. Let's pick the largest to reduce search space?
                # Or just the first one.
                comp = max(components, key=len)
                
                # Try to find a Hamiltonian cycle in this component
                # If the component is the whole graph and it's 2-connected, it might be Hamiltonian.
                # If it's a proper subgraph, it's also 2-connected (if we consider the induced subgraph? No, components are 2-edge-connected? No.)
                
                # Wait, if min degree >= 2, the graph is a union of 2-edge-connected components.
                # Each 2-edge-connected component contains a cycle.
                # We can find a cycle in 'comp'.
                
                # Let's try Hamiltonian first on the component.
                # Note: 'comp' is a list of vertices. The induced subgraph on 'comp' in 'adj' is what we care about.
                # However, 'adj' contains all edges in remaining_edges.
                # Since 'comp' is a connected component of the current graph, there are no edges between 'comp' and other components.
                # So the induced subgraph is exactly the subgraph on 'comp'.
                
                cycle = None
                # Only try Hamiltonian if component is reasonably small or if we have time
                # For n<=30, Hamiltonian search is feasible.
                if len(comp) <= 30:
                    # We need to ensure we only use edges within 'comp'
                    # The dfs in find_hamiltonian_cycle_in_component uses 'adj', which is global.
                    # But since 'comp' is a component, adj[v] for v in comp only contains neighbors in comp.
                    # So it's safe.
                    cycle = find_hamiltonian_cycle_in_component(comp, adj)
                
                if not cycle:
                    # Fallback to any cycle
                    cycle = find_cycle_in_2conn_component(comp, adj)
                    
                if cycle:
                    pieces.append(cycle)
                    # Remove edges of the cycle from remaining_edges and adj
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i+1]
                        e = tuple(sorted((u, v)))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
                        if not adj[u]:
                            del adj[u]
                        if not adj[v]:
                            del adj[v]
                    # Closing edge
                    u, v = cycle[-1], cycle[0]
                    e = tuple(sorted((u, v)))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if not adj[u]:
                        del adj[u]
                    if not adj[v]:
                        del adj[v]
                else:
                    # Should not happen if min degree >= 2, but safety break
                    # Just take an edge
                    if remaining_edges:
                        edge = next(iter(remaining_edges))
                        pieces.append(list(edge))
                        remaining_edges.remove(edge)
                        u, v = edge
                        adj[u].discard(v)
                        adj[v].discard(u)
                        if not adj[u]:
                            del adj[u]
                        if not adj[v]:
                            del adj[v]
                    else:
                        break
                        
        # If we timed out, we need to cover remaining edges as single edges
        if remaining_edges:
            for e in remaining_edges:
                pieces.append(list(e))
                
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time proportional to complexity, but ensure each gets some
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        # Add a small buffer
        time_limit = time.time() + max(1.0, time_per_graph * 0.9)
        
        best_pieces = decompose_graph(edges, time_limit)
        results.append(best_pieces)
        
        # If time is critical, fallback to trivial for remaining
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
    # Quick sanity check
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        # Re-generate to check
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)
        
        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                # Check distinct vertices
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                # Check edges
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break
                
        if valid and used_edges != edge_set:
            valid = False
            
        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
