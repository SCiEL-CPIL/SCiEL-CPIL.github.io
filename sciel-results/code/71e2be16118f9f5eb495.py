import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_with_matching(edges, time_budget=30.0):
        """
        Solve path decomposition using minimum-cost perfect matching.
        
        Key insight: For a graph, the minimum number of edge-disjoint paths that cover
        all edges equals (number of odd-degree vertices) / 2 if we can pair them optimally.
        We use a matching-based approach:
        1. Find all odd-degree vertices
        2. Compute shortest paths between all pairs of odd vertices
        3. Find minimum-cost perfect matching on odd vertices
        4. The matching edges define where to "split" trails
        5. Decompose into paths
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        n = max(max(u, v) for u, v in edges) + 1
        
        # Find odd-degree vertices
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]
        
        if not odd_vertices:
            # All vertices have even degree - Eulerian circuit exists
            # Find Eulerian circuit
            return _find_eulerian_paths(edges, adj, n)
        
        # Compute shortest paths between all pairs of odd vertices
        # Use BFS since all edges have unit weight
        odd_set = set(odd_vertices)
        dist = {}
        parent = {}
        
        for src in odd_vertices:
            # BFS from src
            dist[src] = {src: 0}
            parent[src] = {src: None}
            queue = deque([src])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    e = tuple(sorted((u, v)))
                    if e in edge_set:
                        if v not in dist[src]:
                            dist[src][v] = dist[src][u] + 1
                            parent[src][v] = u
                            queue.append(v)
        
        # Build cost matrix for matching
        m = len(odd_vertices)
        if m == 0:
            return _find_eulerian_paths(edges, adj, n)
        
        # For matching, we need to pair up odd vertices
        # The minimum number of paths = m/2 (if m is even, which it always is)
        # We find minimum-cost perfect matching
        
        # Simple greedy matching with local improvement
        # First, try a simple greedy approach
        remaining = list(odd_vertices)
        pairs = []
        
        # Sort by some heuristic - maybe degree or position
        # For now, use a simple greedy: repeatedly pick the closest pair
        while len(remaining) > 1 and time.time() < start_time + time_budget:
            best_i, best_j = -1, -1
            best_dist = float('inf')
            
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    u, v = remaining[i], remaining[j]
                    if u in dist and v in dist[u]:
                        d = dist[u][v]
                        if d < best_dist:
                            best_dist = d
                            best_i, best_j = i, j
            
            if best_i == -1:
                # No path found between any pair - should not happen in connected graph
                break
            
            u, v = remaining[best_i], remaining[best_j]
            pairs.append((u, v))
            # Remove the two vertices
            remaining.pop(max(best_i, best_j))
            remaining.pop(min(best_i, best_j))
        
        # If any remaining, pair them arbitrarily
        while len(remaining) >= 2:
            pairs.append((remaining[0], remaining[1]))
            remaining.pop(0)
            remaining.pop(0)
        
        # Now we need to construct the paths
        # The matching tells us which odd vertices should be endpoints of the same path
        # We need to find trails that connect these pairs
        
        # Strategy: 
        # 1. Create a multigraph where we add the matched edges
        # 2. Find Eulerian trail/circuit in the augmented graph
        # 3. Split at the matched edges to get the paths
        
        # Actually, a simpler approach:
        # The minimum path cover number is m/2 where m is the number of odd vertices
        # We can find this by:
        # 1. Pairing odd vertices (done above)
        # 2. For each pair, find a shortest path
        # 3. These paths are edge-disjoint if we're careful, but they might not be
        # 
        # Better approach: Use the fact that we can add "virtual" edges between matched
        # odd vertices to make all degrees even, find an Eulerian circuit, then remove
        # the virtual edges to split into paths.
        
        # Build augmented graph
        augmented_edges = list(edges)
        for u, v in pairs:
            # Add virtual edge (we'll track it separately)
            augmented_edges.append((u, v))
        
        # Find Eulerian circuit in augmented graph
        adj_aug = defaultdict(list)
        for u, v in augmented_edges:
            adj_aug[u].append(v)
            adj_aug[v].append(u)
        
        # Hierholzer's algorithm
        def find_eulerian_circuit(adj, n):
            # Build adjacency list with edge indices
            edge_list = list(set((min(u,v), max(u,v)) for u,v in adj.items() for v in adj[u]))
            # Actually, let's use a different approach
            adj_copy = defaultdict(list)
            for u, v in adj.items():
                for w in v:
                    adj_copy[u].append(w)
            
            # Use iterative Hierholzer
            stack = []
            circuit = []
            
            # Start from any vertex with edges
            start = None
            for v in range(n):
                if adj_copy[v]:
                    start = v
                    break
            
            if start is None:
                return []
            
            stack.append(start)
            while stack:
                v = stack[-1]
                if adj_copy[v]:
                    w = adj_copy[v].pop()
                    # Remove the reverse edge too
                    if v in adj_copy[w]:
                        adj_copy[w].remove(v)
                    stack.append(w)
                else:
                    circuit.append(stack.pop())
            
            circuit.reverse()
            return circuit
        
        # Find Eulerian circuit
        circuit = find_eulerian_circuit(adj_aug, n)
        
        if not circuit:
            # Fallback to greedy
            return _greedy_path_decomposition(edges, adj, n)
        
        # Now split the circuit at the virtual edges
        # Virtual edges are the ones in 'pairs'
        virtual_edge_set = set(tuple(sorted((u, v))) for u, v in pairs)
        
        # Extract paths by splitting at virtual edges
        paths = []
        current_path = []
        
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            e = tuple(sorted((u, v)))
            
            if e in virtual_edge_set:
                # This is a virtual edge - end current path and start new one
                if current_path:
                    paths.append(current_path)
                    current_path = [v]
            else:
                # Real edge - continue path
                if not current_path:
                    current_path = [u]
                current_path.append(v)
        
        if current_path:
            paths.append(current_path)
        
        # Remove any paths with only 1 vertex (shouldn't happen)
        paths = [p for p in paths if len(p) >= 2]
        
        # Verify and fix if needed
        if not _verify_paths(paths, edge_set):
            # Fallback to greedy
            return _greedy_path_decomposition(edges, adj, n)
        
        return paths
    
    def _find_eulerian_paths(edges, adj, n):
        """Find Eulerian circuit and return as a single path."""
        # Use Hierholzer's algorithm
        adj_copy = defaultdict(list)
        for u, v in edges:
            adj_copy[u].append(v)
            adj_copy[v].append(u)
        
        # Find start vertex
        start = None
        for v in range(n):
            if adj_copy[v]:
                start = v
                break
        
        if start is None:
            return []
        
        # Iterative Hierholzer
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                w = adj_copy[v].pop()
                if v in adj_copy[w]:
                    adj_copy[w].remove(v)
                stack.append(w)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        if len(circuit) < 2:
            return []
        
        return [circuit]
    
    def _greedy_path_decomposition(edges, adj, n):
        """Fallback greedy path decomposition."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        available = set(edge_set)
        paths = []
        
        while available:
            # Find a long path greedily
            start = next(iter(available))[0]
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = []
                for v in adj[current]:
                    e = tuple(sorted((current, v)))
                    if e in available and v not in visited:
                        neighbors.append(v)
                
                if not neighbors:
                    break
                
                # Pick the neighbor that leads to the longest path (greedy)
                best_next = neighbors[0]
                best_len = 0
                for nxt in neighbors:
                    # Count available edges from nxt
                    cnt = sum(1 for w in adj[nxt] 
                             if tuple(sorted((nxt, w))) in available and w not in visited)
                    if cnt > best_len:
                        best_len = cnt
                        best_next = nxt
                
                visited.add(best_next)
                path.append(best_next)
                current = best_next
                
                # Remove the edge
                e = tuple(sorted((path[-2], path[-1])))
                available.discard(e)
            
            if len(path) >= 2:
                paths.append(path)
        
        return paths
    
    def _verify_paths(paths, edge_set):
        """Verify that paths are valid: simple, use graph edges, edge-disjoint, cover all edges."""
        used_edges = set()
        for path in paths:
            # Check simple
            if len(set(path)) != len(path):
                return False
            
            # Check edges
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check coverage
        return used_edges == edge_set
    
    def solve_graph(edges, seed, time_budget=30.0):
        if not edges:
            return []
        
        # Try matching-based approach first
        if time.time() < start_time + time_budget:
            paths = solve_with_matching(edges, time_budget)
            if paths and _verify_paths(paths, set(tuple(sorted(e)) for e in edges)):
                return paths
        
        # Fallback to greedy
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        n = max(max(u, v) for u, v in edges) + 1
        
        return _greedy_path_decomposition(edges, adj, n)

    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline:
            # Fallback
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = solve_graph(edges, spec[2])
        results.append(paths)
        
    return results
