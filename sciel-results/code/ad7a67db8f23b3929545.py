import time
import random
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges, n):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def bfs_distances(adj, source, n):
        """BFS from source to get shortest distances to all vertices."""
        dist = [-1] * n
        dist[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def find_shortest_path(adj, source, target, n):
        """Find shortest path from source to target using BFS."""
        if source == target:
            return [source]
        prev = [-1] * n
        visited = [False] * n
        visited[source] = True
        queue = deque([source])
        while queue:
            u = queue.popleft()
            if u == target:
                break
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    prev[v] = u
                    queue.append(v)
        # Reconstruct path
        path = []
        curr = target
        while curr != -1:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        if path[0] != source:
            return None  # No path found (shouldn't happen in connected graph)
        return path

    def find_eulerian_circuit(multigraph, n):
        """
        Find an Eulerian circuit in a connected multigraph where all vertices have even degree.
        multigraph: dict mapping edge (u,v) to count
        Returns the circuit as a list of vertices.
        """
        # Build adjacency with edge counts
        adj = defaultdict(list)
        for (u, v), count in multigraph.items():
            for _ in range(count):
                adj[u].append(v)
                adj[v].append(u)
        
        # Check if all vertices have even degree
        for u in adj:
            if len(adj[u]) % 2 != 0:
                # Not Eulerian - shouldn't happen
                return None
        
        # Find a vertex with non-zero degree
        start = None
        for u in adj:
            if adj[u]:
                start = u
                break
        if start is None:
            return [0]  # Empty graph
        
        # Hierholzer's algorithm
        circuit = []
        stack = [start]
        while stack:
            u = stack[-1]
            if adj[u]:
                v = adj[u].pop()
                # Remove the reverse edge
                if v in adj[u]:
                    idx = adj[u].index(v)
                    adj[u].pop(idx)
                else:
                    # Edge (u,v) was self-loop, already handled
                    pass
                stack.append(v)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def solve_graph(edges, n, seed):
        if not edges:
            return []
        
        if time.time() > deadline - 5:
            return [[u, v] for u, v in edges]
        
        adj = build_adjacency(edges, n)
        
        # Find odd-degree vertices
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]
        k = len(odd_vertices) // 2
        
        if k == 0:
            # All vertices have even degree - Eulerian circuit
            # Build multigraph
            multigraph = {}
            for u, v in edges:
                key = tuple(sorted((u, v)))
                multigraph[key] = multigraph.get(key, 0) + 1
            
            circuit = find_eulerian_circuit(multigraph, n)
            if circuit and len(circuit) > 1:
                # The circuit is a single path (closed)
                # For path decomposition, we can return it as one path
                # But a circuit is not a simple path in the usual sense
                # Actually, for path decomposition, we need simple paths
                # An Eulerian circuit visits vertices multiple times
                # So we need to decompose differently
                
                # Actually, if all degrees are even, we can have 0 or more paths
                # The minimum is 0 if we allow circuits, but paths must be simple
                # Let's just return the edges as individual paths for safety
                # Or better: find a spanning tree of the Eulerian graph
                # For now, return individual edges
                return [[u, v] for u, v in edges]
        
        # Compute all-pairs shortest paths between odd vertices
        # For small k (<= 15 for n=30), this is feasible
        # We need distances between all pairs of odd vertices
        
        odd_set = set(odd_vertices)
        
        # Compute distances from each odd vertex
        dist_from_odd = {}
        for ov in odd_vertices:
            dists = bfs_distances(adj, ov, n)
            dist_from_odd[ov] = dists
        
        # Build cost matrix for minimum weight perfect matching
        if k <= 1:
            # Only one pair
            u, v = odd_vertices[0], odd_vertices[1]
            path = find_shortest_path(adj, u, v, n)
            if path is None:
                path = [u, v]  # Fallback
            return [path]
        
        # Cost matrix: k x k where cost[i][j] = distance between odd_vertices[i] and odd_vertices[j]
        # But we need a perfect matching, so we use a 2k x 2k matrix with the standard trick
        # Actually, for minimum weight perfect matching on 2k vertices, we can use
        # linear_sum_assignment on a 2k x 2k matrix where we pair up the vertices
        
        # Standard approach: create a 2k x 2k cost matrix
        # The matching will pair vertex i with some vertex j
        # But linear_sum_assignment finds a permutation, not a matching
        # We need to use the trick: create a matrix where cost[i][j] = dist(odd[i], odd[j])
        # and find minimum weight perfect matching
        
        # Actually, linear_sum_assignment can be used for minimum weight perfect matching
        # by setting up the cost matrix appropriately
        # For a perfect matching on 2k vertices, we create a 2k x 2k matrix
        # where M[i][j] = dist(odd[i], odd[j]) if i != j, and M[i][i] = infinity
        # Then find the minimum weight assignment that pairs up all vertices
        
        # But linear_sum_assignment finds a permutation, which is not the same as a matching
        # We need to use the fact that a perfect matching can be found by solving
        # a minimum weight perfect matching problem
        
        # Alternative: use the fact that for 2k vertices, we can create a 2k x 2k matrix
        # and use linear_sum_assignment, but we need to ensure the result is a matching
        # (i.e., no vertex is matched to itself, and the matching is symmetric)
        
        # Better approach: use scipy's minimum weight perfect matching
        # But that's not in standard scipy
        # Let's use a simple approach: since k is small (<= 15), we can use
        # dynamic programming or a simple greedy with local search
        
        # For now, let's use a simple approach:
        # 1. Compute all-pairs distances
        # 2. Use a greedy matching with local search (2-opt)
        
        # Build distance matrix
        m = len(odd_vertices)  # 2k
        dist_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    dist_matrix[i][j] = 0
                else:
                    d = dist_from_odd[odd_vertices[i]][odd_vertices[j]]
                    if d == -1:
                        d = n  # Shouldn't happen in connected graph
                    dist_matrix[i][j] = d
        
        # Find minimum weight perfect matching using a simple approach
        # Since m <= 30, we can use a bitmask DP for small m, or greedy for larger
        
        if m <= 20:
            # Bitmask DP
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(mask):
                # mask is a bitmask of which vertices are still unmatched
                if mask == 0:
                    return 0
                # Find the first unmatched vertex
                first = -1
                for i in range(m):
                    if mask & (1 << i):
                        first = i
                        break
                if first == -1:
                    return 0
                best = float('inf')
                for j in range(m):
                    if j != first and (mask & (1 << j)):
                        new_mask = mask & ~(1 << first) & ~(1 << j)
                        cost = dist_matrix[first][j] + dp(new_mask)
                        if cost < best:
                            best = cost
                return best
            
            total_cost = dp((1 << m) - 1)
            
            # Now reconstruct the matching
            matching = {}
            mask = (1 << m) - 1
            while mask:
                first = -1
                for i in range(m):
                    if mask & (1 << i):
                        first = i
                        break
                if first == -1:
                    break
                best_cost = float('inf')
                best_j = -1
                for j in range(m):
                    if j != first and (mask & (1 << j)):
                        new_mask = mask & ~(1 << first) & ~(1 << j)
                        cost = dist_matrix[first][j] + dp(new_mask)
                        if cost < best_cost:
                            best_cost = cost
                            best_j = j
                if best_j == -1:
                    break
                matching[odd_vertices[first]] = odd_vertices[best_j]
                matching[odd_vertices[best_j]] = odd_vertices[first]
                mask &= ~(1 << first) & ~(1 << best_j)
        else:
            # Greedy matching with local search
            # Start with a greedy matching
            remaining = list(range(m))
            matching = {}
            while remaining:
                i = remaining[0]
                remaining.pop(0)
                # Find closest remaining vertex
                best_j = -1
                best_dist = float('inf')
                for j in remaining:
                    if dist_matrix[i][j] < best_dist:
                        best_dist = dist_matrix[i][j]
                        best_j = j
                if best_j == -1:
                    break
                remaining.remove(best_j)
                matching[odd_vertices[i]] = odd_vertices[best_j]
                matching[odd_vertices[best_j]] = odd_vertices[i]
            
            # Local search: try to improve by swapping
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Get current pairs
                pairs = []
                used = set()
                for i in range(m):
                    if i not in used:
                        j = matching.get(odd_vertices[i], -1)
                        if j != -1 and j != odd_vertices[i]:
                            j_idx = odd_vertices.index(j)
                            pairs.append((i, j_idx))
                            used.add(i)
                            used.add(j_idx)
                
                # Try all pairs of pairs
                for a in range(len(pairs)):
                    for b in range(a + 1, len(pairs)):
                        i1, j1 = pairs[a]
                        i2, j2 = pairs[b]
                        # Current cost
                        cur_cost = dist_matrix[i1][j1] + dist_matrix[i2][j2]
                        # New cost
                        new_cost = dist_matrix[i1][i2] + dist_matrix[j1][j2]
                        if new_cost < cur_cost:
                            # Swap
                            matching[odd_vertices[i1]] = odd_vertices[i2]
                            matching[odd_vertices[i2]] = odd_vertices[i1]
                            matching[odd_vertices[j1]] = odd_vertices[j2]
                            matching[odd_vertices[j2]] = odd_vertices[j1]
                            pairs[a] = (i1, i2)
                            pairs[b] = (j1, j2)
                            improved = True
        
        # Now we have a matching. For each pair, find the shortest path.
        # Add these paths to the graph to make all degrees even.
        # Then find an Eulerian circuit.
        # Then remove the added edges to get the path decomposition.
        
        # Build the augmented multigraph
        edge_count = defaultdict(int)
        for u, v in edges:
            key = tuple(sorted((u, v)))
            edge_count[key] += 1
        
        # Add the matched paths
        added_paths = []
        for u, v in matching.items():
            if u < v:  # Avoid double counting
                path = find_shortest_path(adj, u, v, n)
                if path is None:
                    path = [u, v]
                added_paths.append(path)
                for i in range(len(path) - 1):
                    key = tuple(sorted((path[i], path[i + 1])))
                    edge_count[key] += 1
        
        # Now find Eulerian circuit in the augmented graph
        # Build adjacency from edge_count
        adj_aug = defaultdict(list)
        for (u, v), count in edge_count.items():
            for _ in range(count):
                adj_aug[u].append(v)
                adj_aug[v].append(u)
        
        # Find Eulerian circuit
        start = None
        for u in adj_aug:
            if adj_aug[u]:
                start = u
                break
        if start is None:
            return [[u, v] for u, v in edges]
        
        circuit = []
        stack = [start]
        while stack:
            u = stack[-1]
            if adj_aug[u]:
                v = adj_aug[u].pop()
                # Remove reverse edge
                if v in adj_aug[u]:
                    idx = adj_aug[u].index(v)
                    adj_aug[u].pop(idx)
                stack.append(v)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # Now we need to extract the paths by removing the added edges
        # The circuit is a closed walk. The added edges split it into k paths.
        # Mark the added edges
        added_edges = set()
        for path in added_paths:
            for i in range(len(path) - 1):
                key = tuple(sorted((path[i], path[i + 1])))
                added_edges.add(key)
        
        # Traverse the circuit and split at added edges
        paths = []
        current_path = [circuit[0]]
        
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            key = tuple(sorted((u, v)))
            current_path.append(v)
            if key in added_edges:
                # Check if this specific edge instance is added
                # We need to track which instances are added
                # For simplicity, let's just check if the edge is in added_edges
                # and decrement a counter
                # Actually, this approach is flawed because we might have multiple instances
                # Let's use a different approach: track edge usage
                
                # For now, let's just split at added edges
                # This might not be perfect but should work in most cases
                pass
        
        # This approach is getting complicated. Let's use a simpler fallback:
        # Just return the greedy path decomposition
        
        # Actually, let's try a different approach:
        # After finding the Eulerian circuit, we can identify the paths by
        # looking at where the added edges are
        
        # Let me use a cleaner approach:
        # 1. Find the Eulerian circuit
        # 2. Walk through the circuit
        # 3. When we traverse an added edge, that's where we split
        
        # But we need to track which specific edge instances are added
        # Let's use a multiset approach
        
        # For now, let's just return the greedy decomposition as a fallback
        # The matching approach gives us the optimal NUMBER of paths, but
        # extracting them correctly is complex
        
        # Let's use a simpler approach: greedy path decomposition
        # but with the knowledge that we need exactly k paths
        
        # Actually, let's try to properly extract the paths
        
        # Rebuild edge_count with tracking
        edge_instances = defaultdict(list)  # key -> list of (path_idx, edge_idx)
        for u, v in edges:
            key = tuple(sorted((u, v)))
            edge_instances[key].append(('orig', None))
        
        for pi, path in enumerate(added_paths):
            for ei in range(len(path) - 1):
                key = tuple(sorted((path[ei], path[ei + 1])))
                edge_instances[key].append(('added', (pi, ei)))
        
        # Now walk the circuit and track which edge instance we're using
        paths = []
        current_path = [circuit[0]]
        edge_used = defaultdict(int)  # key -> number of times used so far
        
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            key = tuple(sorted((u, v)))
            current_path.append(v)
            
            # Find which instance of this edge we're using
            instances = edge_instances[key]
            used = edge_used[key]
            if used < len(instances):
                inst_type, inst_info = instances[used]
                edge_used[key] += 1
                if inst_type == 'added':
                    # This is an added edge, so we split here
                    if len(current_path) > 1:
                        paths.append(current_path[:])
                    current_path = [v]
        
        if len(current_path) > 1:
            paths.append(current_path)
        
        # Verify we have the right number of paths
        if len(paths) != k:
            # Fallback to greedy
            paths = greedy_path_decomposition(edges, adj, n, seed)
        
        return paths

    def greedy_path_decomposition(edges, adj, n, seed):
        """Greedy path decomposition as a fallback."""
        rng = random.Random(seed + 100)
        available = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while available:
            if time.time() > deadline - 5:
                break
            
            # Find a greedy path
            degrees = defaultdict(int)
            for u, v in available:
                degrees[u] += 1
                degrees[v] += 1
            
            if not degrees:
                break
            
            start_candidates = list(degrees.keys())
            rng.shuffle(start_candidates)
            
            best_path = []
            for start in start_candidates[:3]:
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
                    
                    next_node = rng.choice(neighbors)
                    visited.add(next_node)
                    path.append(next_node)
                    current = next_node
                
                if len(path) > len(best_path):
                    best_path = path
            
            if not best_path:
                u, v = next(iter(available))
                best_path = [u, v]
            
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i + 1])))
                if e in available:
                    available.remove(e)
            
            paths.append(best_path)
        
        if available:
            for u, v in available:
                paths.append([u, v])
        
        return paths

    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline - 10:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        if time.time() > deadline - 5:
            results.append([[u, v] for u, v in edges])
            continue
        
        try:
            paths = solve_graph(edges, n, seed)
        except Exception as e:
            # Fallback to greedy
            adj = build_adjacency(edges, n)
            paths = greedy_path_decomposition(edges, adj, n, seed)
        
        results.append(paths)
    
    return results

