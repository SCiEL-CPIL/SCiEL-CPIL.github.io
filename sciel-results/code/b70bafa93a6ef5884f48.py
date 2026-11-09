import time
import random
from collections import defaultdict, deque
import numpy as np

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
            return None  # No path found
        return path

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
            # All vertices have even degree.
            # We can just find an Eulerian circuit and split it into simple paths.
            # But we need to handle the "added edges" logic uniformly.
            # If k=0, no added edges. The whole circuit is one trail.
            # We will use the same logic: added_paths = [].
            pass
        
        # Compute all-pairs shortest paths between odd vertices
        if k > 0:
            odd_set = set(odd_vertices)
            
            # Compute distances from each odd vertex
            dist_from_odd = {}
            for ov in odd_vertices:
                dists = bfs_distances(adj, ov, n)
                dist_from_odd[ov] = dists
            
            m = len(odd_vertices)  # 2k
            dist_matrix = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    if i == j:
                        dist_matrix[i][j] = 0
                    else:
                        d = dist_from_odd[odd_vertices[i]][odd_vertices[j]]
                        if d == -1:
                            d = n  # Shouldn't happen
                        dist_matrix[i][j] = d
            
            # Find minimum weight perfect matching
            matching = {}
            if m <= 20:
                # Bitmask DP
                from functools import lru_cache
                
                @lru_cache(maxsize=None)
                def dp(mask):
                    if mask == 0:
                        return 0
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
                
                # Reconstruct matching
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
                # Greedy matching for larger m (unlikely for n<=30)
                remaining = list(range(m))
                while remaining:
                    i = remaining[0]
                    remaining.pop(0)
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
            
            # Add matched paths
            added_paths = []
            for u, v in matching.items():
                if u < v:
                    path = find_shortest_path(adj, u, v, n)
                    if path is None:
                        path = [u, v]
                    added_paths.append(path)
        else:
            added_paths = []

        # Build augmented multigraph with unique edge IDs
        # edge_id: (u, v, is_added, path_idx, edge_idx_in_path)
        # We need to track which edges are added to split the circuit.
        
        edge_list = [] # List of (u, v, is_added)
        adj_aug = defaultdict(list) # u -> list of (v, edge_id)
        
        # Add original edges
        for u, v in edges:
            eid = len(edge_list)
            edge_list.append((u, v, False))
            adj_aug[u].append((v, eid))
            adj_aug[v].append((u, eid))
        
        # Add added edges
        for pi, path in enumerate(added_paths):
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                eid = len(edge_list)
                edge_list.append((u, v, True))
                adj_aug[u].append((v, eid))
                adj_aug[v].append((u, eid))
        
        # Find Eulerian Circuit
        start = None
        for u in adj_aug:
            if adj_aug[u]:
                start = u
                break
        if start is None:
            return []
        
        circuit_edges = [] # List of edge IDs in order
        circuit_vertices = [start]
        
        stack = [(start, None)] # (vertex, incoming_edge_id)
        while stack:
            u, incoming_eid = stack[-1]
            if adj_aug[u]:
                v, eid = adj_aug[u].pop()
                # Remove reverse edge
                # Find the reverse edge in adj_aug[v]
                # We need to remove the specific instance.
                # Since we stored (v, eid) in adj_aug[u], we need to remove (u, eid) from adj_aug[v].
                # This is O(deg(v)).
                for i, (neighbor, neid) in enumerate(adj_aug[v]):
                    if neid == eid:
                        adj_aug[v].pop(i)
                        break
                stack.append((v, eid))
                circuit_edges.append(eid)
                circuit_vertices.append(v)
            else:
                stack.pop()
        
        # circuit_edges contains the sequence of edge IDs.
        # circuit_vertices contains the sequence of vertices.
        # Note: The circuit is closed, so circuit_vertices[0] == circuit_vertices[-1].
        
        # Split the circuit into trails based on "added" edges.
        # An "added" edge indicates a break.
        # The segments between added edges are the trails.
        
        trails = []
        current_trail_edges = []
        
        # We traverse the circuit.
        # If an edge is "added", we finish the current trail and start a new one?
        # Actually, the added edges connect the end of one path to the start of the next.
        # So the trail ends *before* the added edge? Or *after*?
        # Let's say the added edge is the "connector".
        # The original edges form the paths.
        # So we collect original edges. When we hit an added edge, we save the collected original edges as a trail.
        
        for eid in circuit_edges:
            u, v, is_added = edge_list[eid]
            if is_added:
                if current_trail_edges:
                    trails.append(current_trail_edges)
                    current_trail_edges = []
            else:
                current_trail_edges.append(eid)
        
        if current_trail_edges:
            trails.append(current_trail_edges)
        
        # Now convert each trail (list of edge IDs) into simple paths.
        all_paths = []
        for trail in trails:
            if not trail:
                continue
            
            # Reconstruct the vertex sequence of the trail
            # The trail edges are connected.
            # We need to determine the order of vertices.
            # The circuit gave us the order.
            # We can just use the circuit_vertices corresponding to these edges.
            
            # It's easier to just use the edge list to build the path.
            # But we need to know the start and end.
            # The trail is a sub-sequence of the circuit.
            
            # Let's reconstruct the path vertices from the edge IDs.
            # We know the circuit order.
            # We can map edge ID to position in circuit.
            
            # Actually, we can just walk the trail.
            # Start vertex is the 'u' of the first edge?
            # We need to know the direction.
            # In the circuit, we went from some vertex to another.
            # Let's store (u, v) for each edge in the circuit order.
            
            # Let's re-extract the directed edges from the circuit.
            # circuit_vertices[i] -> circuit_vertices[i+1] via circuit_edges[i].
            
            # We will create a list of (start, end) for each edge in the trail.
            trail_directed = []
            # We need to map edge ID to its position in the circuit to get the direction.
            # This is O(N^2) if we search.
            # Better: store direction when building circuit?
            # We didn't store it.
            
            # Let's just re-derive the path for each trail.
            # A trail is a sequence of edges.
            # We can find the start vertex (degree 1 in the trail subgraph) or just pick one and walk.
            # Since it's a trail from an Eulerian circuit, it's a path (not a cycle) unless the whole graph was Eulerian and no added edges?
            # If k=0, the whole circuit is one trail. It is a cycle.
            # If k>0, the trails are paths (start != end).
            
            # To be safe, we will convert the trail to a list of vertices.
            # We can do this by finding the start vertex.
            # In the subgraph formed by the trail edges, vertices have degree 1 (endpoints) or 2 (internal).
            # If it's a cycle, all have degree 2.
            
            # Let's just use the circuit order.
            # We know the circuit_edges list.
            # We can find the indices of the trail edges in circuit_edges.
            # Since the trail is contiguous in the circuit?
            # Yes, because we split at added edges.
            
            # So we can just take the corresponding vertices from circuit_vertices.
            
            # Let's find the start index of the trail in circuit_edges.
            # This is inefficient if we do it for every trail.
            
            # Alternative:
            # When we split, we also keep the vertex sequence.
            
            # Let's re-do the splitting with vertices.
            
        # Re-do splitting with vertices
        trails_vertices = []
        current_vertices = []
        if circuit_vertices:
            current_vertices.append(circuit_vertices[0])
        
        for i, eid in enumerate(circuit_edges):
            u, v, is_added = edge_list[eid]
            # The edge connects circuit_vertices[i] and circuit_vertices[i+1]
            # We append the next vertex
            current_vertices.append(circuit_vertices[i+1])
            
            if is_added:
                if len(current_vertices) > 1:
                    trails_vertices.append(current_vertices[:])
                # The next trail starts at the end of this one?
                # The added edge goes from A to B.
                # The previous trail ended at A.
                # The next trail starts at B.
                # So the new current_vertices should start with B.
                current_vertices = [circuit_vertices[i+1]]
        
        if len(current_vertices) > 1:
            trails_vertices.append(current_vertices)
        
        # Now convert each trail (list of vertices) to simple paths.
        for tv in trails_vertices:
            # tv is a list of vertices representing a trail.
            # We need to split it into simple paths.
            remaining = tv[:]
            while len(remaining) > 1:
                # Find first repeated vertex
                seen = {}
                repeat_idx = -1
                for i, v in enumerate(remaining):
                    if v in seen:
                        repeat_idx = i
                        break
                    seen[v] = i
                
                if repeat_idx == -1:
                    # No repeat, it's a simple path
                    all_paths.append(remaining)
                    break
                else:
                    # Split at repeat_idx
                    # Path 1: remaining[0] ... remaining[repeat_idx]
                    # Note: remaining[repeat_idx] is the same as remaining[seen[remaining[repeat_idx]]]
                    # The segment remaining[seen[remaining[repeat_idx]]] ... remaining[repeat_idx] is a cycle.
                    # We want to keep all edges.
                    # We can emit remaining[0] ... remaining[seen[remaining[repeat_idx]]] as a path?
                    # Wait, if we emit 0...k, the edges are 0-1, ..., k-1-k.
                    # The edges k...repeat_idx are in the cycle.
                    # We must include them.
                    # So we can't just drop the cycle.
                    
                    # Correct split:
                    # The trail is v0...vk...vj where vk=vj.
                    # We can split into:
                    # 1. v0...vk (simple path, since first repeat is at j, so 0..k is simple)
                    # 2. vk...vj (cycle)
                    # 3. vj...vm (rest)
                    # But we need to cover all edges.
                    # If we output v0...vk, we cover edges 0..k-1.
                    # The remaining edges are k..j-1 (cycle) and j..m-1 (rest).
                    # The cycle and rest are connected at vj (which is vk).
                    # So the remaining trail is vk...vj...vm.
                    # This is the same as starting the next iteration with vk...vm?
                    # No, the edges k..j-1 are the cycle.
                    # If we just set remaining = [vk, ..., vm], we lose the distinction?
                    # No, the list of vertices [vk, ..., vj, ..., vm] contains the cycle.
                    # So we can just set remaining = remaining[seen[remaining[repeat_idx]]:]
                    # And emit the path remaining[0:seen[remaining[repeat_idx]]+1]
                    
                    split_at = seen[remaining[repeat_idx]]
                    path_segment = remaining[:split_at+1]
                    if len(path_segment) > 1:
                        all_paths.append(path_segment)
                    remaining = remaining[split_at:]
                    
                    # If remaining is just one vertex, break
                    if len(remaining) <= 1:
                        break
        
        return all_paths

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
            # Simple greedy fallback
            available = set(tuple(sorted(e)) for e in edges)
            paths = []
            while available:
                u, v = next(iter(available))
                paths.append([u, v])
                available.remove(tuple(sorted((u, v))))
        
        results.append(paths)
    
    return results
