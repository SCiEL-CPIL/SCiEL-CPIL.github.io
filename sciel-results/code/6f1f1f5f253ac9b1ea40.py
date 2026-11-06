import time
import random
import sys
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_weight_perfect_matching(weights, size):
        """
        Find minimum weight perfect matching in a complete graph with 'size' vertices.
        weights is a 2D array of size x size.
        size must be even and small (<= 20 for DP to be fast).
        """
        if size == 0:
            return 0, []
        
        # DP over subsets
        # dp[mask] = min cost to match vertices in mask
        # mask is a bitmask of size 'size'
        dp = [float('inf')] * (1 << size)
        dp[0] = 0
        parent = [None] * (1 << size) # To reconstruct
        
        for mask in range(1, 1 << size):
            # Find the lowest set bit
            lsb = (mask & -mask).bit_length() - 1
            if mask & (1 << lsb) == 0:
                continue
            
            # Try pairing lsb with every other vertex i in mask
            rest = mask ^ (1 << lsb)
            for i in range(size):
                if rest & (1 << i):
                    prev_mask = rest ^ (1 << i)
                    cost = weights[lsb][i] + dp[prev_mask]
                    if cost < dp[mask]:
                        dp[mask] = cost
                        parent[mask] = (lsb, i)
        
        # Reconstruct
        matches = []
        mask = (1 << size) - 1
        while mask:
            lsb, i = parent[mask]
            matches.append((lsb, i))
            mask ^= (1 << lsb)
            mask ^= (1 << i)
            
        return dp[(1 << size) - 1], matches

    def decompose_graph(edges):
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find odd degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        edges_to_remove = set()
        
        if odd_vertices:
            # Map odd vertices to indices 0..k-1
            k = len(odd_vertices)
            odd_to_idx = {v: i for i, v in enumerate(odd_vertices)}
            idx_to_odd = {i: v for i, v in enumerate(odd_vertices)}
            
            # Compute shortest paths between all pairs of odd vertices
            # Use Dijkstra from each odd vertex
            n = max(adj.keys()) + 1 if adj else 0
            dist = [[float('inf')] * n for _ in range(n)]
            next_hop = [[-1] * n for _ in range(n)]
            
            for v in odd_vertices:
                # Dijkstra
                dist_v = [float('inf')] * n
                dist_v[v] = 0
                next_hop_v = [-1] * n
                pq = [(0, v)]
                while pq:
                    d, u = min(pq)
                    if d > dist_v[u]:
                        continue
                    for w in adj[u]:
                        nd = d + 1
                        if nd < dist_v[w]:
                            dist_v[w] = nd
                            next_hop_v[w] = u
                            pq.append((nd, w))
                
                for w in odd_vertices:
                    dist[v][w] = dist_v[w]
                    # Store next hop for path reconstruction
                    # We need to reconstruct the path later
                    # For now, store the dist matrix
                    
            # We need to reconstruct paths to know which edges to remove.
            # Let's build a distance matrix for the odd vertices only
            w_matrix = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    if i == j:
                        w_matrix[i][j] = 0
                    else:
                        w_matrix[i][j] = dist[odd_vertices[i]][odd_vertices[j]]
            
            # Find min weight perfect matching
            cost, matches = min_weight_perfect_matching(w_matrix, k)
            
            # For each match (i, j), find the shortest path between odd_vertices[i] and odd_vertices[j]
            # and add its edges to edges_to_remove
            # We need a way to get the path. Let's re-run Dijkstra to store parents or use a simple BFS for small graphs.
            # Given N<=30, BFS is fine.
            
            for i, j in matches:
                u_start = odd_vertices[i]
                u_end = odd_vertices[j]
                
                # BFS to find shortest path
                parent = {u_start: None}
                queue = [u_start]
                while queue:
                    u = queue.pop(0)
                    if u == u_end:
                        break
                    for w in adj[u]:
                        if w not in parent:
                            parent[w] = u
                            queue.append(w)
                
                # Reconstruct path
                if u_end not in parent:
                    continue # Should not happen if connected
                path = []
                curr = u_end
                while curr != u_start:
                    path.append(curr)
                    curr = parent[curr]
                path.append(u_start)
                path.reverse()
                
                # Add edges of this path to edges_to_remove
                for idx in range(len(path) - 1):
                    u, v = path[idx], path[idx+1]
                    edges_to_remove.add(tuple(sorted((u, v))))
        
        # Remove edges_to_remove from the graph
        remaining_edges = set(edges) - edges_to_remove
        # The removed edges are single pieces
        pieces = [list(e) for e in edges_to_remove]
        
        # Decompose remaining_edges (Eulerian) into cycles
        if remaining_edges:
            # Rebuild adj for remaining
            adj_rem = defaultdict(set)
            for u, v in remaining_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            # Hierholzer's Algorithm with heuristic
            # We want to minimize the number of cycles.
            # Heuristic: Always extend to the neighbor with the highest remaining degree.
            
            used_edges = set()
            
            def get_cycle(start):
                stack = [start]
                path = []
                while stack:
                    u = stack[-1]
                    # Find an unused edge from u
                    # Sort neighbors by remaining degree (desc) to prioritize "bottlenecks" or keep options open?
                    # Actually, picking the edge that leads to a vertex with the *lowest* remaining degree (after this edge is used)
                    # is often good for Eulerian paths (Fleury's heuristic), but for cycle decomposition,
                    # we want to avoid trapping ourselves.
                    # Let's try: pick neighbor with highest degree in remaining graph.
                    
                    neighbors = list(adj_rem[u])
                    # Filter out used edges
                    valid_neighbors = []
                    for v in neighbors:
                        e = tuple(sorted((u, v)))
                        if e not in used_edges:
                            valid_neighbors.append(v)
                    
                    if not valid_neighbors:
                        stack.pop()
                        if stack:
                            path.append(stack[-1])
                        continue
                    
                    # Heuristic: pick neighbor with highest degree in remaining graph
                    # Degree in remaining graph = len(adj_rem[v]) - (number of used edges incident to v)
                    # Approximation: just use len(adj_rem[v])
                    # Better: calculate actual remaining degree
                    def rem_deg(x):
                        deg = 0
                        for y in adj_rem[x]:
                            e = tuple(sorted((x, y)))
                            if e not in used_edges:
                                deg += 1
                        return deg
                    
                    valid_neighbors.sort(key=lambda v: rem_deg(v), reverse=True)
                    v = valid_neighbors[0]
                    
                    e = tuple(sorted((u, v)))
                    used_edges.add(e)
                    stack.append(v)
                    
                # The path is built in reverse
                if path:
                    path.reverse()
                    # The cycle is path + [start]
                    return path + [start]
                return None
            
            # We need to find all cycles.
            # We iterate through vertices. If a vertex has unused edges, start a cycle.
            all_vertices = list(adj_rem.keys())
            for v in all_vertices:
                # Check if v has unused edges
                has_unused = False
                for w in adj_rem[v]:
                    e = tuple(sorted((v, w)))
                    if e not in used_edges:
                        has_unused = True
                        break
                if has_unused:
                    cycle = get_cycle(v)
                    if cycle:
                        # Remove duplicate start/end if any (shouldn't be)
                        # The cycle is a list of vertices [v0, v1, ..., vk, v0]
                        # We return it as is
                        pieces.append(cycle)
                        
        return pieces

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]
    
    results = []
    for i, edges in enumerate(graphs):
        if time.time() > deadline - 2.0:
            # Fallback: trivial decomposition
            results.append([list(e) for e in edges])
            continue
            
        try:
            pieces = decompose_graph(edges)
            results.append(pieces)
        except Exception as e:
            # Fallback on error
            results.append([list(e) for e in edges])

    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
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
                if len(set(piece)) != len(piece):
                    valid = False
                    break
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
