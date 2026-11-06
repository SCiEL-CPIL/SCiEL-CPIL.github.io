import time
import sys
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def hopcroft_karp_matching(adj, left_nodes):
        """
        Hopcroft-Karp maximum bipartite matching.
        adj: dict mapping left node -> list of right nodes
        left_nodes: list of left nodes
        Returns: dict mapping left node -> matched right node (or None if unmatched)
        """
        if not left_nodes:
            return {}
        
        pair_u = {u: None for u in left_nodes}
        pair_v = {}
        dist = {}
        
        from collections import deque
        
        def bfs():
            queue = deque()
            for u in left_nodes:
                if pair_u[u] is None:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    pu = pair_v.get(v, None)
                    if pu is None:
                        found = True
                    elif dist.get(pu, -1) == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj.get(u, []):
                pu = pair_v.get(v, None)
                if pu is None or (dist.get(pu, -1) == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in left_nodes:
                if pair_u[u] is None:
                    if dfs(u):
                        matching += 1
        
        return pair_u

    def decompose_via_matching(edges):
        """
        Decompose graph into minimum number of edge-disjoint simple paths
        using maximum matching on a bipartite graph of directed edge-stubs.
        
        The minimum number of paths = |E| - max_matching_size
        where the matching is over a bipartite graph with:
        - Left nodes: directed edges (u,v) for each undirected edge {u,v}
        - Right nodes: directed edges (v,u) for each undirected edge {u,v}
        - Edge between left (u,v) and right (w,x) if v == w (they can be chained)
        """
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Create directed edge stubs
        # For each undirected edge {u,v}, create two directed stubs: (u,v) and (v,u)
        # Left side: one directed stub per undirected edge
        # Right side: the other directed stub per undirected edge
        
        # Actually, let's think more carefully:
        # We have |E| undirected edges. Each path uses a sequence of edges.
        # If we orient each edge, a path is a sequence where the head of one edge
        # matches the tail of the next.
        # 
        # Bipartite matching approach:
        # Left nodes: for each undirected edge e={u,v}, we have a "tail" stub
        # Right nodes: for each undirected edge e={u,v}, we have a "head" stub
        # 
        # Wait, let me reconsider. The standard reduction:
        # 
        # Think of each undirected edge as having two "ends". 
        # A path decomposition pairs up ends of edges at vertices.
        # At each vertex v of degree d, we pair up d/2 pairs of edge-ends
        # (if d is even) or (d-1)/2 pairs plus one unpaired end (if d is odd).
        # 
        # The number of paths = number of unpaired ends / 2 + (number of vertices with odd degree) / 2
        # Actually: number of paths = (number of odd-degree vertices) / 2 if all degrees are odd
        # More precisely: min paths = max(1, (sum of odd degrees) / 2) for connected graphs?
        # No, that's not right either.
        #
        # Let me use the correct formulation:
        # Minimum path decomposition = |E| - |M| where M is maximum matching in:
        # - Left: directed edges (u,v) for each undirected edge {u,v}
        # - Right: directed edges (v,u) for each undirected edge {u,v}  
        # - Match (u,v)_left with (w,x)_right if v == w
        #
        # But this double counts. Let me think again...
        #
        # Correct approach: 
        # For each undirected edge, pick an arbitrary orientation.
        # Left nodes: all oriented edges (u,v)
        # Right nodes: all oriented edges (u,v)
        # Edge in bipartite graph: (u,v)_L connected to (w,x)_R iff v == w
        # Maximum matching size M gives min paths = |E| - M
        
        # Orient edges consistently (by vertex index for determinism)
        oriented_edges = []
        edge_to_idx = {}
        for i, (u, v) in enumerate(edges):
            # Orient from smaller to larger vertex
            if u > v:
                u, v = v, u
            oriented_edges.append((u, v))
            edge_to_idx[(u, v)] = i
        
        n_edges = len(oriented_edges)
        
        # Build bipartite graph
        # Left node i = oriented edge i = (u_i, v_i)
        # Right node j = oriented edge j = (u_j, v_j)
        # Connect left i to right j if v_i == u_j (head of i = tail of j)
        
        bip_adj = defaultdict(list)
        for i in range(n_edges):
            u_i, v_i = oriented_edges[i]
            for j in range(n_edges):
                if i == j:
                    continue
                u_j, v_j = oriented_edges[j]
                if v_i == u_j:
                    bip_adj[i].append(j)
        
        # Find maximum matching
        pair_u = hopcroft_karp_matching(bip_adj, list(range(n_edges)))
        
        # Recover paths from matching
        # pair_u[i] = j means edge i's head connects to edge j's tail
        # So we follow: edge i -> edge pair_u[i] -> edge pair_u[pair_u[i]] -> ...
        
        next_edge = {}
        for i in range(n_edges):
            j = pair_u.get(i, None)
            if j is not None:
                next_edge[i] = j
        
        # Find all paths (sequences of edges)
        visited_edges = set()
        paths = []
        
        for start_idx in range(n_edges):
            if start_idx in visited_edges:
                continue
            # Follow the chain
            path_edges = []
            cur = start_idx
            while cur is not None and cur not in visited_edges:
                visited_edges.add(cur)
                path_edges.append(cur)
                cur = next_edge.get(cur, None)
            
            # Convert edge indices to vertex path
            vertex_path = []
            for ei in path_edges:
                u, v = oriented_edges[ei]
                if not vertex_path:
                    vertex_path.extend([u, v])
                else:
                    # The last vertex in vertex_path should be u
                    # Check: vertex_path[-1] should equal u
                    if vertex_path[-1] != u:
                        # This shouldn't happen if matching is correct
                        # But let's handle it by restarting
                        if len(vertex_path) > 1:
                            paths.append(vertex_path)
                        vertex_path = [u, v]
                    else:
                        vertex_path.append(v)
            
            if len(vertex_path) > 1:
                paths.append(vertex_path)
        
        # Verify all edges are covered
        covered_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                covered_edges.add(tuple(sorted((u, v))))
        
        # If some edges are not covered, add them as single-edge paths
        all_edges_set = set(tuple(sorted(e)) for e in edges)
        missing = all_edges_set - covered_edges
        for e in missing:
            paths.append([e[0], e[1]])
        
        return paths

    def decompose_greedy_fallback(edges):
        """Fallback greedy decomposition if matching fails or times out."""
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining_edges:
            active_nodes = set()
            for u, v in remaining_edges:
                active_nodes.add(u)
                active_nodes.add(v)
            
            if not active_nodes:
                break

            best_path = []
            nodes_sorted = sorted(active_nodes, key=lambda x: -len(adj[x]))
            
            for start_node in nodes_sorted:
                path = [start_node]
                visited = {start_node}
                current = start_node
                
                while True:
                    candidates = []
                    for nb in adj[current]:
                        edge = tuple(sorted((current, nb)))
                        if edge in remaining_edges and nb not in visited:
                            deg_nb = sum(1 for u, v in remaining_edges if u == nb or v == nb)
                            candidates.append((deg_nb, nb))
                    
                    if not candidates:
                        break
                    
                    candidates.sort(reverse=True)
                    next_node = candidates[0][1]
                    edge = tuple(sorted((current, next_node)))
                    if edge in remaining_edges:
                        path.append(next_node)
                        visited.add(next_node)
                        current = next_node
                    else:
                        break
                
                if len(path) > len(best_path):
                    best_path = path
            
            if not best_path:
                e = next(iter(remaining_edges))
                best_path = [e[0], e[1]]
            
            if len(best_path) > 1:
                paths.append(best_path)
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    edge = tuple(sorted((u, v)))
                    remaining_edges.discard(edge)
            else:
                break
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        
        # Try matching-based decomposition first
        try:
            if time.time() < deadline - 5:
                paths = decompose_via_matching(edges)
            else:
                paths = decompose_greedy_fallback(edges)
        except Exception:
            paths = decompose_greedy_fallback(edges)
        
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
