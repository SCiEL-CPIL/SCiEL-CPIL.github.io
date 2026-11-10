"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
from random import Random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    _start = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def path_cover_via_matching(edges, adj, time_limit=40.0):
        """
        Minimum path cover via maximum matching on a bipartite graph.
        
        Each edge e = (u,v) is a "task". We can chain edge e1=(a,b) into e2=(b,c)
        (or e2=(c,b)) if they share a vertex. The number of paths = |edges| - |matching|,
        where matching is a maximum set of such valid chains (each edge can be 
        "tail" of at most one chain and "head" of at most one chain).
        
        We build a bipartite graph: left nodes = edges as "start", right nodes = edges as "end".
        Edge i -> j in bipartite graph if edge i's end vertex equals edge j's start vertex
        (considering both orientations). We want maximum bipartite matching.
        
        Actually, let's think more carefully. Each edge e=(u,v) can be traversed as u->v or v->u.
        If we choose orientation u->v, then the "end" vertex is v. Another edge e'=(v,w) can follow.
        
        This is equivalent to: we have |E| items. We want to maximize the number of "links" 
        where link (i,j) means edge i is immediately followed by edge j in a path.
        Constraints: each edge can have at most one successor and at most one predecessor.
        This is exactly a maximum matching in a bipartite graph where:
          - Left side: edges as potential "predecessors" 
          - Right side: edges as potential "successors"
          - Arc (i,j) exists if edge i and edge j can be adjacent in a path (share a vertex,
            and the shared vertex is the "end" of i and "start" of j for some orientation).
        
        But we also need to ensure the resulting structure is a collection of simple paths 
        (no cycles, no vertex repeats). The matching gives us a set of directed adjacencies.
        If we ignore the simplicity constraint, we might get cycles. But for these small graphs,
        let's first try and then fix cycles.
        
        Actually, a cleaner formulation: 
        - Each edge e=(u,v) has two possible orientations: u->v or v->u.
        - We choose one orientation per edge.
        - Then we form a directed graph where each edge is a directed arc.
        - The number of paths = number of connected components in the directed graph that are paths.
        - To minimize paths, we maximize the number of "connections" where the end of one 
          directed edge equals the start of another.
        
        This is getting complex. Let me use a simpler approach:
        
        Build a bipartite graph:
          - Left nodes: edges (as "first edge in a potential pair")
          - Right nodes: edges (as "second edge in a potential pair")  
          - For each pair (e1, e2) where e1 != e2:
            e1=(a,b), e2=(c,d). They can be chained if {a,b} ∩ {c,d} != empty AND
            the shared vertex is an endpoint of both (which it always is since edges are simple).
            More precisely: if b==c or b==d or a==c or a==d, they can potentially be chained.
            But we need to choose orientations consistently.
            
        Simplification: For each unordered pair of edges that share a vertex, add an arc 
        in the bipartite graph. The maximum matching gives us the maximum number of 
        "adjacent edge pairs" we can form. Each such pair reduces the path count by 1.
        Initial path count = |E| (each edge is its own path). After matching of size k,
        path count = |E| - k.
        
        Wait, that's not quite right either. Let me think again.
        
        Actually, the standard reduction: minimum path cover in a DAG = |V| - max matching.
        But our graph isn't a DAG. 
        
        Let me use a different approach: model it as finding the maximum number of 
        edge-disjoint paths that cover all edges, which is equivalent to finding a 
        decomposition into the fewest paths.
        
        Alternative: Use the fact that in any path decomposition, the number of paths 
        is at least (number of odd-degree vertices) / 2. And we can achieve this bound 
        if the graph is connected (by adding virtual edges to make it Eulerian, as in 
        the current program). But the current program already does this.
        
        Let me try a completely different angle: use ILP via scipy's linprog or 
        just do a smarter greedy with backtracking.
        
        Actually, let me implement the matching approach properly:
        
        For each edge e=(u,v), consider it as a node in a directed multigraph where 
        we can go from u to v or v to u. We want to assign a direction to each edge 
        and then find the minimum number of directed paths that cover all edges.
        
        This is equivalent to: maximize the number of "internal vertices" in the 
        path decomposition, where an internal vertex is one where an incoming edge 
        is followed by an outgoing edge.
        
        Let me just implement a clean version:
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        if n_edges == 1:
            return [list(edges[0])]
        
        # For each edge, store its two endpoints
        # Build adjacency: for each vertex, which edges are incident
        vertex_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)
        
        # Build bipartite graph for matching
        # Left: edges as "start", Right: edges as "end"
        # Arc (i, j) if edge i and edge j share a vertex AND i != j
        # The weight is 1 for each valid arc
        # We want maximum bipartite matching
        
        # Cost matrix for Hungarian: cost = 0 if valid connection, cost = INF if not
        INF = 1e9
        cost = np.full((n_edges, n_edges), INF, dtype=float)
        
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(n_edges):
                if i == j:
                    continue
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    cost[i][j] = 0.0
        
        # Solve maximum bipartite matching via Hungarian (minimize cost, where 0 = valid)
        # linear_sum_assignment minimizes the sum of costs
        # We want to maximize the number of valid assignments, so minimize cost
        # where valid = 0 and invalid = INF
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # The matching pairs row_ind[k] with col_ind[k]
        # A valid match is where cost[row_ind[k]][col_ind[k]] == 0
        # Count valid matches
        valid_matches = []
        for k in range(len(row_ind)):
            if cost[row_ind[k]][col_ind[k]] < INF / 2:
                valid_matches.append((row_ind[k], col_ind[k]))
        
        # Number of paths = n_edges - len(valid_matches)
        # But we need to reconstruct the actual paths
        
        # Build the chain structure
        # next_edge[i] = j if edge i is followed by edge j
        # prev_edge[j] = i if edge i is followed by edge j
        next_edge = [None] * n_edges
        prev_edge = [None] * n_edges
        for (i, j) in valid_matches:
            next_edge[i] = j
            prev_edge[j] = i
        
        # Find all paths by following chains
        # A path starts at an edge with no predecessor
        paths = []
        used = [False] * n_edges
        
        for i in range(n_edges):
            if prev_edge[i] is not None:
                continue  # not a start
            # Follow the chain
            path_edges = []
            cur = i
            while cur is not None and not used[cur]:
                used[cur] = True
                path_edges.append(cur)
                cur = next_edge[cur]
            
            # Convert edge indices to vertex path
            if not path_edges:
                continue
            
            # Determine orientations to form a valid vertex path
            # For a sequence of edges e1, e2, ..., ek, we need to orient them 
            # so that the end of ei matches the start of e(i+1)
            vertex_path = []
            for idx, ei in enumerate(path_edges):
                u, v = edges[ei]
                if idx == 0:
                    # First edge: choose orientation
                    vertex_path.append(u)
                    vertex_path.append(v)
                else:
                    # Must match: the last vertex of current path must be one endpoint of this edge
                    last_v = vertex_path[-1]
                    if last_v == u:
                        vertex_path.append(v)
                    elif last_v == v:
                        vertex_path.append(u)
                    else:
                        # Orientation conflict - this shouldn't happen if matching is valid
                        # Fallback: just append both
                        vertex_path.append(u)
                        vertex_path.append(v)
            
            if len(vertex_path) >= 2:
                paths.append(vertex_path)
        
        # Check if all edges are covered
        if not all(used):
            # Some edges weren't included - add them as single-edge paths
            for i in range(n_edges):
                if not used[i]:
                    u, v = edges[i]
                    paths.append([u, v])
                    used[i] = True
        
        # Verify and fix: ensure all paths are simple and use valid edges
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))
        
        # Validate paths
        covered = set()
        valid_paths = []
        for p in paths:
            if len(p) < 2:
                continue
            if len(set(p)) != len(p):
                # Not simple - break into single edges
                for i in range(len(p) - 1):
                    u, v = p[i], p[i+1]
                    key = (u, v) if u <= v else (v, u)
                    if key in edge_set and key not in covered:
                        valid_paths.append([u, v])
                        covered.add(key)
                continue
            ok = True
            for i in range(len(p) - 1):
                u, v = p[i], p[i+1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set or key in covered:
                    ok = False
                    break
                covered.add(key)
            if ok:
                valid_paths.append(p)
        
        # Add any uncovered edges as single-edge paths
        for (u, v) in edges:
            key = (u, v) if u <= v else (v, u)
            if key not in covered:
                valid_paths.append([u, v])
                covered.add(key)
        
        return valid_paths

    def greedy_path_cover(edges):
        """Fallback greedy path cover."""
        if not edges:
            return []
        
        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v) if u <= v else (v, u))
        
        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        used_edges = set()
        paths = []
        
        while edge_set - used_edges:
            remaining = edge_set - used_edges
            if not remaining:
                break
            
            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
            
            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x])) if rem_adj else None
            if start is None:
                break
            
            path = [start]
            visited_in_path = {start}
            current = start
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited_in_path]
                if not neighbors:
                    break
                best_neighbor = max(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited_in_path]))
                path.append(best_neighbor)
                visited_in_path.add(best_neighbor)
                current = best_neighbor
            
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)
            
            valid = all(e in remaining for e in path_edges)
            if valid and path_edges:
                used_edges |= path_edges
                paths.append(path)
            else:
                e = next(iter(remaining))
                used_edges.add(e)
                paths.append([e[0], e[1]])
        
        return paths

    def eulerian_decomposition(edges):
        """Eulerian-based decomposition for good initial solution."""
        if not edges:
            return []
        
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        def find_eulerian_circuit(edge_list, adj_map):
            if not edge_list:
                return []
            
            edge_count = defaultdict(int)
            for (u, v) in edge_list:
                key = (u, v) if u <= v else (v, u)
                edge_count[key] += 1
            
            hier_adj = defaultdict(list)
            for (u, v) in edge_list:
                key = (u, v) if u <= v else (v, u)
                hier_adj[u].append((v, key))
                hier_adj[v].append((u, key))
            
            used_count = defaultdict(int)
            start = next(iter(adj_map))
            stack = [start]
            circuit = []
            while stack:
                v = stack[-1]
                found = False
                for (nb, key) in hier_adj[v]:
                    if used_count[key] < edge_count[key]:
                        used_count[key] += 1
                        stack.append(nb)
                        found = True
                        break
                if not found:
                    circuit.append(stack.pop())
            circuit.reverse()
            return circuit
        
        def bfs_shortest_path(src, dst, adj_map):
            if src == dst:
                return []
            visited = {src}
            parent = {src: None}
            q = deque([src])
            while q:
                cur = q.popleft()
                for nb in adj_map[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = cur
                        if nb == dst:
                            path = []
                            node = dst
                            while node is not None:
                                path.append(node)
                                node = parent[node]
                            path.reverse()
                            return path
                        q.append(nb)
            return None
        
        if not odd_vertices:
            circuit = find_eulerian_circuit(edges, adj)
            if circuit:
                return [circuit]
            return greedy_path_cover(edges)
        
        pairs = []
        for i in range(0, len(odd_vertices) - 1, 2):
            pairs.append((odd_vertices[i], odd_vertices[i + 1]))
        
        virtual_edges = []
        for (a, b) in pairs:
            sp = bfs_shortest_path(a, b, adj)
            if sp and len(sp) >= 2:
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    virtual_edges.append(key)
        
        all_edges_for_euler = list(edges) + virtual_edges
        circuit = find_eulerian_circuit(all_edges_for_euler, adj)
        
        virtual_set = set(virtual_edges)
        
        paths = []
        current_path = [circuit[0]]
        for i in range(1, len(circuit)):
            u, v = circuit[i - 1], circuit[i]
            key = (u, v) if u <= v else (v, u)
            current_path.append(v)
            if key in virtual_set:
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = [v]
        
        if len(current_path) >= 2:
            paths.append(current_path)
        
        paths = [p for p in paths if len(p) >= 2]
        
        orig_edge_set = set()
        for (u, v) in edges:
            orig_edge_set.add((u, v) if u <= v else (v, u))
        
        covered = set()
        valid = True
        for p in paths:
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in orig_edge_set:
                    valid = False
                    break
                if key in covered:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break
        
        if valid and covered == orig_edge_set:
            return paths
        else:
            return greedy_path_cover(edges)

    def local_search_improve(paths, edges, adj, time_limit=5.0):
        """Local search to reduce number of paths via merging."""
        start_time = time.time()
        best_paths = [list(p) for p in paths]
        
        def is_simple_path(path):
            return len(set(path)) == len(path)
        
        while time.time() - start_time < time_limit:
            improved = False
            n_paths = len(best_paths)
            
            for i in range(n_paths):
                if time.time() - start_time > time_limit:
                    return best_paths
                for j in range(i + 1, n_paths):
                    p1 = best_paths[i]
                    p2 = best_paths[j]
                    
                    candidates = []
                    
                    u, v = p1[-1], p2[0]
                    if (min(u,v), max(u,v)) in adj:
                        new_path = p1 + p2[1:]
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                    
                    u, v = p1[0], p2[0]
                    if (min(u,v), max(u,v)) in adj:
                        new_path = list(reversed(p1)) + p2[1:]
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                    
                    u, v = p1[-1], p2[-1]
                    if (min(u,v), max(u,v)) in adj:
                        new_path = p1 + list(reversed(p2[1:]))
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                    
                    u, v = p1[0], p2[-1]
                    if (min(u,v), max(u,v)) in adj:
                        new_path = list(reversed(p1)) + list(reversed(p2[1:]))
                        if is_simple_path(new_path):
                            candidates.append(new_path)
                    
                    if candidates:
                        merged = candidates[0]
                        new_paths = []
                        for k in range(n_paths):
                            if k == i or k == j:
                                continue
                            new_paths.append(best_paths[k])
                        new_paths.append(merged)
                        best_paths = new_paths
                        improved = True
                        break
                if improved:
                    break
                    
            if not improved:
                if len(best_paths) > 0:
                    idx = random.randint(0, len(best_paths) - 1)
                    best_paths[idx] = list(reversed(best_paths[idx]))
                else:
                    break
        
        return best_paths

    def validate_paths(paths, edges):
        """Validate that paths are simple, use graph edges, are edge-disjoint, and cover all edges."""
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))
        
        covered = set()
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    return False
                if key in covered:
                    return False
                covered.add(key)
        
        return covered == edge_set

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 15:
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        adj = build_adjacency(edges)
        
        # Strategy 1: Eulerian decomposition (structural, good baseline)
        try:
            paths_euler = eulerian_decomposition(edges)
        except Exception:
            paths_euler = greedy_path_cover(edges)
        
        if not paths_euler or not validate_paths(paths_euler, edges):
            paths_euler = greedy_path_cover(edges)
        
        # Strategy 2: Matching-based path cover
        remaining_time = _DEADLINE - time.time() - 5.0
        if remaining_time > 10:
            try:
                paths_match = path_cover_via_matching(edges, adj, time_limit=min(remaining_time, 30.0))
                if paths_match and validate_paths(paths_match, edges):
                    if len(paths_match) < len(paths_euler):
                        paths_euler = paths_match
            except Exception:
                pass
        
        # Strategy 3: Local search improvement
        remaining_time = _DEADLINE - time.time() - 3.0
        if remaining_time > 2:
            try:
                improved = local_search_improve(paths_euler, edges, adj, time_limit=min(remaining_time, 10.0))
                if improved and validate_paths(improved, edges):
                    if len(improved) < len(paths_euler):
                        paths_euler = improved
            except Exception:
                pass
        
        # Final validation
        if not validate_paths(paths_euler, edges):
            paths_euler = greedy_path_cover(edges)
            if not validate_paths(paths_euler, edges):
                paths_euler = [[list(e)] for e in edges]
        
        results.append(paths_euler)
    
    return results
# EVOLVE-BLOCK-END
