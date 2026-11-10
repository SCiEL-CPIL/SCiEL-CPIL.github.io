"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
from random import Random
import numpy as np

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

    def simulated_annealing_path_cover(edges, adj, time_limit=30.0):
        """
        Simulated annealing over edge-orientation and successor-assignment.
        
        State: 
          - perm: permutation of edge indices (defines the order in which edges are considered)
          - orient: for each edge index, 0 or 1 (which endpoint is 'start')
        
        We build paths by scanning perm in order. For each edge, we try to attach it 
        to an existing path if it shares a vertex with the end of that path (considering orientation).
        If not, we start a new path.
        
        Cost: number of paths + penalty for non-simple paths.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        if n_edges == 1:
            return [list(edges[0])]
        
        # Precompute edge endpoints
        edge_u = [e[0] for e in edges]
        edge_v = [e[1] for e in edges]
        
        # State: perm (permutation of 0..n_edges-1), orient (0 or 1 for each position in perm)
        # orient[i] = 0 means edge perm[i] goes from edge_u[perm[i]] to edge_v[perm[i]]
        # orient[i] = 1 means edge perm[i] goes from edge_v[perm[i]] to edge_u[perm[i]]
        
        def get_oriented_endpoints(perm, orient, idx):
            """Get (start, end) for edge at position idx in perm."""
            ei = perm[idx]
            if orient[idx] == 0:
                return (edge_u[ei], edge_v[ei])
            else:
                return (edge_v[ei], edge_u[ei])
        
        def build_paths_from_state(perm, orient):
            """Build paths by scanning perm in order, attaching edges to existing paths."""
            paths = []  # list of lists of (start, end) vertices
            path_ends = []  # last vertex of each path
            
            for idx in range(n_edges):
                start, end = get_oriented_endpoints(perm, orient, idx)
                attached = False
                # Try to attach to an existing path if its end matches start
                for p_idx in range(len(paths)):
                    if path_ends[p_idx] == start:
                        # Attach: extend path p_idx with this edge
                        paths[p_idx].append(end)
                        path_ends[p_idx] = end
                        attached = True
                        break
                if not attached:
                    # Start a new path
                    paths.append([start, end])
                    path_ends.append(end)
            
            return paths
        
        def evaluate_state(perm, orient):
            """Return (num_paths, penalty) for a state."""
            paths = build_paths_from_state(perm, orient)
            num_paths = len(paths)
            penalty = 0
            for p in paths:
                if len(set(p)) != len(p):
                    penalty += 100  # heavy penalty for non-simple path
            return num_paths, penalty
        
        def get_paths_as_vertex_lists(perm, orient):
            """Convert state to list of vertex lists (paths)."""
            paths = build_paths_from_state(perm, orient)
            return [list(p) for p in paths]
        
        # Initialize state
        perm = list(range(n_edges))
        orient = [0] * n_edges
        random.shuffle(perm)
        
        best_perm = perm[:]
        best_orient = orient[:]
        best_cost = float('inf')
        best_paths = None
        
        cur_perm = perm[:]
        cur_orient = orient[:]
        cur_cost, cur_penalty = evaluate_state(cur_perm, cur_orient)
        cur_total = cur_cost + cur_penalty
        
        if cur_total < best_cost:
            best_cost = cur_total
            best_perm = cur_perm[:]
            best_orient = cur_orient[:]
            best_paths = get_paths_as_vertex_lists(best_perm, best_orient)
        
        start_time = time.time()
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
            
            # Generate neighbor
            new_perm = cur_perm[:]
            new_orient = cur_orient[:]
            
            move_type = random.random()
            if move_type < 0.5:
                # Swap two positions in perm
                i, j = random.sample(range(n_edges), 2)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            elif move_type < 0.8:
                # Flip orientation at a random position
                i = random.randint(0, n_edges - 1)
                new_orient[i] = 1 - new_orient[i]
            else:
                # Flip orientations at two random positions
                i, j = random.sample(range(n_edges), 2)
                new_orient[i] = 1 - new_orient[i]
                new_orient[j] = 1 - new_orient[j]
            
            new_cost, new_penalty = evaluate_state(new_perm, new_orient)
            new_total = new_cost + new_penalty
            
            delta = new_total - cur_total
            if delta < 0 or random.random() < np.exp(-delta / T):
                cur_perm = new_perm
                cur_orient = new_orient
                cur_cost = new_cost
                cur_penalty = new_penalty
                cur_total = new_total
                
                if cur_total < best_cost:
                    best_cost = cur_total
                    best_perm = cur_perm[:]
                    best_orient = cur_orient[:]
                    best_paths = get_paths_as_vertex_lists(best_perm, best_orient)
            
            T *= cooling
        
        return best_paths if best_paths else greedy_path_cover(edges)

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
        
        # Strategy 2: Simulated annealing
        remaining_time = _DEADLINE - time.time() - 5.0
        if remaining_time > 10:
            try:
                paths_sa = simulated_annealing_path_cover(edges, adj, time_limit=min(remaining_time, 40.0))
                if paths_sa and validate_paths(paths_sa, edges):
                    if len(paths_sa) < len(paths_euler):
                        paths_euler = paths_sa
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
