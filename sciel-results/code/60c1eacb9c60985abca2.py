"""Path decomposition: max-matching on edge-ends + local search + merge."""
import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_end_matching_decompose(edges, adj, time_limit=60.0, rng_seed=0):
        """
        Formulate path decomposition as matching on directed edge-ends.
        Each edge (u,v) has two directed ends: (u->v) and (v->u).
        A 'connection' between end (u->v) of edge i and end (x->y) of edge j
        is valid if v == x (or v == y with appropriate direction) and the
        combined path is simple.
        
        We build an auxiliary graph where each directed edge-end is a node,
        and we find a maximum matching. The number of paths = |E| - |matching|.
        """
        rng = random.Random(rng_seed)
        m = len(edges)
        if m == 0:
            return []
        if m == 1:
            return [list(edges[0])]

        # Create directed edge-ends: for edge (u,v), we have (u,v) and (v,u)
        # Index them: edge i has ends 2*i (u->v) and 2*i+1 (v->u)
        # So total nodes = 2*m
        
        # Build adjacency in auxiliary graph:
        # Node (2*i) represents edge i starting from u_i going to v_i
        # Node (2*i+1) represents edge i starting from v_i going to u_i
        # 
        # A connection from node a to node b means: after traversing the directed
        # edge corresponding to node a, we continue with the directed edge for node b.
        # This is valid if the end vertex of edge a == start vertex of edge b.
        
        # For edge i = (u_i, v_i):
        #   Node 2*i: direction u_i -> v_i, starts at u_i, ends at v_i
        #   Node 2*i+1: direction v_i -> u_i, starts at v_i, ends at u_i
        
        # Connection from node a to node b is valid if:
        #   end_vertex(a) == start_vertex(b) and a != b and a != complement(b)
        #   where complement(2*i) = 2*i+1 and vice versa (same edge, opposite direction)
        
        # Build end_vertex and start_vertex arrays
        start_v = [0] * (2 * m)
        end_v = [0] * (2 * m)
        for i, (u, v) in enumerate(edges):
            start_v[2 * i] = u
            end_v[2 * i] = v
            start_v[2 * i + 1] = v
            end_v[2 * i + 1] = u

        # Build adjacency list for auxiliary graph
        # adj_aux[node] = list of nodes it can connect to
        adj_aux = defaultdict(list)
        for a in range(2 * m):
            for b in range(2 * m):
                if a == b:
                    continue
                # a and b should be different edges (not complements)
                edge_a = a // 2
                edge_b = b // 2
                if edge_a == edge_b:
                    continue
                if end_v[a] == start_v[b]:
                    adj_aux[a].append(b)

        # Find maximum matching using greedy + local search
        # Matching: dict mapping matched node -> matched partner
        matching = {}  # node -> partner node
        
        # Greedy: repeatedly pick a valid connection
        used_nodes = set()
        for a in range(2 * m):
            if time.time() - start_time > time_limit:
                break
            if a in used_nodes:
                continue
            # Find best partner for a
            best_b = None
            for b in adj_aux.get(a, []):
                if b in used_nodes:
                    continue
                best_b = b
                break
            if best_b is not None:
                matching[a] = best_b
                matching[best_b] = a
                used_nodes.add(a)
                used_nodes.add(best_b)

        # Local search: try to improve matching by finding augmenting paths
        # or by swapping matches
        improved = True
        max_iters = 50
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > time_limit:
                break
            
            # Try to find augmenting paths (BFS from unmatched nodes)
            unmatched = [i for i in range(2 * m) if i not in matching]
            for start in unmatched:
                if time.time() - start_time > time_limit:
                    break
                # BFS to find augmenting path
                parent = {start: None}
                queue = [start]
                found = False
                while queue and not found:
                    curr = queue.pop(0)
                    if time.time() - start_time > time_limit:
                        break
                    for nb in adj_aux.get(curr, []):
                        if nb in parent:
                            continue
                        if nb in matching:
                            partner = matching[nb]
                            if partner not in parent:
                                parent[partner] = nb
                                queue.append(partner)
                        else:
                            # Found augmenting path
                            parent[nb] = curr
                            found = True
                            break
                    if found:
                        break
                
                if found and nb in parent:
                    # Augment
                    path = []
                    node = nb
                    while node is not None:
                        path.append(node)
                        if parent[node] is None:
                            break
                        node = parent[node]
                    # path is [start, ..., nb] with alternating unmatched/matched edges
                    # Actually let me redo this more carefully
                    # Reconstruct augmenting path
                    aug_path = []
                    node = nb
                    while node is not None:
                        aug_path.append(node)
                        if parent[node] is None:
                            break
                        node = parent[node]
                    # aug_path goes from nb back to start
                    # The edges in the augmenting path are:
                    # (aug_path[0], aug_path[1]), (aug_path[2], aug_path[3]), ...
                    # where odd-indexed pairs are matched edges, even-indexed are unmatched
                    # Actually, let's just flip: for each consecutive pair in the path,
                    # if they're matched, unmatch them; if not, match them.
                    # But the path alternates: unmatched, matched, unmatched, matched, ...
                    # Let me just do it directly.
                    for i in range(len(aug_path) - 1):
                        u_node = aug_path[i]
                        v_node = aug_path[i + 1]
                        if u_node in matching and matching[u_node] == v_node:
                            # Unmatch
                            del matching[u_node]
                            del matching[v_node]
                        else:
                            # Match
                            matching[u_node] = v_node
                            matching[v_node] = u_node
                    improved = True
                    break

            # Try 1-swap improvements
            if not improved:
                for a in list(matching.keys()):
                    if time.time() - start_time > time_limit:
                        break
                    b = matching[a]
                    # Try to swap a's match
                    for new_b in adj_aux.get(a, []):
                        if new_b == b or new_b in matching:
                            continue
                        # If we match a->new_b, what happens to b?
                        # b becomes unmatched. Can we match b to something?
                        # For simplicity, only do direct swaps
                        pass
                    # Try 2-opt style: if a is matched to b, and c is matched to d,
                    # can we rematch a->c and b->d?
                    break  # Skip complex swaps for now

        # Reconstruct paths from matching
        # Each edge i has two directed ends. The matching tells us which ends are connected.
        # A path is a sequence of directed edges where each consecutive pair is matched.
        
        # Build the path structure:
        # For each edge, determine its role in paths.
        # Node 2*i is "edge i, direction u->v"
        # Node 2*i+1 is "edge i, direction v->u"
        
        # A path starts at a node with no incoming match and follows matches.
        # Incoming to node a means some node b is matched to a (matching[b] == a).
        
        paths_found = []
        visited_edges = set()
        
        for i in range(m):
            if i in visited_edges:
                continue
            
            # Try to build a path starting from edge i
            # Determine which direction edge i takes
            # Check if 2*i or 2*i+1 has an incoming match
            # If 2*i has incoming (matching[2*i+1] == 2*i or matching[something] == 2*i)
            
            # Let's find the start of the path containing edge i
            # Start: a node with no predecessor in the matching
            # Node a has a predecessor if there exists b such that matching[b] == a
            
            has_pred_0 = False  # does node 2*i have a predecessor?
            has_pred_1 = False  # does node 2*i+1 have a predecessor?
            
            for b in range(2 * m):
                if b in matching and matching[b] == 2 * i:
                    has_pred_0 = True
                    break
            for b in range(2 * m):
                if b in matching and matching[b] == 2 * i + 1:
                    has_pred_1 = True
                    break
            
            # Determine starting direction
            if has_pred_0 and not has_pred_1:
                # Start with direction 2*i+1 (v->u)
                start_node = 2 * i + 1
            elif has_pred_1 and not has_pred_0:
                # Start with direction 2*i (u->v)
                start_node = 2 * i
            elif has_pred_0 and has_pred_1:
                # Both have predecessors - this is a cycle! 
                # Pick one and break it
                start_node = 2 * i
            else:
                # Neither has predecessors - isolated or start
                start_node = 2 * i
            
            # Follow the path
            path_nodes = []
            curr = start_node
            while curr is not None:
                path_nodes.append(curr)
                if curr in matching:
                    curr = matching[curr]
                else:
                    break
            
            # Convert path_nodes to vertex sequence
            vertex_path = [start_v[start_node]]
            for node in path_nodes:
                vertex_path.append(end_v[node])
            
            # Verify simplicity
            if len(vertex_path) == len(set(vertex_path)):
                # Mark edges as visited
                for node in path_nodes:
                    visited_edges.add(node // 2)
                paths_found.append(vertex_path)
            else:
                # Cycle detected - break it by making each edge a separate path
                for node in path_nodes:
                    e_idx = node // 2
                    if e_idx not in visited_edges:
                        visited_edges.add(e_idx)
                        u, v = edges[e_idx]
                        # Determine direction
                        if node == 2 * e_idx:
                            paths_found.append([u, v])
                        else:
                            paths_found.append([v, u])
        
        # Add any remaining edges
        for i in range(m):
            if i not in visited_edges:
                u, v = edges[i]
                paths_found.append([u, v])
        
        return paths_found

    def merge_paths(paths, edges, adj, time_limit=20.0):
        """Merge paths that share endpoints to reduce count."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        changed = True
        iterations = 0
        max_iterations = 200

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() - start_time > time_limit:
                break

            endpoint_map = defaultdict(list)
            for i, path in enumerate(paths):
                endpoint_map[path[0]].append((i, 0))
                endpoint_map[path[-1]].append((i, 1))

            merged_any = False
            new_paths = list(paths)

            for vertex in sorted(endpoint_map.keys()):
                if time.time() - start_time > time_limit:
                    break
                entries = endpoint_map[vertex]
                if len(entries) < 2:
                    continue

                for a in range(len(entries)):
                    if merged_any:
                        break
                    for b in range(a + 1, len(entries)):
                        i, ei = entries[a]
                        j, ej = entries[b]
                        if i == j or i >= len(new_paths) or j >= len(new_paths):
                            continue

                        pi = new_paths[i]
                        pj = new_paths[j]

                        merged_path = None
                        if ei == 1 and ej == 0 and pi[-1] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[1:]
                        elif ei == 0 and ej == 1 and pi[0] == vertex and pj[-1] == vertex:
                            merged_path = pj + pi[1:]
                        elif ei == 1 and ej == 1 and pi[-1] == vertex and pj[-1] == vertex:
                            merged_path = pi + pj[::-1][1:]
                        elif ei == 0 and ej == 0 and pi[0] == vertex and pj[0] == vertex:
                            merged_path = pi + pj[::-1][1:]

                        if merged_path is None:
                            continue

                        if len(merged_path) != len(set(merged_path)):
                            continue
                        valid = True
                        for k in range(len(merged_path) - 1):
                            u, v = merged_path[k], merged_path[k + 1]
                            if tuple(sorted((u, v))) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            continue

                        remove_idx = sorted([i, j], reverse=True)
                        for idx in remove_idx:
                            new_paths.pop(idx)
                        new_paths.append(merged_path)
                        merged_any = True
                        changed = True
                        break
                if merged_any:
                    break

            paths = new_paths

        return paths

    def greedy_decompose(edges, adj, time_limit=60.0, rng_seed=0):
        """Fallback greedy decomposition."""
        rng = random.Random(rng_seed)
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths_found = []

        while remaining_edges:
            if time.time() - start_time > time_limit:
                break

            remaining_adj = defaultdict(set)
            for u, v in remaining_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)

            if not remaining_adj:
                break

            best_path = None
            best_len = 0

            candidates = sorted(remaining_adj.keys(),
                                key=lambda x: len(remaining_adj[x]),
                                reverse=True)
            candidates = candidates[:min(30, len(candidates))]

            for start in candidates:
                if time.time() - start_time > time_limit:
                    break
                path = _extend_path(start, remaining_adj, remaining_edges, rng, max_steps=50)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path

            all_vertices = list(remaining_adj.keys())
            random_starts = rng.sample(all_vertices, min(15, len(all_vertices)))
            for start in random_starts:
                if time.time() - start_time > time_limit:
                    break
                path = _extend_path(start, remaining_adj, remaining_edges, rng, max_steps=50)
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path

            if best_path is None or best_len < 2:
                e = next(iter(remaining_edges))
                best_path = list(e)

            paths_found.append(best_path)

            for k in range(len(best_path) - 1):
                u, v = best_path[k], best_path[k + 1]
                remaining_edges.discard(tuple(sorted((u, v))))

        for e in remaining_edges:
            paths_found.append(list(e))

        return paths_found

    def _extend_path(start, remaining_adj, remaining_edges, rng, max_steps=50):
        path = [start]
        visited = {start}
        current = start

        for _ in range(max_steps):
            neighbors = [nb for nb in remaining_adj.get(current, []) if nb not in visited]
            if not neighbors:
                break

            def score(nb):
                return len([x for x in remaining_adj.get(nb, []) if x not in visited])

            neighbors.sort(key=score)

            if rng.random() < 0.2 and len(neighbors) > 1:
                next_v = rng.choice(neighbors)
            else:
                next_v = neighbors[0]

            path.append(next_v)
            visited.add(next_v)
            current = next_v

        return path

    def validate_and_fix(paths, edges):
        """Validate paths: simple, use graph edges, edge-disjoint, cover all edges."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        valid_paths = []
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            # Check simplicity
            if len(path) != len(set(path)):
                # Try to fix by splitting
                for i in range(len(path) - 1):
                    if path[i] in path[i+1:]:
                        # Split here
                        sub1 = path[:i+1]
                        sub2 = path[i:]
                        valid_paths.append(sub1)
                        valid_paths.append(sub2)
                        break
                else:
                    valid_paths.append(path)
                continue
            
            # Check edges
            path_valid = True
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    path_valid = False
                    break
                used_edges.add(e)
            
            if path_valid:
                valid_paths.append(path)
        
        # Add any uncovered edges
        for e in edge_set:
            if e not in used_edges:
                valid_paths.append(list(e))
                used_edges.add(e)
        
        return valid_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]

    for n, p, seed in specs:
        if time.time() - start_time > deadline - 15:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        edges = instance(n, p, seed)
        adj = get_adjacency(edges)

        best_paths = None
        best_count = float('inf')

        # Try matching-based approach with different seeds
        for attempt_seed in range(5):
            if time.time() - start_time > deadline - 25:
                break
            match_paths = edge_end_matching_decompose(edges, adj, time_limit=deadline - 30, rng_seed=attempt_seed)
            merged_paths = merge_paths(match_paths, edges, adj, time_limit=deadline - 25)
            fixed_paths = validate_and_fix(merged_paths, edges)

            if len(fixed_paths) < best_count:
                best_count = len(fixed_paths)
                best_paths = fixed_paths

        # Try greedy as fallback/complement
        for attempt_seed in range(3):
            if time.time() - start_time > deadline - 20:
                break
            greedy_paths = greedy_decompose(edges, adj, time_limit=deadline - 30, rng_seed=attempt_seed)
            merged_paths = merge_paths(greedy_paths, edges, adj, time_limit=deadline - 25)
            fixed_paths = validate_and_fix(merged_paths, edges)

            if len(fixed_paths) < best_count:
                best_count = len(fixed_paths)
                best_paths = fixed_paths

        if best_paths is None:
            edges = instance(n, p, seed)
            best_paths = [list(e) for e in edges]

        results.append(best_paths)

    return results
