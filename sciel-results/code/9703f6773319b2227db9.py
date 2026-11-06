import time
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    for i, spec in enumerate(specs):
        time_left = deadline - time.time()
        if time_left <= 0:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        paths = []

        def extract_path_beam(beam_width=5, max_depth=100):
            """Beam search to find a long simple path in remaining_edges."""
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            if not active_vertices:
                return None

            vertices_list = list(active_vertices)
            random.shuffle(vertices_list)

            best_path = None

            # Limit starts based on time
            max_starts = min(30, len(vertices_list))

            for start in vertices_list[:max_starts]:
                # Check if start has any remaining edge
                has_edge = False
                for n in adj[start]:
                    e = (min(start, n), max(start, n))
                    if e in remaining_edges:
                        has_edge = True
                        break
                if not has_edge:
                    continue

                # Beam search
                # Each beam entry: (path, visited_set, current_node)
                beam = [([start], {start}, start)]

                for depth in range(max_depth):
                    if time.time() > deadline - 2.0:
                        break
                    new_beam = []
                    for path, visited, node in beam:
                        # Find valid neighbors
                        valid_neighbors = []
                        for n in adj[node]:
                            if n in visited:
                                continue
                            e = (min(node, n), max(node, n))
                            if e in remaining_edges:
                                valid_neighbors.append(n)

                        if not valid_neighbors:
                            # Dead end; keep as candidate
                            if best_path is None or len(path) > len(best_path):
                                best_path = path[:]
                            continue

                        # Score neighbors: prefer those with fewer options (Warnsdorff)
                        scored = []
                        for n in valid_neighbors:
                            count = 0
                            for m in adj[n]:
                                if m in visited:
                                    continue
                                e = (min(n, m), max(n, m))
                                if e in remaining_edges:
                                    count += 1
                            scored.append((count, n))
                        scored.sort()

                        # Take top beam_width neighbors for this beam entry
                        for _, n in scored[:beam_width]:
                            new_path = path + [n]
                            new_visited = visited | {n}
                            new_beam.append((new_path, new_visited, n))

                    if not new_beam:
                        break

                    # Keep only top beam_width entries by path length
                    new_beam.sort(key=lambda x: -len(x[0]))
                    beam = new_beam[:beam_width]

                    # Update best if any beam entry is longer
                    if beam:
                        longest = max(len(p) for p, v, n in beam)
                        if best_path is None or longest > len(best_path):
                            best_path = beam[0][0][:]

                # After beam search, check final beam entries
                for path, visited, node in beam:
                    if best_path is None or len(path) > len(best_path):
                        best_path = path[:]

            return best_path

        # Phase 1: Exact minimum path decomposition via bipartite matching
        # A path decomposition of a graph corresponds to a matching in the
        # bipartite graph where left/right copies of vertices are connected
        # by edges. Minimize number of paths = maximize matching size.
        
        # Build bipartite adjacency for matching
        # Left node i represents vertex i as start, right node i as end
        # Edge (u,v) in original graph -> edge from left_u to right_v and left_v to right_u
        n_vertices = max(max(u, v) for u, v in edges) + 1 if edges else 0
        
        # Create cost matrix for Hungarian algorithm
        # We want to maximize the number of matched edges (minimize paths)
        # Cost matrix: 0 if edge exists, large cost otherwise
        INF_COST = 10000
        cost_matrix = [[INF_COST] * n_vertices for _ in range(n_vertices)]
        
        for u, v in edges:
            cost_matrix[u][v] = 0
            cost_matrix[v][u] = 0
        
        # Run Hungarian algorithm to find minimum-cost maximum matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # The matched pairs (row_ind[i], col_ind[i]) where cost is 0
        # represent directed edges in our path decomposition
        # Build the directed graph of matched edges
        matched_edges = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] == 0:
                matched_edges.add((r, c))
        
        # Build adjacency for path reconstruction
        # out_edges[u] = list of v such that (u,v) is matched
        out_edges = defaultdict(list)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for u, v in matched_edges:
            out_edges[u].append(v)
            out_degree[u] += 1
            in_degree[v] += 1
        
        # Find all paths: start nodes have in_degree=0, end nodes have out_degree=0
        # Each path starts at a node with in_degree=0 and follows matched edges
        start_nodes = [v for v in range(n_vertices) if in_degree[v] == 0]
        
        # Reconstruct paths
        for start in start_nodes:
            if out_degree[start] == 0:
                # Isolated vertex, no path
                continue
            path = [start]
            current = start
            while out_degree[current] > 0:
                current = out_edges[current][0]
                path.append(current)
                out_degree[current] -= 1
            paths.append(path)
        
        # Verify all edges are covered
        covered_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                covered_edges.add((min(u, v), max(u, v)))
        
        # If any edges are not covered (shouldn't happen with perfect matching logic),
        # add them as single-edge paths
        for e in edges:
            if e not in covered_edges:
                paths.append(list(e))

        results.append(paths)
        
        # Debug: print path count for verification
        # print(f"Graph {i}: {len(paths)} paths, {len(edges)} edges")

    return results
