import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before the 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_best_cycle(adj, rng):
        """
        Finds a long cycle in the graph using iterative DFS with proper cycle extraction.
        """
        candidates = [u for u, neighbors in adj.items() if len(neighbors) > 1]
        if not candidates:
            return None

        best_cycle = None
        best_len = 0

        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        starts = candidates[:min(8, len(candidates))]
        if len(candidates) > 8:
            starts += rng.sample(candidates, min(8, len(candidates) - 8))
        starts = list(set(starts))
        rng.shuffle(starts)

        for start in starts:
            color = {}
            parent = {}
            path = []
            path_index = {}
            local_best_cycle = None
            local_best_len = 0

            # Iterative DFS
            stack = [(start, iter(adj[start]))]
            color[start] = 1
            parent[start] = -1
            path.append(start)
            path_index[start] = 0

            while stack:
                u, neighbors_iter = stack[-1]
                found_next = False
                for w in neighbors_iter:
                    if w == parent[u]:
                        continue
                    if color.get(w, 0) == 0:
                        color[w] = 1
                        parent[w] = u
                        path.append(w)
                        path_index[w] = len(path) - 1
                        stack.append((w, iter(adj[w])))
                        found_next = True
                        break
                    elif color.get(w, 0) == 1:
                        # Back edge: w is on current path
                        idx = path_index.get(w)
                        if idx is not None:
                            cyc = path[idx:]
                            if len(cyc) > local_best_len:
                                local_best_len = len(cyc)
                                local_best_cycle = list(cyc)
                if not found_next:
                    stack.pop()
                    color[u] = 2
                    if path and path[-1] == u:
                        path.pop()
                        if path_index:
                            del path_index[u]

            if local_best_cycle and local_best_len > best_len:
                best_len = local_best_len
                best_cycle = local_best_cycle
            if best_len >= 25:
                break

        return best_cycle

    def try_merge_pieces(pieces, edges_set):
        """Try to merge adjacent single-edge pieces into cycles."""
        # Build edge to piece index mapping
        edge_to_piece = {}
        for i, piece in enumerate(pieces):
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                edge_to_piece[e] = i

        # Try to find cycles formed by consecutive single edges
        improved = True
        max_iters = 50
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            new_pieces = list(pieces)
            used = [False] * len(pieces)

            for i in range(len(pieces)):
                if used[i] or len(pieces[i]) != 2:
                    continue
                u, v = pieces[i]
                # Try to extend from u and v
                # Check if there's a path back
                # Simple approach: look for another single edge that shares a vertex
                # and check if we can form a triangle or longer cycle
                # For now, just try to find a cycle of length 3-5
                pass

            # More effective: look for cycles in the subgraph of single-edge pieces
            single_edges = set()
            for i, piece in enumerate(pieces):
                if len(piece) == 2 and not used[i]:
                    single_edges.add((min(piece[0], piece[1]), max(piece[0], piece[1])))

            if len(single_edges) < 3:
                break

            # Build adjacency for single edges
            single_adj = defaultdict(set)
            for u, v in single_edges:
                single_adj[u].add(v)
                single_adj[v].add(u)

            # Find cycles in this subgraph
            visited = set()
            for start in list(single_adj.keys()):
                if start in visited:
                    continue
                # DFS to find cycle
                stack = [(start, [start], {start})]
                while stack:
                    node, path, seen = stack.pop()
                    for nb in single_adj[node]:
                        if nb == path[0] and len(path) >= 3:
                            # Found a cycle
                            cyc = list(path)
                            # Remove these edges from single_edges
                            for j in range(len(cyc)):
                                a, b = cyc[j], cyc[(j + 1) % len(cyc)]
                                e = (min(a, b), max(a, b))
                                if e in single_edges:
                                    single_edges.discard(e)
                                    # Mark the piece as used
                                    if e in edge_to_piece:
                                        used[edge_to_piece[e]] = True
                            # Add the cycle as a piece
                            new_pieces.append(list(cyc))
                            improved = True
                            break
                        elif nb not in seen:
                            stack.append((nb, path + [nb], seen | {nb}))
                visited.add(start)

            if not single_edges:
                break

            # Rebuild pieces list
            if improved:
                pieces = [p for i, p in enumerate(new_pieces) if not (len(p) == 2 and used[i])]
                # Actually this is getting complex, let's use a simpler approach

        return pieces

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_pieces = None
        best_count = float('inf')
        
        # Adaptive restarts based on time
        max_restarts = 300
        
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            rng = random.Random(i * 7 + 3)
            current_adj = {u: set(v) for u, v in adj.items()}
            current_edges = set(edges)
            pieces = []
            
            while True:
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                if not any(len(neigh) > 0 for neigh in current_adj.values()):
                    break
                
                cycle = find_best_cycle(current_adj, rng)
                if cycle is None:
                    break
                
                for j in range(len(cycle)):
                    a, b = cycle[j], cycle[(j + 1) % len(cycle)]
                    e = (min(a, b), max(a, b))
                    current_edges.discard(e)
                    if b in current_adj[a]:
                        current_adj[a].discard(b)
                    if a in current_adj[b]:
                        current_adj[b].discard(a)
                
                pieces.append(list(cycle))
            
            for e in current_edges:
                pieces.append([e[0], e[1]])
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            return [[u, v] for u, v in edges]
            
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
