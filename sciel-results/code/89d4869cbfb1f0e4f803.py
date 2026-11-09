"""Decompose fixed Erdős-Gallai instance graphs into few cycles and edges."""

def construct():
    from random import Random
    import sys

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(n, edges):
        # Build adjacency and remaining edge set
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        remaining = set(edge_set)
        pieces = []

        def has_edge(a, b):
            e = (a, b) if a < b else (b, a)
            return e in remaining

        def remove_edge(a, b):
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)

        def rem_degree(v):
            return sum(1 for w in adj[v] if has_edge(v, w))

        # Find bridges using Tarjan's algorithm or simple DFS
        # Since n is small, we can use a simple check:
        # An edge (u,v) is a bridge if there is no path from u to v avoiding (u,v).
        # We can do this for each edge, or use a global bridge finding.
        # Given n<=30, O(E * (V+E)) is fine.
        
        def is_bridge(u, v):
            # Temporarily remove edge
            remove_edge(u, v)
            # BFS from u to see if v is reachable
            visited = {u}
            queue = [u]
            while queue:
                node = queue.pop(0)
                for w in adj[node]:
                    if not has_edge(node, w):
                        continue
                    if w == v:
                        remove_edge(u, v) # Restore
                        return False
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
            remove_edge(u, v) # Restore
            return True

        # Identify bridges and non-bridges
        bridges = []
        non_bridges = []
        for (u, v) in sorted(edge_set):
            if is_bridge(u, v):
                bridges.append((u, v))
            else:
                non_bridges.append((u, v))
        
        # Bridges are single-edge pieces
        for (u, v) in bridges:
            pieces.append([u, v])
            remove_edge(u, v)
            
        # Now decompose the remaining graph (which has no bridges)
        # We will repeatedly find cycles.
        # To improve, we will generate multiple candidate cycles and pick the best one.
        
        def find_candidate_cycles(start, num_candidates=10):
            """Find several cycles starting from start using randomized DFS."""
            candidates = []
            for _ in range(num_candidates):
                path = [start]
                onpath = {start}
                current = start
                success = False
                max_steps = 100
                for _ in range(max_steps):
                    nbrs = [w for w in adj[current] if has_edge(current, w)]
                    if not nbrs:
                        break
                    if start in nbrs and len(path) >= 3:
                        candidates.append(path[:])
                        success = True
                        break
                    # Filter neighbors
                    ext = [w for w in nbrs if w not in onpath]
                    if not ext:
                        break
                    # Random choice
                    ext.sort(key=lambda w: rem_degree(w), reverse=True)
                    # Weighted random
                    r = Random().random()
                    if r < 0.7:
                        next_node = ext[0]
                    elif r < 0.9:
                        next_node = ext[1] if len(ext) > 1 else ext[0]
                    else:
                        next_node = ext[Random().randint(0, len(ext)-1)]
                    current = next_node
                    path.append(current)
                    onpath.add(current)
                if success:
                    # We found a cycle. 
                    # We can try to extend it? No, simple cycle.
                    pass
            return candidates

        def get_best_cycle():
            """Find the best cycle in the current remaining graph."""
            # Try from all vertices with degree >= 2
            candidates = [v for v in range(n) if rem_degree(v) >= 2]
            if not candidates:
                return None
            
            best_cycle = None
            best_len = 0
            
            # Try a few random starts to avoid bias
            Random().shuffle(candidates)
            
            for start in candidates[:20]:
                # Generate multiple candidates from this start
                cands = find_candidate_cycles(start, num_candidates=5)
                for c in cands:
                    if len(c) > best_len:
                        best_len = len(c)
                        best_cycle = c
            
            # If we didn't find a good one, try a more systematic DFS
            if best_cycle is None or best_len < 3:
                # Fallback: simple DFS
                for start in candidates:
                    path = [start]
                    onpath = {start}
                    stack = [(start, 0)] # node, next neighbor index
                    # This is a bit complex, let's just use the randomized one more aggressively
                    pass
            
            return best_cycle

        while remaining:
            # Check if any cycles exist
            # A cycle exists if there is a vertex with degree >= 2 in the remaining graph
            # (and the component is 2-edge-connected, which it is by construction)
            
            best_cycle = get_best_cycle()
            
            if best_cycle is None:
                # No cycles found. This shouldn't happen if we correctly identified non-bridges,
                # but if the graph became disconnected or something went wrong, 
                # or if we are left with a single edge (which is a bridge, but we removed them initially?
                # No, removing edges can create new bridges.
                # So we must re-evaluate bridges?
                # Or simply, if no cycle is found, the remaining edges are bridges.
                for (u, v) in sorted(remaining):
                    pieces.append([u, v])
                remaining.clear()
                break

            # Add the cycle
            pieces.append(best_cycle)
            for i in range(len(best_cycle)):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % len(best_cycle)]
                remove_edge(a, b)

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)
    return results
