"""Decompose fixed Erdős-Gallai instance graphs into few cycles and edges."""

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(n, edges):
        # Build adjacency and remaining edge set
        adj = [set() for _ in range(n)]
        remaining = set()
        deg = [0] * n
        
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            remaining.add((u, v) if u < v else (v, u))
            deg[u] += 1
            deg[v] += 1

        pieces = []

        def remove_edge(u, v):
            e = (u, v) if u < v else (v, u)
            if e in remaining:
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1

        def find_greedy_cycle(start):
            """
            Deterministic Greedy DFS to find a cycle.
            Always chooses the neighbor with the highest remaining degree.
            """
            # Stack elements: (node, path, on_path_set)
            # We use a simple iterative DFS
            stack = [(start, [start], {start})]
            best_cycle = None
            
            # Limit search to avoid explosion on dense subgraphs
            max_depth = n + 5
            steps = 0
            limit = 5000
            
            while stack and steps < limit:
                steps += 1
                node, path, on_path = stack.pop()
                
                # If path is too long, skip expansion (shouldn't happen in simple DFS without repeats, but safety)
                if len(path) > max_depth:
                    continue

                # Get neighbors sorted by degree descending
                nbrs = list(adj[node])
                nbrs.sort(key=lambda x: deg[x], reverse=True)
                
                # Check if we can close the cycle
                if start in nbrs and len(path) >= 3:
                    cyc = path[:]
                    if best_cycle is None or len(cyc) > len(best_cycle):
                        best_cycle = cyc
                    # We don't break immediately to see if a longer one exists, 
                    # but greedy usually finds good ones early. 
                    # However, to save time, if we found a very long cycle, we can stop.
                    if len(cyc) > n * 0.8:
                        return cyc

                # Extend to neighbors
                for w in nbrs:
                    if w == start:
                        continue # Already handled closing
                    if w in on_path:
                        continue # Would create a smaller cycle or non-simple, but we want simple cycles
                        # Actually, if w is in on_path, it forms a cycle. 
                        # But we are looking for a cycle containing `start`.
                        # If w is in on_path and w != start, we have a cycle not containing start?
                        # No, the path is from start to node. If we go to w in on_path, we close a cycle.
                        # The cycle would be path[index(w):] + [w].
                        # This cycle does NOT necessarily contain start unless w==start.
                        # We want cycles to remove edges. Any cycle is good.
                        # But our DFS is rooted at start. 
                        # Standard DFS for cycle finding usually returns the first cycle found.
                        # Here, we specifically want to find *a* cycle.
                        # If we find a cycle not containing start, it's still valid to remove.
                        # However, the logic above only checks `start in nbrs`.
                        # We should check if ANY neighbor is in on_path.
                        
                        # Let's refine: if w is in on_path, we found a cycle.
                        # The cycle is path from w to node, plus edge node-w.
                        # Let's record it.
                        if w in on_path:
                            idx = path.index(w)
                            cyc = path[idx:]
                            # The edge (node, w) closes it.
                            # The cycle vertices are path[idx], ..., node, w.
                            # But w is already at idx. So vertices are path[idx] to path[-1].
                            # Length is len(path) - idx.
                            if len(cyc) >= 3:
                                if best_cycle is None or len(cyc) > len(best_cycle):
                                    best_cycle = cyc
                                # Don't add to stack, this is a dead end for this branch
                                continue
                            
                    # Extend
                    new_path = path + [w]
                    new_on_path = on_path | {w}
                    stack.append((w, new_path, new_on_path))
                    
            return best_cycle

        # Main loop
        while remaining:
            # Find vertex with max degree >= 2
            max_deg = 0
            start_v = -1
            for v in range(n):
                if deg[v] > max_deg:
                    max_deg = deg[v]
                    start_v = v
            
            if max_deg < 2:
                break
            
            # Try to find a cycle starting from start_v
            cyc = find_greedy_cycle(start_v)
            
            if cyc is None:
                # If greedy from max degree fails, try other high degree vertices
                # Get top 5 candidates
                candidates = [v for v in range(n) if deg[v] >= 2]
                candidates.sort(key=lambda x: deg[x], reverse=True)
                found = False
                for v in candidates[:5]:
                    if v == start_v:
                        continue
                    cyc = find_greedy_cycle(v)
                    if cyc is not None:
                        found = True
                        break
                
                if not found:
                    # No cycles, remaining is a forest
                    for (u, v) in sorted(remaining):
                        pieces.append([u, v])
                    remaining.clear()
                    break
            
            # Add cycle
            pieces.append(cyc)
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                remove_edge(u, v)

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)
    return results
