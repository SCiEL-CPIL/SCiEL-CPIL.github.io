"""Decompose three fixed random graphs into edge-disjoint cycles and single edges, minimizing total pieces."""

def construct():
    from random import Random
    import itertools

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Decompose a graph given by its edge list into edge-disjoint cycles and single edges.
        Strategy:
        1. Find all simple cycles up to a reasonable length limit (e.g., 7 or 8).
        2. Solve a Maximum Weighted Set Packing problem (or Integer Linear Programming)
           to select the maximum number of edge-disjoint cycles.
        3. The remaining edges are left as single-edge pieces.
        4. Return the list of pieces (cycles as vertex lists, edges as [u, v]).
        
        Since n is small (20-30), we can enumerate cycles up to a certain length.
        For larger graphs, we might need to be smarter, but for n<=30, this is feasible.
        """
        if not edges:
            return []

        edge_set = set(edges)
        n = max(max(u, v) for u, v in edges) + 1
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Enumerate all simple cycles of length 3 to MAX_CYCLE_LEN
        MAX_CYCLE_LEN = 7
        cycles = []
        seen_cycles = set()

        # Use DFS to find cycles. To avoid duplicates, only start from the smallest vertex in the cycle.
        for start in range(n):
            # DFS from start, only visiting vertices > start (except when closing the cycle back to start)
            stack = [(start, [start], set([start]))]
            while stack:
                node, path, visited = stack.pop()
                for nei in adj[node]:
                    if nei == start and len(path) >= 3:
                        # Found a cycle
                        cycle = tuple(sorted(path))  # Normalize by sorting? No, order matters for the cycle structure.
                        # Better normalization: rotate so min vertex is first, then reverse if needed to ensure consistent direction
                        # Actually, for set packing, we just need a unique representation of the set of edges.
                        cycle_edges = frozenset()
                        for i in range(len(path)):
                            u, v = path[i], path[(i + 1) % len(path)]
                            cycle_edges.add(tuple(sorted((u, v))))
                        if cycle_edges not in seen_cycles:
                            seen_cycles.add(cycle_edges)
                            cycles.append(path)
                    elif nei > start and nei not in visited and len(path) < MAX_CYCLE_LEN:
                        new_visited = visited | {nei}
                        stack.append((nei, path + [nei], new_visited))

        # We have a list of cycles. Each cycle is a list of vertices.
        # Convert to edge sets for packing.
        cycle_edge_sets = []
        for c in cycles:
            es = frozenset()
            for i in range(len(c)):
                u, v = c[i], c[(i + 1) % len(c)]
                es.add(tuple(sorted((u, v))))
            cycle_edge_sets.append(es)

        # Now solve the maximum edge-disjoint cycle packing.
        # This is a Set Packing problem. Since the number of cycles is not huge for n=30,
        # we can use a greedy approach with backtracking or a simple ILP solver if available.
        # However, standard library doesn't have ILP. We can use a greedy heuristic that works well:
        # Sort cycles by length (shorter first? or longer first?) 
        # Actually, to maximize the *number* of cycles, we should prefer shorter cycles?
        # No, the objective is to minimize total pieces = (total_edges - sum(len(cycle))) / 1 + num_cycles
        # Wait. Total pieces = num_cycles + (total_edges - sum(edges_in_cycles))
        # We want to maximize sum(edges_in_cycles) - num_cycles.
        # Each cycle of length L contributes L edges and 1 piece. So it saves L-1 pieces compared to L single edges.
        # So we want to maximize the sum of (L-1) over all selected cycles, subject to edge-disjointness.
        
        # Let's use a greedy algorithm:
        # 1. Calculate "gain" for each cycle: len(cycle) - 1.
        # 2. Sort by gain descending.
        # 3. Greedily pick cycles if they don't conflict with already picked ones.
        # This is a heuristic, but often good. For n=30, the number of cycles is manageable.
        # To improve, we can try a local search or use a more sophisticated packing.
        # Given the constraints, let's try a greedy with random restarts or just a single greedy pass.
        # Actually, a better approach for small instances is to use a Maximum Matching in an auxiliary graph?
        # No, Set Packing is NP-hard.
        
        # Let's try a simple ILP-like approach using brute force if the number of cycles is small,
        # or a greedy with backtracking.
        # For now, let's use a greedy algorithm sorted by gain/length ratio or just gain.
        
        # Let's refine the greedy:
        # Sort cycles by (len(c)-1) descending.
        selected = []
        used_edges = set()
        
        # Sort by gain descending
        sorted_cycles = sorted(range(len(cycles)), key=lambda i: len(cycles[i]) - 1, reverse=True)
        
        for idx in sorted_cycles:
            c_edges = cycle_edge_sets[idx]
            if c_edges.isdisjoint(used_edges):
                selected.append(idx)
                used_edges.update(c_edges)
                
        # Construct the result
        pieces = []
        for idx in selected:
            pieces.append(cycles[idx][:])  # Copy the cycle
        for e in edge_set:
            if e not in used_edges:
                pieces.append(list(e))
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
        
    return results

# EVOLVE-BLOCK-END
