import time
import random
import sys
from collections import defaultdict

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

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []

        remaining_edges = set(edges)
        pieces = []

        def find_max_matching(edge_set):
            # Build adjacency for matching
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            
            if not adj:
                return {}

            # Hopcroft-Karp-like or simple augmenting path
            # Since graphs are small (n<=30), simple DFS augmenting path is fine
            match = {}
            
            def dfs_augment(u, visited):
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        if v not in match or dfs_augment(match[v], visited):
                            match[u] = v
                            match[v] = u
                            return True
                return False

            # Order vertices by degree descending to find larger matchings faster?
            # Actually, standard Kuhn's algorithm works.
            # To maximize cardinality, we just run it for all vertices.
            # Randomizing order can help find different matchings, but we want max cardinality.
            # Kuhn's algorithm finds max cardinality matching in bipartite graphs.
            # For general graphs, we need Edmonds' Blossom or a heuristic.
            # Given small size, we can use a simple greedy + local improvement or just greedy.
            # Let's try a simple greedy matching first, then try to improve.
            
            # Greedy matching
            edges_list = list(edge_set)
            random.shuffle(edges_list) # Randomize for diversity
            for u, v in edges_list:
                if u not in match and v not in match:
                    match[u] = v
                    match[v] = u
            
            # Try to improve with augmenting paths (simple version)
            # This is not guaranteed to find global max, but good enough
            # Actually, let's just use the greedy matching. It's fast and often good.
            # Or better: use a simple augmenting path search for a few iterations.
            
            # Let's stick to a robust simple method:
            # 1. Greedy
            # 2. Try to add edges that connect unmatched nodes if possible (not standard)
            
            # For now, greedy is sufficient for this task's scale.
            return match

        def match_to_cycles(match):
            cycles = []
            visited = set()
            for u in match:
                if u in visited:
                    continue
                path = [u]
                visited.add(u)
                current = u
                while match[current] != u:
                    current = match[current]
                    if current in visited:
                        break
                    path.append(current)
                    visited.add(current)
                
                # Check if it's a cycle
                if len(path) >= 3 and match[path[-1]] == path[0]:
                    cycles.append(path)
                elif len(path) == 2:
                    # This is just an edge, not a cycle
                    pass
            return cycles

        # Main loop
        while remaining_edges and time.time() < time_limit:
            # Find a matching
            match = find_max_matching(remaining_edges)
            
            # Convert matching to cycles
            cycles = match_to_cycles(match)
            
            if cycles:
                # Pick the longest cycle
                cycles.sort(key=len, reverse=True)
                best_cycle = cycles[0]
                pieces.append(best_cycle)
                
                # Remove edges of this cycle
                for i in range(len(best_cycle) - 1):
                    u, v = best_cycle[i], best_cycle[i+1]
                    e1 = (u, v)
                    e2 = (v, u)
                    if e1 in remaining_edges:
                        remaining_edges.remove(e1)
                    elif e2 in remaining_edges:
                        remaining_edges.remove(e2)
                u, v = best_cycle[-1], best_cycle[0]
                e1 = (u, v)
                e2 = (v, u)
                if e1 in remaining_edges:
                    remaining_edges.remove(e1)
                elif e2 in remaining_edges:
                    remaining_edges.remove(e2)
            else:
                # No cycles found in the matching, pick a single edge
                if remaining_edges:
                    edge = next(iter(remaining_edges))
                    pieces.append(list(edge))
                    remaining_edges.remove(edge)
                else:
                    break

        # If there are remaining edges, add them as single edges
        if time.time() < deadline - 2.0:
            for edge in remaining_edges:
                pieces.append(list(edge))
        
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        time_limit = time.time() + max(1.0, time_per_graph * 0.9)

        best_pieces = decompose_graph(edges, time_limit)
        results.append(best_pieces)

        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

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
