"""Decompositions of three fixed graphs into few cycles and edges (Erdős-Gallai)."""
import time
from random import Random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin under 120s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n, time_budget):
        """
        Decompose edge set into cycles and single edges, minimizing piece count.
        Strategy: repeatedly find long cycles in the remaining edge set,
        preferring high-degree vertices and longer cycles.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []
        remaining = set(edge_set)

        def find_cycle(start, visited_edges, adj_local):
            """
            DFS to find a cycle starting and ending at start.
            Returns list of vertices forming the cycle, or None.
            """
            # We look for a path from start back to start using only unvisited edges
            # This is a Hamiltonian-path-in-subgraph problem, but we use greedy DFS
            # with backtracking limited by time.
            
            best_cycle = None
            best_len = 0
            
            def dfs(node, path, used_edges):
                nonlocal best_cycle, best_len
                if time.time() > time_budget:
                    return
                if len(path) > best_len:
                    # Check if we can close the cycle
                    if node == start and len(path) >= 4:  # at least 3 distinct vertices + return
                        # Actually path[0] == start, so we need path[-1] to connect back to start
                        # The cycle is path[0], path[1], ..., path[-1], path[0]
                        # We need edge (path[-1], start) to exist and be unused
                        pass
                    # Record longest path found so far (even if not closed)
                    # But we only want cycles, so we need to close
                    pass
                
                # Try extending from node
                for neighbor in sorted(adj_local[node], key=lambda x: -len(adj_local[x])):
                    e = tuple(sorted((node, neighbor)))
                    if e in used_edges:
                        continue
                    if neighbor == start and len(path) >= 3:
                        # Found a cycle!
                        cycle = list(path)
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle[:]
                    elif neighbor not in path:
                        path.append(neighbor)
                        used_edges.add(e)
                        dfs(neighbor, path, used_edges)
                        used_edges.discard(e)
                        path.pop()
                        if time.time() > time_budget:
                            return
                
            dfs(start, [start], set())
            return best_cycle

        def find_cycle_greedy(start, adj_local, remaining_edges):
            """
            Greedy approach: build a path from start, trying to return to start.
            """
            # Use a modified DFS that prioritizes longer paths
            visited_nodes = set()
            used_edges = set()
            path = [start]
            visited_nodes.add(start)
            
            def extend(node, depth, max_depth=50):
                if time.time() > time_budget:
                    return None
                if depth > max_depth:
                    return None
                
                # Try to close the cycle
                if node != start and (tuple(sorted((node, start))) in remaining_edges and 
                    tuple(sorted((node, start))) not in used_edges):
                    if len(path) >= 3:
                        return list(path)
                
                # Try extending
                neighbors = list(adj_local[node])
                # Sort by degree (prefer high-degree) and by whether they're start
                neighbors.sort(key=lambda x: (-len(adj_local[x]), x == start))
                
                for neighbor in neighbors:
                    e = tuple(sorted((node, neighbor)))
                    if e in used_edges:
                        continue
                    if neighbor in visited_nodes and neighbor != start:
                        continue
                    
                    if neighbor == start and len(path) >= 3:
                        return list(path)
                    
                    path.append(neighbor)
                    visited_nodes.add(neighbor)
                    used_edges.add(e)
                    
                    result = extend(neighbor, depth + 1, max_depth)
                    if result is not None:
                        return result
                    
                    path.pop()
                    visited_nodes.discard(neighbor)
                    used_edges.discard(e)
                
                return None
            
            return extend(start, 0)

        while remaining:
            if time.time() > deadline:
                break
            
            # Find the best cycle to extract
            best_cycle = None
            best_len = 0
            
            # Try starting from high-degree vertices
            candidates = sorted(remaining, key=lambda e: -(len(adj[e[0]]) + len(adj[e[1]])))
            tried_starts = set()
            
            for u, v in candidates:
                if time.time() > deadline:
                    break
                for start in (u, v):
                    if start in tried_starts:
                        continue
                    tried_starts.add(start)
                    
                    cycle = find_cycle_greedy(start, adj, remaining)
                    if cycle is not None and len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                        if best_len >= 10:  # Good enough, stop searching
                            break
                if best_len >= 10:
                    break
            
            if best_cycle is None:
                # No cycle found; take a single edge
                e = next(iter(remaining))
                pieces.append(list(e))
                remaining.discard(e)
                adj[e[0]].discard(e[1])
                adj[e[1]].discard(e[0])
                continue
            
            # Extract the cycle
            cycle = best_cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle)
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Fallback: return remaining edges as single edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - 5)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
