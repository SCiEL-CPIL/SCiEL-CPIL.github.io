"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []

        remaining = set(edges)
        pieces = []

        def find_cycles_in_2matching(edge_set):
            """Find all cycles in a 2-regular subgraph (each vertex degree <= 2)."""
            if not edge_set:
                return []
            adj = {}
            for u, v in edge_set:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            # Find all cycles in this 2-regular graph
            visited = set()
            cycles = []
            for start in adj:
                if start in visited:
                    continue
                # Follow the path
                path = [start]
                visited.add(start)
                current = start
                while True:
                    neighbors = [nb for nb in adj.get(current, []) if nb not in visited]
                    if not neighbors:
                        # Check if we can close a cycle back to start
                        if start in adj.get(current, []) and len(path) >= 3:
                            cycles.append(path[:])
                        break
                    next_node = neighbors[0]
                    if next_node == start and len(path) >= 3:
                        cycles.append(path[:])
                        break
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
            return cycles

        def extract_2matching_cycles(rem):
            """Extract cycles from a 2-matching of rem. Returns (cycles, used_edges)."""
            if len(rem) < 3:
                return [], set()
            
            # Build adjacency
            adj = {}
            deg = {}
            for u, v in rem:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            # Greedy 2-matching: for each vertex, keep at most 2 edges
            # Process vertices in order of degree (highest first)
            vertices = sorted(deg.keys(), key=lambda x: -deg[x])
            selected = set()
            vertex_count = {}
            
            for v in vertices:
                if vertex_count.get(v, 0) >= 2:
                    continue
                # Sort neighbors by degree (prefer connecting to high-degree vertices)
                neighbors = sorted(adj.get(v, []), key=lambda nb: -deg.get(nb, 0))
                for nb in neighbors:
                    if vertex_count.get(v, 0) >= 2:
                        break
                    if vertex_count.get(nb, 0) >= 2:
                        continue
                    edge = (min(v, nb), max(v, nb))
                    if edge in rem and edge not in selected:
                        selected.add(edge)
                        vertex_count[v] = vertex_count.get(v, 0) + 1
                        vertex_count[nb] = vertex_count.get(nb, 0) + 1
            
            # Extract cycles from the selected 2-matching
            cycles = find_cycles_in_2matching(selected)
            return cycles, selected

        def find_longest_cycle_dfs(rem):
            """Find a long cycle using DFS with backtracking."""
            if len(rem) < 3:
                return None
            
            adj = {}
            for u, v in rem:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            
            # Find vertices with degree >= 2
            deg = {}
            for u, v in rem:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            candidates = [v for v in deg if deg[v] >= 2]
            if not candidates:
                return None
            
            # Sort candidates by degree descending
            candidates.sort(key=lambda x: -deg[x])
            
            best_cycle = None
            
            for start in candidates[:min(10, len(candidates))]:  # Limit starts
                # DFS to find longest cycle through start
                def dfs(current, path, visited):
                    nonlocal best_cycle
                    if len(path) >= 3 and start in adj.get(current, []):
                        cycle = path[:]
                        if best_cycle is None or len(cycle) > len(best_cycle):
                            best_cycle = cycle[:]
                        return
                    for nb in adj.get(current, []):
                        if nb == start and len(path) >= 3:
                            cycle = path[:]
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle[:]
                            continue
                        if nb not in visited:
                            path.append(nb)
                            visited.add(nb)
                            dfs(nb, path, visited)
                            path.pop()
                            visited.discard(nb)
                
                dfs(start, [start], {start})
            
            return best_cycle

        # Main loop: repeatedly extract cycles
        while remaining:
            if time.time() > deadline - 5:
                break
            
            # Strategy 1: Try 2-matching extraction
            cycles, used_edges = extract_2matching_cycles(remaining)
            
            if cycles:
                # Take the longest cycle
                cycles.sort(key=lambda c: -len(c))
                best_cycle = cycles[0]
                # Remove edges of this cycle
                for i in range(len(best_cycle) - 1):
                    u, v = best_cycle[i], best_cycle[i + 1]
                    edge = (min(u, v), max(u, v))
                    remaining.discard(edge)
                pieces.append(best_cycle[:])
            else:
                # Strategy 2: Try DFS for longest cycle
                cycle = find_longest_cycle_dfs(remaining)
                if cycle:
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i + 1]
                        edge = (min(u, v), max(u, v))
                        remaining.discard(edge)
                    pieces.append(cycle[:])
                else:
                    # No more cycles
                    for e in sorted(remaining):
                        pieces.append([e[0], e[1]])
                    remaining = set()
                    break
        
        if remaining:
            for e in sorted(remaining):
                pieces.append([e[0], e[1]])
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
