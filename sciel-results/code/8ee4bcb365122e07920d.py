import time
import random
from random import Random

def construct():
    deadline = time.time() + 140  # Safety margin

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
        
        best_pieces = None
        best_count = float('inf')
        
        # Run multiple randomized greedy attempts
        for run in range(50):
            # Seed the random module for reproducibility within the run
            random.seed(42 + run)
            
            # Build static adjacency for the full graph
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            
            remaining = set(edges)
            pieces = []
            
            def find_long_cycle(rem_edges):
                if not rem_edges:
                    return None
                
                # Identify nodes in remaining graph
                nodes = set()
                for u, v in rem_edges:
                    nodes.add(u)
                    nodes.add(v)
                
                if not nodes:
                    return None

                node_list = list(nodes)
                random.shuffle(node_list)
                
                best_cycle = None
                best_len = 0
                
                # Try to find a long cycle starting from various nodes
                for start in node_list:
                    # Limit attempts per start node to balance speed and quality
                    for _ in range(20):
                        path = [start]
                        visited = {start}
                        
                        def dfs(node):
                            nonlocal best_cycle, best_len
                            for nb in adj.get(node, []):
                                edge = (min(node, nb), max(node, nb))
                                if edge not in rem_edges:
                                    continue
                                if nb == start and len(path) >= 3:
                                    if len(path) > best_len:
                                        best_len = len(path)
                                        best_cycle = path[:]
                                    return True
                                if nb not in visited:
                                    visited.add(nb)
                                    path.append(nb)
                                    dfs(nb)
                                    path.pop()
                                    visited.discard(nb)
                        
                        dfs(start)
                        # Early termination if we found a very long cycle
                        if best_len > 20:
                            return best_cycle
                    if best_len > 20:
                        return best_cycle
                        
                return best_cycle

            while remaining:
                cycle = find_long_cycle(remaining)
                if cycle:
                    # Extract the cycle
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        edge = (min(u, v), max(u, v))
                        remaining.discard(edge)
                    pieces.append(cycle)
                else:
                    # No cycles left, remaining are single edges
                    for e in sorted(remaining):
                        pieces.append([e[0], e[1]])
                    remaining = set()
                    break
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        return best_pieces if best_pieces else [list(e) for e in edges]

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Fallback to trivial decomposition if time runs out
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
