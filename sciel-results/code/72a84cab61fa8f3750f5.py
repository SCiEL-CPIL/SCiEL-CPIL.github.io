"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Instance generator copied from the verifier
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n):
        """
        Finds a relatively long cycle using a greedy heuristic with random restarts.
        """
        best_cycle = []
        restarts = 1000
        
        for _ in range(restarts):
            if time.time() > deadline:
                break
            start = random.randint(0, n - 1)
            if not adj[start]:
                continue
                
            path = [start]
            visited = {start}
            
            # Greedy expansion
            while True:
                current = path[-1]
                # Sort neighbors by degree (descending) to prefer high-degree nodes
                # This often leads to longer paths/cycles in random graphs
                neighbors = sorted(adj[current], key=lambda x: len(adj[x]), reverse=True)
                
                next_node = None
                for v in neighbors:
                    if v == path[0] and len(path) >= 3:
                        # Found a cycle
                        next_node = v
                        break
                    if v not in visited:
                        next_node = v
                        break
                        
                if next_node is None:
                    break # Stuck
                    
                if next_node == path[0]:
                    # Cycle found
                    if len(path) > len(best_cycle):
                        best_cycle = path
                    break
                else:
                    path.append(next_node)
                    visited.add(next_node)
                    
            if len(best_cycle) == n:
                break
                
        return best_cycle if best_cycle else None

    def decompose(edges, n):
        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: add all remaining edges as single pieces
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
            
            # Rebuild adj for remaining edges
            rem_adj = {i: [] for i in range(n)}
            for u, v in remaining_edges:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
                
            cycle = find_longest_cycle(rem_adj, n)
            
            if cycle:
                # Add cycle to pieces
                pieces.append(cycle)
                # Remove cycle edges from remaining_edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    remaining_edges.discard(edge)
            else:
                # No cycle found
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
                
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            pass
        
        edges = instance(*spec)
        n = spec[0]
        pieces = decompose(edges, n)
        results.append(pieces)
        
    return results

# EVOLVE-BLOCK-END
