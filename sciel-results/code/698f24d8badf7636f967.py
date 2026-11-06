"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(set)
        remaining = set()
        for e in edges:
            c = canonical(e)
            remaining.add(c)
            adj[c[0]].add(c[1])
            adj[c[1]].add(c[0])
        
        pieces = []
        
        def try_find_cycle(start, adj, remaining, max_len=20):
            """
            Find a simple cycle starting and ending at 'start'.
            Uses randomized DFS with backtracking.
            Returns list of vertices in the cycle (excluding the final return to start) or None.
            """
            # We need a path from start to some neighbor of start, not using start in between.
            # Let's do a DFS from start.
            
            # To make it deterministic-ish but varied, we can shuffle neighbors
            # But for speed, let's just try a few random walks
            
            best_cycle = None
            
            # Try a few times with different randomizations
            for _ in range(10):
                visited = {start}
                path = [start]
                # We need to find a path from start to a neighbor of start
                # that doesn't visit start again.
                
                # Random DFS
                stack = [(start, list(adj[start]))]
                # Actually, simple recursive DFS is easier to manage for simple cycles
                # Let's do a BFS/DFS to find a cycle containing start.
                
                # Simpler: Find any cycle in the remaining graph.
                # If it contains start, great. If not, we can't use it for start.
                # But we want to minimize pieces.
                # A good heuristic: find the longest cycle possible.
                
                # Let's try to find a cycle using a randomized walk that tries to close.
                current = start
                path = [start]
                visited = {start}
                
                while True:
                    neighbors = [n for n in adj[current] if n not in visited or n == start]
                    # Filter out visited, but allow start if path length >= 2
                    if len(path) >= 2:
                        if start in neighbors:
                            # We found a cycle
                            if best_cycle is None or len(path) > len(best_cycle):
                                best_cycle = list(path)
                            break
                        neighbors = [n for n in neighbors if n != start]
                    else:
                        neighbors = [n for n in neighbors if n != start]
                    
                    if not neighbors:
                        break
                    
                    # Choose next node. Prefer high degree or random.
                    # Shuffle to avoid bias
                    random.shuffle(neighbors)
                    next_node = neighbors[0]
                    
                    # If we are at a dead end, we should backtrack.
                    # For simplicity in this heuristic, if we get stuck, we restart the walk.
                    # This is inefficient but okay for small graphs.
                    # Let's add a simple backtracking mechanism.
                    
                    # Actually, let's implement a proper DFS to find a cycle.
                    # This is better.
                    break # Exit random walk, use DFS below
                
                # DFS to find a cycle
                # We want a simple cycle.
                # Let's find a cycle in the graph.
                
                def dfs_find_cycle(u, start_node, visited, path):
                    for v in adj[u]:
                        if v == start_node and len(path) >= 2:
                            return list(path)
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            res = dfs_find_cycle(v, start_node, visited, path)
                            if res:
                                return res
                            path.pop()
                            visited.remove(v)
                    return None

                visited = {start}
                res = dfs_find_cycle(start, start, visited, [start])
                if res:
                    if best_cycle is None or len(res) > len(best_cycle):
                        best_cycle = res
            
            return best_cycle

        def find_longest_cycle_heuristic(adj, remaining):
            """
            Find a long cycle in the graph.
            Strategy:
            1. Try to find a cycle containing each vertex with highest degree.
            2. Keep the longest one found.
            """
            best_cycle = None
            vertices = sorted(adj.keys(), key=lambda x: -len(adj[x]))
            
            # Limit number of starts to save time
            for v in vertices[:10]:
                if not adj[v]:
                    continue
                # Try to find a cycle through v
                # We can use the DFS approach
                visited = {v}
                path = [v]
                
                # Randomized DFS to find a long cycle
                # To find a LONG cycle, we should prefer paths that don't close early.
                # This is hard. Let's just find ANY cycle through v, and maybe try to extend?
                # Or just find a cycle.
                
                # Let's try a few random DFS orders
                for _ in range(5):
                    # Shuffle neighbors in a copy? No, we can just shuffle the list we iterate over.
                    # We need to be careful with state.
                    
                    def dfs(u, start_node, visited, path):
                        # Try neighbors in random order
                        nbrs = list(adj[u])
                        random.shuffle(nbrs)
                        for v in nbrs:
                            if v == start_node and len(path) >= 2:
                                return list(path)
                            if v not in visited:
                                visited.add(v)
                                path.append(v)
                                res = dfs(v, start_node, visited, path)
                                if res:
                                    return res
                                path.pop()
                                visited.remove(v)
                        return None

                    visited = {v}
                    res = dfs(v, v, visited, [v])
                    if res:
                        if best_cycle is None or len(res) > len(best_cycle):
                            best_cycle = res
                            
            return best_cycle

        # Main loop: Extract cycles until no edges remain
        while remaining:
            if time.time() - t0 > DEADLINE - 5:
                break
                
            # Find a cycle
            cycle = find_longest_cycle_heuristic(adj, remaining)
            
            if cycle and len(cycle) >= 3:
                # Verify cycle edges are in remaining
                valid = True
                for i in range(len(cycle)):
                    e = canonical((cycle[i], cycle[(i + 1) % len(cycle)]))
                    if e not in remaining:
                        valid = False
                        break
                
                if valid:
                    pieces.append(cycle)
                    # Remove edges
                    for i in range(len(cycle)):
                        e = canonical((cycle[i], cycle[(i + 1) % len(cycle)]))
                        remaining.remove(e)
                        adj[cycle[i]].remove(cycle[(i + 1) % len(cycle)])
                        adj[cycle[(i + 1) % len(cycle)]].remove(cycle[i])
                        if not adj[cycle[i]]:
                            del adj[cycle[i]]
                        if not adj[cycle[(i + 1) % len(cycle)]]:
                            del adj[cycle[(i + 1) % len(cycle)]]
                else:
                    # Should not happen if logic is correct, but fallback
                    break
            else:
                # No cycle found, remaining edges are all bridges (forest)
                # Add them as single edges
                for e in remaining:
                    pieces.append(list(e))
                    adj[e[0]].remove(e[1])
                    adj[e[1]].remove(e[0])
                    if not adj[e[0]]:
                        del adj[e[0]]
                    if not adj[e[1]]:
                        del adj[e[1]]
                remaining = set()
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
