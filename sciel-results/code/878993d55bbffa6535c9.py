import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        remaining = set()
        for u, v in edges:
            remaining.add((min(u, v), max(u, v)))
        
        pieces = []
        rng = Random(12345)
        
        # We will greedily extract cycles.
        # To maximize score, we want to find the longest possible cycles first.
        
        while remaining:
            if time.time() > deadline - 5:
                break
            
            # Find longest cycle in the remaining graph
            best_cycle = None
            best_len = 0
            
            # Get adjacency list for remaining graph
            adj = {i: set() for i in range(n)}
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            # Vertices with degree < 2 cannot be in a cycle
            valid_starts = [v for v in range(n) if len(adj[v]) >= 2]
            
            if not valid_starts:
                break
                
            # Randomized DFS to find long cycles
            # We try multiple random starts and random neighbor orders
            num_attempts = 100
            for attempt in range(num_attempts):
                if time.time() > deadline - 5:
                    break
                    
                # Pick a random start vertex
                start = rng.choice(valid_starts)
                
                # DFS state
                visited = {start}
                path = [start]
                parent = {start: -1}
                
                def dfs(node):
                    nonlocal best_cycle, best_len
                    if time.time() > deadline - 5:
                        return False
                    
                    # Pruning: even if we visit all remaining nodes, can we beat best?
                    # This is a weak bound, but better than nothing
                    # Actually, just let it run, n is small (<=30)
                    
                    for nei in adj[node]:
                        if nei == start and len(path) >= 3:
                            # Found a cycle
                            cycle = list(path)
                            # cycle is start -> ... -> node -> start
                            # The path is [start, ..., node]
                            # The cycle edges are (start, p1), ..., (node, start)
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle[:]
                            return True
                        elif nei not in visited:
                            visited.add(nei)
                            path.append(nei)
                            parent[nei] = node
                            
                            if dfs(nei):
                                return True
                            
                            # Backtrack
                            path.pop()
                            visited.remove(nei)
                            del parent[nei]
                    return False
                
                # Shuffle neighbors for randomization? 
                # Python sets are unordered, but iteration order is somewhat stable.
                # To ensure randomness, we can create a list and shuffle.
                # But doing this inside DFS is expensive.
                # Instead, we rely on different start nodes and the fact that 
                # the graph structure might change if we removed edges, but here 
                # we are just searching the same graph.
                
                # To get different paths, we can shuffle the order in which we 
                # pick the start, or shuffle the adjacency lists before DFS.
                
                # Let's shuffle the adjacency lists for this attempt
                shuffled_adj = {v: list(u) for v, u in adj.items()}
                for v in shuffled_adj:
                    rng.shuffle(shuffled_adj[v])
                
                # Redefine dfs to use shuffled_adj
                def dfs_shuffled(node):
                    nonlocal best_cycle, best_len
                    if time.time() > deadline - 5:
                        return False
                    
                    for nei in shuffled_adj[node]:
                        if nei == start and len(path) >= 3:
                            cycle = list(path)
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle[:]
                            return True
                        elif nei not in visited:
                            visited.add(nei)
                            path.append(nei)
                            parent[nei] = node
                            
                            if dfs_shuffled(nei):
                                return True
                            
                            path.pop()
                            visited.remove(nei)
                            del parent[nei]
                    return False

                dfs_shuffled(start)
                
                # Early exit if we found a Hamiltonian cycle
                if best_len == n:
                    break

            if best_cycle:
                pieces.append(best_cycle)
                # Remove edges of the cycle from remaining
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
            else:
                # No cycle found, break
                break

        # Remaining edges are singletons
        for u, v in sorted(remaining):
            pieces.append([u, v])
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
