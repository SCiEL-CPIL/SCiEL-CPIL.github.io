import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        if not edges:
            return []

        # Build adjacency and degree structures
        adj = defaultdict(set)
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        pieces = []
        rng = random.Random(12345)

        # Helper to remove an edge
        def remove_edge(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1

        # Helper to remove a cycle
        def remove_cycle(cycle):
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                remove_edge(u, v)

        # Find a long cycle using randomized DFS with backtracking
        # We limit the search to avoid exponential blowup
        def find_long_cycle():
            best_cycle = None
            best_len = 0
            
            # Candidates: vertices with degree >= 2
            candidates = [v for v in range(n) if deg[v] >= 2]
            if not candidates:
                return None
            
            # Sort by degree descending to find longer cycles
            candidates.sort(key=lambda x: -deg[x])
            
            # Try a few starts
            max_starts = min(10, len(candidates))
            step_start = time.time()
            max_search_time = 0.5  # 500ms per cycle search step
            
            for start in candidates[:max_starts]:
                if time.time() - step_start > max_search_time:
                    break
                
                # DFS with backtracking
                # State: (current, prev, path, visited)
                # We use a stack for iterative DFS to avoid recursion limits
                # But for backtracking, recursion is easier. Let's use recursion with a limit.
                
                def dfs(curr, prev, path, visited):
                    nonlocal best_cycle, best_len
                    if time.time() - step_start > max_search_time:
                        return None
                    if curr == start and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        return path
                    for nei in adj[curr]:
                        if nei == prev:
                            continue
                        if nei == start:
                            if len(path) >= 3:
                                if len(path) > best_len:
                                    best_len = len(path)
                                    best_cycle = path[:]
                                return path
                        elif nei not in visited:
                            visited.add(nei)
                            path.append(nei)
                            result = dfs(nei, curr, path, visited)
                            if result:
                                return result
                            path.pop()
                            visited.remove(nei)
                    return None
                
                visited = {start}
                path = [start]
                dfs(start, -1, path, visited)
                
                if best_len > n * 0.8:
                    break

            return best_cycle

        # Main loop
        while True:
            if time.time() > deadline - 2:
                break
            
            # Check if any edges remain
            remaining_edges = sum(deg) // 2
            if remaining_edges == 0:
                break
            
            # Try to find a cycle
            cycle = find_long_cycle()
            
            if cycle:
                pieces.append(cycle)
                remove_cycle(cycle)
            else:
                # No cycle found, take a single edge
                # Find any edge
                edge_found = None
                for u in range(n):
                    if adj[u]:
                        v = next(iter(adj[u]))
                        edge_found = (u, v)
                        break
                
                if edge_found is None:
                    break
                
                u, v = edge_found
                pieces.append([u, v])
                remove_edge(u, v)
                
                # If we are stuck with many single edges, it might be faster to just
                # dump the rest as single edges if no cycles can be found quickly
                # But let's continue the loop to check for cycles again after removals
                # unless time is tight

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time for the remaining graphs, return empty or best effort
            # But we must return 3 lists. If we run out of time, we return empty lists
            # for the rest? No, the contract says "one per fixed instance".
            # If we can't compute, we should probably return the trivial decomposition
            # or empty if we can't even build the graph. But building is fast.
            # Let's just break and append empty lists? No, that's invalid (doesn't cover edges).
            # We must ensure we return valid decompositions.
            # If time is tight, we fall back to trivial decomposition for remaining graphs.
            n, p, seed = spec
            edges = instance(n, p, seed)
            pieces = [[u, v] for u, v in edges]
            results.append(pieces)
            continue

        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
