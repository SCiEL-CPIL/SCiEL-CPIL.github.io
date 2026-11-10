# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n, time_budget):
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        best_pieces = None
        best_count = len(edge_set) + 1

        def build_adj(rem):
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def find_longest_cycle_heuristic(adj, rem):
            """
            Heuristic to find a long simple cycle.
            Uses DFS with backtracking, prioritizing paths that are long.
            """
            if not rem:
                return None
            
            # Get vertices with degree >= 2
            candidates = [u for u in adj if len(adj[u]) >= 2]
            if not candidates:
                return None
            
            best_cycle = None
            best_len = 0
            
            # Try starting from a few random candidates
            random.shuffle(candidates)
            for start in candidates[:5]:
                visited = {start}
                path = [start]
                
                def dfs(u):
                    nonlocal best_cycle, best_len
                    if len(path) > best_len:
                        # Check if we can close the cycle
                        if start in adj[u] and len(path) >= 3:
                            cycle = path[:]
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle[:]
                    
                    # Explore neighbors
                    nbrs = list(adj[u])
                    # Prioritize neighbors not visited, and maybe those with higher degree
                    random.shuffle(nbrs)
                    for nb in nbrs:
                        if nb == start:
                            if len(path) >= 3:
                                cycle = path[:]
                                if len(cycle) > best_len:
                                    best_len = len(cycle)
                                    best_cycle = cycle[:]
                            continue
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            dfs(nb)
                            path.pop()
                            visited.remove(nb)
                
                dfs(start)
            
            return best_cycle

        def greedy_decompose_longest(rem_set):
            rem = set(rem_set)
            pieces = []
            while rem:
                adj = build_adj(rem)
                cycle = find_longest_cycle_heuristic(adj, rem)
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        rem.discard(e)
                    pieces.append(cycle[:])
                else:
                    # No cycle found; take one edge
                    e = tuple(next(iter(rem)))
                    rem.discard(e)
                    pieces.append(list(e))
            return pieces

        # Run multiple restarts with different random seeds for shuffling
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            # Re-seed random for variety in tie-breaking/shuffling
            random.seed(r * 1000 + 42)
            pieces = greedy_decompose_longest(edge_set)
            cnt = len(pieces)
            if cnt < best_count:
                best_count = cnt
                best_pieces = [p[:] for p in pieces]

        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 3:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
