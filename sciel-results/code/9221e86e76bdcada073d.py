import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    def decompose_graph(n, edges, time_limit):
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        edge_set = set(edges)
        remaining_edges = set(edges)
        paths = []

        def extract_greedy_path(start, remaining, adj):
            path = [start]
            visited = {start}
            cur = start
            while True:
                candidates = []
                for nb in adj[cur]:
                    e = (min(cur, nb), max(cur, nb))
                    if e in remaining and nb not in visited:
                        candidates.append(nb)
                if not candidates:
                    break
                # Pick neighbor with highest degree in remaining graph
                # Approximate degree by len(adj[nb]) intersect remaining?
                # Just use original degree as heuristic
                nb = max(candidates, key=lambda x: len(adj[x]))
                e = (min(cur, nb), max(cur, nb))
                path.append(nb)
                visited.add(nb)
                cur = nb
            return path

        def find_best_path(remaining, adj, n):
            best_path = []
            # Try multiple random starts
            # If remaining is small, try all nodes
            nodes = list(remaining.nodes() if hasattr(remaining, 'nodes') else set())
            # remaining is a set of edges. Get nodes from edges.
            nodes = set()
            for u, v in remaining:
                nodes.add(u)
                nodes.add(v)
            
            if not nodes:
                return None
            
            # Sample starts
            if len(nodes) <= 10:
                starts = list(nodes)
            else:
                starts = random.sample(list(nodes), 10)
            
            for s in starts:
                p = extract_greedy_path(s, remaining, adj)
                if len(p) > len(best_path):
                    best_path = p
            return best_path

        while remaining_edges:
            if time.time() > DEADLINE - 5:
                # Fallback: add remaining edges as single-edge paths
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            path = find_best_path(remaining_edges, adj, n)
            if not path or len(path) < 2:
                # Fallback: pick any edge
                e = next(iter(remaining_edges))
                path = list(e)
            
            # Remove edges of path
            for i in range(len(path) - 1):
                e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                remaining_edges.discard(e)
            
            paths.append(path)

        return paths

    results = []
    for idx, spec in enumerate(specs):
        if time.time() > DEADLINE - 10:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        n, p, seed = spec
        edges = instance(n, p, seed)
        
        remaining_time = DEADLINE - time.time() - 5
        time_for_this = remaining_time / (len(specs) - idx)
        
        paths = decompose_graph(n, edges, time_limit=time_for_this)
        results.append(paths)

    return results
