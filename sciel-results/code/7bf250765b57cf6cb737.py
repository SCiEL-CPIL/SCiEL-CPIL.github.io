import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    def find_simple_path(u, v, edges, adj):
        # BFS is faster for shortest path, but we just need ANY simple path.
        # DFS is fine.
        visited = {u}
        stack = [(u, [u])]
        while stack:
            node, path = stack.pop()
            if node == v:
                return path
            for nb in adj[node]:
                e = (min(node, nb), max(node, nb))
                if e in edges and nb not in visited:
                    stack.append((nb, path + [nb]))
        return None

    def find_longest_simple_path(start, edges, adj):
        beam = [([start], {start}, start)]
        best = [start]
        max_depth = 40
        beam_width = 15
        
        for _ in range(max_depth):
            new_beam = []
            for path, visited, node in beam:
                valid_neighbors = []
                for nb in adj[node]:
                    if nb in visited:
                        continue
                    e = (min(node, nb), max(node, nb))
                    if e in edges:
                        valid_neighbors.append(nb)
                
                if not valid_neighbors:
                    if len(path) > len(best):
                        best = path[:]
                    continue
                
                scored = []
                for nb in valid_neighbors:
                    count = 0
                    for m in adj[nb]:
                        if m in visited:
                            continue
                        e2 = (min(nb, m), max(nb, m))
                        if e2 in edges:
                            count += 1
                    scored.append((count, nb))
                scored.sort()
                
                for _, nb in scored[:beam_width]:
                    new_path = path + [nb]
                    new_visited = visited | {nb}
                    new_beam.append((new_path, new_visited, nb))
                    
            if not new_beam:
                break
            
            new_beam.sort(key=lambda x: -len(x[0]))
            beam = new_beam[:beam_width]
            
            if beam:
                longest = max(len(p) for p, v, n_ in beam)
                if longest > len(best):
                    best = beam[0][0][:]
                    
        return best

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        remaining_edges = set(edges)
        paths = []
        
        # 1. Pair odd vertices and remove paths
        odd_vertices = [v for v in range(n) if deg[v] % 2 != 0]
        random.shuffle(odd_vertices)
        
        for i in range(0, len(odd_vertices), 2):
            if i + 1 >= len(odd_vertices):
                break
            u = odd_vertices[i]
            v = odd_vertices[i+1]
            
            # Try to find a path. If not found (shouldn't happen), skip
            path = find_simple_path(u, v, remaining_edges, adj)
            
            if path:
                paths.append(path)
                for j in range(len(path) - 1):
                    a, b = path[j], path[j+1]
                    e = (min(a, b), max(a, b))
                    remaining_edges.discard(e)
                    deg[a] -= 1
                    deg[b] -= 1
        
        # 2. Cover remaining edges (Eulerian)
        curr_edges = set(remaining_edges)
        while curr_edges:
            if time.time() > deadline - 5.0:
                break
            
            active_nodes = [v for v in range(n) if any((min(v, nb), max(v, nb)) in curr_edges for nb in adj[v])]
            if not active_nodes:
                break
            random.shuffle(active_nodes)
            
            best_path = None
            for s in active_nodes[:3]:
                p = find_longest_simple_path(s, curr_edges, adj)
                if p and (best_path is None or len(p) > len(best_path)):
                    best_path = p
            
            if best_path:
                paths.append(best_path)
                for j in range(len(best_path) - 1):
                    a, b = best_path[j], best_path[j+1]
                    e = (min(a, b), max(a, b))
                    curr_edges.discard(e)
            else:
                e = next(iter(curr_edges))
                paths.append([e[0], e[1]])
                curr_edges.discard(e)

        all_results.append(paths)

    return all_results
