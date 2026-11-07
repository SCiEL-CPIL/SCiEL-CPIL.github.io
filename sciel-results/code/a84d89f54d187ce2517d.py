import time
import sys

def construct():
    start_time = time.time()
    deadline = 145.0

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        color = [-1] * n
        # saturation[v] is a set of colors used by colored neighbors
        saturation = [set() for _ in range(n)]
        
        sys.setrecursionlimit(10000)
        
        def backtrack():
            # Find uncolored vertex with highest saturation, then highest degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(n):
                if color[v] == -1:
                    sat = len(saturation[v])
                    deg = len(adj[v])
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_v = v
                        best_sat = sat
                        best_deg = deg
                        
            if best_v == -1:
                return True # All colored
            
            v = best_v
            for c in lists[v]:
                if c in saturation[v]:
                    continue
                
                # Assign
                color[v] = c
                # Update neighbors
                for u in adj[v]:
                    if color[u] == -1:
                        saturation[u].add(c)
                
                if backtrack():
                    return True
                
                # Undo
                color[v] = -1
                for u in adj[v]:
                    if color[u] == -1:
                        saturation[u].remove(c)
            
            return False
        
        if backtrack():
            return color
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            break
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        out.append([int(c) for c in coloring])
        
    while len(out) < 4:
        out.append([0])
        
    return out
