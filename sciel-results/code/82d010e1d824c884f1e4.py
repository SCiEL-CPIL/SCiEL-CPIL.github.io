import random
import itertools

def construct():
    for n in [47, 46]:
        k = n - 39
        if k < 0:
            k = 0
            
        for attempt in range(100):
            p = (k + 2) / (n - 1)
            if p > 0.9: p = 0.9
            if p < 0.1: p = 0.1

            edges = set()
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        edges.add((u, v))
            
            # Maintain adjacency list
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # Pruning loop
            for _ in range(5000):
                # Find a C4
                c4_edges = None
                # Iterate over pairs
                for u in range(n):
                    for v in range(u + 1, n):
                        # Common neighbors
                        common = adj[u].intersection(adj[v])
                        if len(common) >= 2:
                            # Pick two common neighbors
                            w, x = list(common)[:2]
                            # The C4 is u-w-v-x-u
                            # Edges: (u,w), (w,v), (v,x), (x,u)
                            c4_edges = [(u,w), (w,v), (v,x), (x,u)]
                            break
                    if c4_edges:
                        break
                        
                if not c4_edges:
                    break
                    
                # Remove one edge
                edge_to_remove = random.choice(c4_edges)
                a, b = edge_to_remove
                edges.discard(edge_to_remove)
                adj[a].discard(b)
                adj[b].discard(a)
                
            # Check min degree
            degrees = [len(adj[i]) for i in range(n)]
            if min(degrees) >= k:
                return {'n': n, 'edges': [list(e) for e in edges]}
                
    return {'n': 39, 'edges': []}
