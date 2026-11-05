import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    adj = [[False] * n for _ in range(n)]
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = True
            adj[b][a] = True
    
    # Precompute edge sets for fast lookup
    edge_set = set()
    for a in range(3):
        for b in range(3, 16):
            edge_set.add((a, b))
            edge_set.add((b, a))
    
    def edge_exists(u, v):
        return (u, v) in edge_set
    
    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False
        edge_set.discard((u, v))
        edge_set.discard((v, u))
    
    def creates_q3(u, v):
        """Check if adding edge (u,v) creates a Q3 containing this edge."""
        others = [x for x in range(n) if x != u and x != v]
        for six in itertools.combinations(others, 6):
            verts = [u, v] + list(six)
            # Quick edge count
            ec = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if edge_exists(verts[i], verts[j]):
                        ec += 1
            if ec != 12:
                continue
            # Degree check
            degs = [0]*8
            for i in range(8):
                for j in range(8):
                    if i != j and edge_exists(verts[i], verts[j]):
                        degs[i] += 1
            if any(d != 3 for d in degs):
                continue
            # Bipartite check
            color = [-1]*8
            ok = True
            for start in range(8):
                if color[start] == -1:
                    color[start] = 0
                    stack = [start]
                    while stack and ok:
                        x = stack.pop()
                        for y in range(8):
                            if x != y and edge_exists(verts[x], verts[y]):
                                if color[y] == -1:
                                    color[y] = 1 - color[x]
                                    stack.append(y)
                                elif color[y] == color[x]:
                                    ok = False
                                    break
                if not ok:
                    break
            if ok:
                return True
        return False
    
    missing = [(i,j) for i in range(n) for j in range(i+1,n) if not edge_exists(i,j)]
    random.seed(42)
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline - 5:
            break
        if creates_q3(u, v):
            continue
        add_edge(u, v)
    
    edges = [[i,j] for i in range(n) for j in range(i+1,n) if edge_exists(i,j)]
    return edges
