import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            edges.add((u, v))
    
    def is_q3(vertices):
        if len(vertices) != 8:
            return False
        adj = {v: set() for v in vertices}
        for u in vertices:
            for v in vertices:
                if u != v and ((u, v) in edges or (v, u) in edges):
                    adj[u].add(v)
        
        for v in vertices:
            if len(adj[v]) != 3:
                return False
        
        color = {}
        queue = []
        for v in vertices:
            if v not in color:
                color[v] = 0
                queue.append(v)
                while queue:
                    u = queue.pop(0)
                    for w in adj[u]:
                        if w not in color:
                            color[w] = 1 - color[u]
                            queue.append(w)
                        elif color[w] == color[u]:
                            return False
        
        if len(color) != 8:
            return False
        
        for start in vertices:
            dist = {start: 0}
            queue = [start]
            while queue:
                u = queue.pop(0)
                for w in adj[u]:
                    if w not in dist:
                        dist[w] = dist[u] + 1
                        queue.append(w)
            if max(dist.values()) > 3:
                return False
            if len(dist) != 8:
                return False
        
        return True

    while time.time() < deadline - 10:
        q3s = []
        for verts in itertools.combinations(range(n), 8):
            if is_q3(verts):
                q3s.append(verts)
        
        if not q3s:
            break
        
        q3 = random.choice(q3s)
        q3_edge_list = [(u, v) for u in q3 for v in q3 if u < v and (u, v) in edges]
        if q3_edge_list:
            edges.remove(random.choice(q3_edge_list))
    
    return [[u, v] for u, v in edges]
