import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0
    n = 40
    
    # State
    adj = [[False] * n for _ in range(n)]
    edges = set()
    neighbors = [set() for _ in range(n)]
    
    def add_edge(u, v):
        if u == v or adj[u][v]:
            return False
        adj[u][v] = True
        adj[v][u] = True
        neighbors[u].add(v)
        neighbors[v].add(u)
        edges.add((u, v) if u < v else (v, u))
        return True

    def remove_edge(u, v):
        if not adj[u][v]:
            return False
        adj[u][v] = False
        adj[v][u] = False
        neighbors[u].remove(v)
        neighbors[v].remove(u)
        edges.remove((u, v) if u < v else (v, u))
        return True

    def creates_c6(u, v):
        # Check if adding (u,v) creates a C6
        # Path u-a-b-d-v of length 4
        # a in N(u), d in N(v)
        # b in N(a) and b in N(d)
        # b != u, v, a, d
        # a != d
        
        nu = neighbors[u]
        nv = neighbors[v]
        
        for a in nu:
            for d in nv:
                if a == d:
                    continue
                # Check common neighbors of a and d
                common = neighbors[a].intersection(neighbors[d])
                for b in common:
                    if b != u and b != v and b != a and b != d:
                        return True
        return False

    # Initialize with 8 K5s
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u = 5*k + a
                v = 5*k + b
                add_edge(u, v)
                
    best_edges = len(edges)
    best_edge_set = set(edges)
    
    current_edges = len(edges)
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        # Try to add an edge
        added = False
        # Try 50 random additions
        for _ in range(50):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or adj[u][v]:
                continue
            if not creates_c6(u, v):
                add_edge(u, v)
                current_edges += 1
                added = True
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_edge_set = set(edges)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                break
        
        if not added:
            # Stuck, remove a random edge to unstick
            if edges:
                e = random.choice(list(edges))
                remove_edge(e[0], e[1])
                current_edges -= 1
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                # Restart from best state
                for e in list(edges):
                    remove_edge(e[0], e[1])
                for e in best_edge_set:
                    add_edge(e[0], e[1])
                current_edges = best_edges
                no_improve_count = 0
                max_no_improve += 50
                
    return [list(e) for e in edges]
