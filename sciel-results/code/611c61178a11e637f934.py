import random
import time

def construct():
    n = 40
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin for 150s limit
    
    def edges_to_adj(edges):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def is_in_c6(u, v, adj):
        # Check if edge (u,v) is part of a C6
        # Equivalent to: dist(u,v) in G\{(u,v)} is exactly 5
        adj[u][v] = False
        adj[v][u] = False
        
        dist = [-1] * n
        dist[u] = 0
        queue = [u]
        found_5 = False
        found_less = False
        
        for d in range(5):
            next_queue = []
            for node in queue:
                for nei in range(n):
                    if adj[node][nei] and dist[nei] == -1:
                        dist[nei] = d + 1
                        next_queue.append(nei)
                        if nei == v:
                            if d + 1 < 5:
                                found_less = True
                            else:
                                found_5 = True
            queue = next_queue
            if found_less or found_5:
                break
            if not queue:
                break
        
        adj[u][v] = True
        adj[v][u] = True
        
        return found_5

    def creates_c6(u, v, adj):
        # Check if adding (u,v) creates a C6
        # Equivalent to: dist(u,v) in G is exactly 5
        dist = [-1] * n
        dist[u] = 0
        queue = [u]
        found_less = False
        
        for d in range(4):
            next_queue = []
            for node in queue:
                for nei in range(n):
                    if adj[node][nei] and dist[nei] == -1:
                        dist[nei] = d + 1
                        next_queue.append(nei)
                        if nei == v:
                            found_less = True
            queue = next_queue
            if found_less:
                break
            if not queue:
                break
        
        if found_less:
            return False
            
        # Check if v is at dist 5
        for nei in range(n):
            if adj[v][nei] and dist[nei] == 4:
                return True
                
        return False

    best_edges = []
    best_count = 0
    
    # Seed: 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                best_edges.append([5 * k + a, 5 * k + b])
    best_count = len(best_edges)
    
    all_possible = [[u, v] for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        # Random initialization
        m_init = random.randint(80, 110)
        random.shuffle(all_possible)
        current_edges = [e[:] for e in all_possible[:m_init]]
        adj = edges_to_adj(current_edges)
        
        # Cleaning
        changed = True
        while changed and time.time() < deadline:
            changed = False
            random.shuffle(current_edges)
            for e in current_edges:
                u, v = e
                if is_in_c6(u, v, adj):
                    adj[u][v] = False
                    adj[v][u] = False
                    current_edges.remove(e)
                    changed = True
                    break
            
        # Addition
        non_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    non_edges.add((u, v))
        
        non_edge_list = list(non_edges)
        fails = 0
        while non_edge_list and time.time() < deadline and fails < 50:
            idx = random.randint(0, len(non_edge_list) - 1)
            u, v = non_edge_list[idx]
            
            if not creates_c6(u, v, adj):
                adj[u][v] = True
                adj[v][u] = True
                current_edges.append([u, v])
                non_edge_list.pop(idx)
                fails = 0
            else:
                fails += 1
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [e[:] for e in current_edges]
            
    return best_edges
