import random
import time

def construct():
    n = 16
    target_edges = 65
    
    # Initialize with K_8,8 + 1 edge (Score 26)
    # Vertices 0-7 in A, 8-15 in B
    adj = [[False]*n for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        if u == v: return
        if (u,v) in edges or (v,u) in edges: return
        adj[u][v] = True
        adj[v][u] = True
        edges.add((min(u,v), max(u,v)))
        
    def remove_edge(u, v):
        if (u,v) not in edges and (v,u) not in edges: return
        adj[u][v] = False
        adj[v][u] = False
        edges.discard((min(u,v), max(u,v)))
        edges.discard((max(u,v), min(u,v)))

    # Build K_8,8
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
            
    # Add one edge in A
    add_edge(0, 1)
    
    # Calculate degrees
    def get_degrees():
        degs = [0]*n
        for u in range(n):
            for v in range(n):
                if adj[u][v]:
                    degs[u] += 1
        return degs

    def calc_score():
        degs = get_degrees()
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: continue
                for w in range(v+1, n):
                    if adj[u][w] and adj[v][w]:
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    current_score = calc_score()
    best_graph = [row[:] for row in adj]
    best_score = current_score
    
    start_time = time.time()
    timeout = 10 # seconds
    
    while time.time() - start_time < timeout:
        # Random swap
        # Remove a random edge
        e_list = list(edges)
        if not e_list: break
        u, v = random.choice(e_list)
        remove_edge(u, v)
        
        # Add a random non-edge
        found = False
        attempts = 0
        while not found and attempts < 100:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x != y and not adj[x][y]:
                add_edge(x, y)
                found = True
            attempts += 1
        if not found:
            add_edge(u, v)
            continue
            
        new_score = calc_score()
        
        if new_score < best_score:
            best_score = new_score
            best_graph = [row[:] for row in adj]
            if best_score < 24: # Early exit if we beat the target
                break
        else:
            # Revert
            remove_edge(x, y)
            add_edge(u, v)
            
    # Reconstruct edges from best_graph
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_graph[u][v]:
                final_edges.append([u, v])
                
    return {'n': n, 'edges': final_edges}
