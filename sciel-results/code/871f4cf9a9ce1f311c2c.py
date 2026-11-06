import random
import time

def construct():
    n = 20
    # adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    # neighbors list for quick access
    neighbors = [set() for _ in range(n)]
    
    def is_valid_add(u, v):
        if adj[u][v]:
            return False
        # 1. Common neighbors of u and v
        common_uv = neighbors[u] & neighbors[v]
        if len(common_uv) > 2:
            return False
        
        # 2. For each w in N(v), v becomes a common neighbor of u and w.
        #    So we need current common neighbors of u and w to be <= 1.
        for w in neighbors[v]:
            if w == u: continue
            common_uw = neighbors[u] & neighbors[w]
            if len(common_uw) > 1:
                return False
                
        # 3. For each w in N(u), u becomes a common neighbor of v and w.
        for w in neighbors[u]:
            if w == v: continue
            common_vw = neighbors[v] & neighbors[w]
            if len(common_vw) > 1:
                return False
                
        return True

    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        neighbors[u].add(v)
        neighbors[v].add(u)

    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False
        neighbors[u].remove(v)
        neighbors[v].remove(u)

    def count_edges():
        return sum(len(neighbors[i]) for i in range(n)) // 2

    def get_edges():
        edges = []
        for u in range(n):
            for v in neighbors[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    best_edges = 0
    best_graph = None
    
    start_time = time.time()
    time_limit = 110.0 # Leave margin for safety
    
    # Heuristic initial graph: Start empty
    for i in range(n):
        neighbors[i] = set()
    for i in range(n):
        for j in range(n):
            adj[i][j] = False
            
    # Pre-compute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() - start_time < time_limit:
        # Random restart
        for i in range(n):
            neighbors[i] = set()
        for i in range(n):
            for j in range(n):
                adj[i][j] = False
        
        # Greedy addition phase
        # Shuffle edge order
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if is_valid_add(u, v):
                add_edge(u, v)
        
        current_edges = count_edges()
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = get_edges()
            
        # Local search phase
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to add any valid edge
            # Shuffle to avoid bias? Or just iterate.
            # Iterating is deterministic given the state.
            for u, v in all_pairs:
                if not adj[u][v] and is_valid_add(u, v):
                    add_edge(u, v)
                    improved = True
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_graph = get_edges()
                    break # Restart scanning? 
                    # Breaking restarts the scan from the beginning, which is good for finding the "next" best.
                    # However, it might be slow. 
                    # Since n=20, 190 checks is cheap.
            
            if not improved:
                # Try a kick: remove a random edge
                if current_edges > 0:
                    # Pick a random existing edge
                    existing_edges = [(u, v) for u, v in all_pairs if adj[u][v]]
                    if existing_edges:
                        u, v = random.choice(existing_edges)
                        remove_edge(u, v)
                        current_edges -= 1
                        improved = True # Loop again to try adding

    # Fallback to seed if search failed or returned nothing
    if best_graph is None:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
        
    return best_graph
