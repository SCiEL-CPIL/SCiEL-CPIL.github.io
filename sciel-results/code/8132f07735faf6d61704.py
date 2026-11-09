import time
import random
import copy

def construct():
    # Baseline: K_8,8 plus one edge (0,1)
    n = 16
    # Initialize adjacency matrix
    # 1 if edge exists
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    # Add K_8,8 edges
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
            edges.append((u, v))
            
    # Add one internal edge
    edges.append((0, 1))
    adj[0][1] = 1
    adj[1][0] = 1
    
    def get_score(adj):
        # Calculate max degree sum over triangles
        deg = [0]*n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    deg[i] += 1
        
        max_sum = 0
        # Iterate over all triples
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    current_score = get_score(adj)
    best_score = current_score
    best_adj = copy.deepcopy(adj)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Simulated Annealing / Local Search
    T = 2.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Generate a random swap
        # Pick a random edge to remove
        u, v = edges[random.randint(0, len(edges)-1)]
        
        # Pick a random non-edge to add
        # To avoid bias, pick random pair x,y
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if x == y:
            continue
        if adj[x][y] == 1:
            continue
            
        # Check if swap is valid: removing (u,v) and adding (x,y)
        # We need to ensure (x,y) is not (u,v) or (v,u)
        if (x == u and y == v) or (x == v and y == u):
            continue
            
        # Perform swap on a copy or in-place with rollback
        # In-place is faster
        adj[u][v] = 0
        adj[v][u] = 0
        adj[x][y] = 1
        adj[y][x] = 1
        
        # Update edge list is not strictly necessary for score calc, 
        # but we used it for picking removal edge. 
        # For simplicity, we can just pick removal edge from adj matrix directly in next iterations?
        # No, `edges` list becomes stale. 
        # Better: pick u,v from adj matrix.
        
        new_score = get_score(adj)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = copy.deepcopy(adj)
        elif new_score == current_score:
            if random.random() < 0.5:
                current_score = new_score
            else:
                # Rollback
                adj[u][v] = 1
                adj[v][u] = 1
                adj[x][y] = 0
                adj[y][x] = 0
        else:
            # Delta = new_score - current_score
            if random.random() < pow(2.71828, -(new_score - current_score) / T):
                current_score = new_score
            else:
                # Rollback
                adj[u][v] = 1
                adj[v][u] = 1
                adj[x][y] = 0
                adj[y][x] = 0
                
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
