import random
import time

def construct():
    n = 16
    target_edges = 65
    deadline = time.time() + 100.0
    
    best_graph = None
    best_score = float('inf')
    
    # Helper to evaluate
    def evaluate(adj, deg):
        max_sum = 0
        for u in range(n):
            mask_u = adj[u]
            for v in range(u + 1, n):
                if (mask_u >> v) & 1:
                    common = mask_u & adj[v]
                    m = common
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        m ^= lsb
        return max_sum

    # Helper to get edge list from adj
    def get_edges(adj):
        edges = []
        for u in range(n):
            mask_u = adj[u]
            for v in range(u + 1, n):
                if (mask_u >> v) & 1:
                    edges.append((u, v))
        return edges

    # Local search
    def search(adj, deg, edges_set):
        current_score = evaluate(adj, deg)
        best_local_score = current_score
        best_local_adj = [a for a in adj]
        best_local_deg = [d for d in deg]
        
        while time.time() < deadline - 1.0:
            # Pick random edge to remove
            if not edges_set: break
            e_out = random.choice(list(edges_set))
            u, v = e_out
            
            # Pick random non-edge to add
            # Generate random pair
            x, y = random.sample(range(n), 2)
            if x > y: x, y = y, x
            if (x, y) in edges_set:
                continue
            if x == y:
                continue

            # Apply move
            # Remove (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            deg[u] -= 1
            deg[v] -= 1
            edges_set.remove((u, v))
            
            # Add (x,y)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            deg[x] += 1
            deg[y] += 1
            edges_set.add((x, y))
            
            new_score = evaluate(adj, deg)
            
            # Acceptance
            if new_score <= current_score:
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_adj = [a for a in adj]
                    best_local_deg = [d for d in deg]
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges_set.add((u, v))
                
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                deg[x] -= 1
                deg[y] -= 1
                edges_set.remove((x, y))
                
        return best_local_adj, best_local_deg, best_local_score

    # Initializations
    # 1. Seed-like
    adj = [0] * n
    deg = [0] * n
    edges_set = set()
    for u in range(8):
        for v in range(8, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            edges_set.add((u, v))
    # Add one edge
    u, v = 0, 1
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    deg[u] += 1
    deg[v] += 1
    edges_set.add((u, v))
    
    b_adj, b_deg, b_score = search(adj, deg, edges_set)
    if b_score < best_score:
        best_score = b_score
        best_graph = (b_adj, b_deg)
        
    # 2. Random
    for _ in range(20):
        if time.time() > deadline - 10.0: break
        adj = [0] * n
        deg = [0] * n
        edges_set = set()
        # Generate random graph with 65 edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:65]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            edges_set.add((u, v))
            
        b_adj, b_deg, b_score = search(adj, deg, edges_set)
        if b_score < best_score:
            best_score = b_score
            best_graph = (b_adj, b_deg)
            
    # Convert best graph to edge list
    edges = []
    for u in range(n):
        mask_u = best_graph[0][u]
        for v in range(u + 1, n):
            if (mask_u >> v) & 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
