import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    n = 16
    m = 65
    
    # Helper to compute score
    def get_score(adj, deg):
        max_sum = 0
        # Iterate over all triangles
        # For speed, we can iterate over edges and common neighbors
        # But n=16 is small, so O(n^3) is fine (~4096 ops)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def random_graph():
        # Generate a random graph with exactly m edges
        # To be close to the optimum, we should start with a structure similar to Turan graph T_2(16)
        # T_2(16) is K_{8,8} with 64 edges. We need 1 more edge.
        # Let's start with K_{8,8} + random internal edge, then randomize if needed.
        adj = [[0]*n for _ in range(n)]
        # Create K_{8,8}
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = 1
                adj[j][i] = 1
        # Add 1 random internal edge
        while True:
            u = random.randint(0, 15)
            v = random.randint(0, 15)
            if u != v and not adj[u][v]:
                adj[u][v] = 1
                adj[v][u] = 1
                break
        return adj

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    def compute_degrees(adj):
        return [sum(row) for row in adj]

    best_score = float('inf')
    best_adj = None
    
    # Initial solution
    init_adj = random_graph()
    init_deg = compute_degrees(init_adj)
    init_score = get_score(init_adj, init_deg)
    best_score = init_score
    best_adj = [row[:] for row in init_adj]
    
    current_adj = [row[:] for row in init_adj]
    current_deg = compute_degrees(current_adj)
    current_score = init_score
    
    # Local Search with Simulated Annealing
    # Temperature schedule
    T = 5.0
    cooling = 0.999
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() - start_time < deadline - 5:
        # Generate a random move: remove edge e1, add edge e2
        # To maintain m=65, we must swap.
        
        # Pick a random existing edge to remove
        existing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if current_adj[u][v]]
        if not existing_edges:
            break
        e1 = random.choice(existing_edges)
        
        # Pick a random non-existing edge to add
        # Optimization: Pick random u, v. If not connected, it's a candidate.
        # But we need to ensure it's not e1.
        # Let's just pick from all_edges not in current_adj
        # For n=16, total edges 120. m=65. Non-edges ~ 55.
        
        candidates = []
        for e in all_edges:
            if not current_adj[e[0]][e[1]]:
                if e != e1: # e1 is existing, so this is always true unless we messed up logic
                    candidates.append(e)
        
        if not candidates:
            break
            
        e2 = random.choice(candidates)
        
        # Apply move
        u1, v1 = e1
        u2, v2 = e2
        
        current_adj[u1][v1] = 0
        current_adj[v1][u1] = 0
        current_deg[u1] -= 1
        current_deg[v1] -= 1
        
        current_adj[u2][v2] = 1
        current_adj[v2][u2] = 1
        current_deg[u2] += 1
        current_deg[v2] += 1
        
        new_score = get_score(current_adj, current_deg)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.71828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in current_adj]
        else:
            # Revert
            current_adj[u1][v1] = 1
            current_adj[v1][u1] = 1
            current_deg[u1] += 1
            current_deg[v1] += 1
            
            current_adj[u2][v2] = 0
            current_adj[v2][u2] = 0
            current_deg[u2] -= 1
            current_deg[v2] -= 1
            
        T *= cooling
        if T < 0.01:
            # Restart with new random graph if stuck too long or T too low
            # Actually, just reset T to allow exploration
            T = 1.0
            
    # Final check: ensure we have exactly 65 edges
    # The swap logic preserves edge count, so best_adj should have 65 edges.
    # But let's verify and fix if necessary (should not happen).
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
                
    if len(edges) != m:
        # Fallback to seed if something went wrong
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        
    return {'n': n, 'edges': edges}

