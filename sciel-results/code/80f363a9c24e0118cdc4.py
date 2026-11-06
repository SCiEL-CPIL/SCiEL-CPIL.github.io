import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    
    n = 16
    target_edges = 65
    
    def build_adj_deg(edges):
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score(adj, deg):
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    continue
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    def compute_score_edges(edges):
        adj, deg = build_adj_deg(edges)
        return compute_score(adj, deg)
    
    def is_valid(edges):
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    # Baseline: K_{6,10} + 5 matching edges in B (vertices 6-15)
    # A = {0,1,2,3,4,5}, B = {6,7,8,9,10,11,12,13,14,15}
    base_edges = []
    for u in range(6):
        for v in range(6, 16):
            base_edges.append([u, v])
    # 5 matching edges in B
    matching = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in matching:
        base_edges.append([u, v])
    
    assert len(base_edges) == 65, f"Expected 65 edges, got {len(base_edges)}"
    base_score = compute_score_edges(base_edges)
    
    best_edges = base_edges[:]
    best_score = base_score
    
    # Local search with edge swaps
    def local_search(edges, max_iters=50000):
        current = edges[:]
        adj, deg = build_adj_deg(current)
        current_score = compute_score(adj, deg)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 3.0:
                break
            
            # Try edge swaps: remove one edge, add another
            for i in range(len(current)):
                if time.time() > deadline - 3.0:
                    break
                u1, v1 = current[i]
                # Try random non-edges to add
                for _ in range(3):
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    if adj[u2][v2]:
                        continue
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    
                    # Compute new score
                    new_adj = [[row[:] for row in adj]]
                    new_deg = deg[:]
                    new_adj[u1][v1] = False
                    new_adj[v1][u1] = False
                    new_deg[u1] -= 1
                    new_deg[v1] -= 1
                    new_adj[u2][v2] = True
                    new_adj[v2][u2] = True
                    new_deg[u2] += 1
                    new_deg[v2] += 1
                    
                    new_score = compute_score(new_adj, new_deg)
                    
                    if new_score < current_score:
                        current[i] = [u2, v2]
                        adj = new_adj
                        deg = new_deg
                        current_score = new_score
                        improved = True
                        break
        
        return current, current_score
    
    # Run local search from baseline
    ls_edges, ls_score = local_search(base_edges, max_iters=100000)
    if ls_score < best_score:
        best_score = ls_score
        best_edges = ls_edges[:]
    
    # Multiple restarts with perturbations
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Start from current best with perturbation
        perturbed = best_edges[:]
        num_swap = random.randint(1, 3)
        adj_check, _ = build_adj_deg(perturbed)
        for _ in range(num_swap):
            idx = random.randint(0, len(perturbed) - 1)
            u1, v1 = perturbed[idx]
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2 or adj_check[u2][v2]:
                    continue
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                break
            perturbed[idx] = [u2, v2]
            adj_check[u1][v1] = False
            adj_check[v1][u1] = False
            adj_check[u2][v2] = True
            adj_check[v2][u2] = True
        
        if not is_valid(perturbed):
            continue
        
        p_score = compute_score_edges(perturbed)
        if p_score < best_score:
            best_score = p_score
            best_edges = perturbed[:]
        
        # Local search from perturbation
        ls_edges2, ls_score2 = local_search(perturbed, max_iters=30000)
        if ls_score2 < best_score:
            best_score = ls_score2
            best_edges = ls_edges2[:]
    
    return {'n': n, 'edges': best_edges}
