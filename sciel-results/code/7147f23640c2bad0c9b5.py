import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 512
    vertices = list(range(1, 512))
    weights = {a: 1.0 / (512.0 - a) for a in vertices}
    
    # Sort by weight descending
    sorted_vertices = sorted(vertices, key=lambda x: weights[x], reverse=True)
    
    # Build H (Compatibility Graph)
    # H[i][j] is True if gcd(sorted_vertices[i], sorted_vertices[j]) == 1
    H = [[False] * N for _ in range(N)]
    for i in range(N):
        vi = sorted_vertices[i]
        for j in range(i + 1, N):
            vj = sorted_vertices[j]
            if gcd(vi, vj) == 1:
                H[i][j] = True
                H[j][i] = True

    best_weight = 0.0
    best_clique = []
    
    def bound(candidates):
        if not candidates:
            return 0.0
            
        color_vertices = []
        bound_sum = 0.0
        
        for v_idx in candidates:
            placed = False
            w_v = weights[sorted_vertices[v_idx]]
            
            for k, c_verts in enumerate(color_vertices):
                can_join = True
                for u_idx in c_verts:
                    if H[v_idx][u_idx]:
                        can_join = False
                        break
                
                if can_join:
                    color_vertices[k].append(v_idx)
                    placed = True
                    break
            
            if not placed:
                color_vertices.append([v_idx])
                bound_sum += w_v
                
        return bound_sum

    def dfs(candidates, current_weight, current_clique):
        nonlocal best_weight, best_clique
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = list(current_clique)
            return
            
        b = bound(candidates)
        if current_weight + b <= best_weight:
            return
            
        # Iterate candidates
        for i in range(len(candidates)):
            if time.time() > deadline:
                return
                
            v_idx = candidates[i]
            
            new_candidates = []
            for j in range(i + 1, len(candidates)):
                u_idx = candidates[j]
                if H[v_idx][u_idx]:
                    new_candidates.append(u_idx)
            
            current_clique.append(v_idx)
            dfs(new_candidates, current_weight + weights[sorted_vertices[v_idx]], current_clique)
            current_clique.pop()
            
    initial_candidates = list(range(N))
    dfs(initial_candidates, 0.0, [])
    
    result = [sorted_vertices[idx] for idx in best_clique]
    
    if not result:
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
        
    return result
