import numpy as np
import time
import random

def construct():
    n = 16
    target_edges = 65
    
    # Precompute all triangles
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    
    tri_array = np.array(triangles, dtype=int)
    
    # Helper to compute score from adjacency matrix
    def compute_score(A):
        d = A.sum(axis=1)
        u, v, w = tri_array[:, 0], tri_array[:, 1], tri_array[:, 2]
        active = A[u, v] & A[v, w] & A[w, u]
        if not np.any(active):
            return 0
        sums = d[u] + d[v] + d[w]
        return np.max(sums[active])
    
    # Initial Construction: K_{8,8} + 1 edge
    A = np.zeros((n, n), dtype=bool)
    for u in range(8):
        for v in range(8, 16):
            A[u, v] = True
            A[v, u] = True
    A[0, 1] = True
    A[1, 0] = True
    
    current_score = compute_score(A)
    best_A = A.copy()
    best_score = current_score
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Simulated Annealing with targeted moves
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() - start_time < time_limit:
        # Pick a move: Swap an edge
        # 1. Pick an existing edge to remove
        # 2. Pick a non-existing edge to add
        
        # Get list of current edges
        current_edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    current_edges_list.append((u, v))
        
        if not current_edges_list:
            break
            
        # Targeted move: try to remove an edge from a high-degree-sum triangle
        # First, identify the worst triangles
        d = A.sum(axis=1)
        u, v, w = tri_array[:, 0], tri_array[:, 1], tri_array[:, 2]
        active = A[u, v] & A[v, w] & A[w, u]
        
        if np.any(active):
            sums = d[u] + d[v] + d[w]
            active_sums = sums[active]
            max_sum = np.max(active_sums)
            
            # Get triangles with sum close to max
            threshold = max_sum - 2
            high_tri_indices = np.where(active & (sums >= threshold))[0]
            
            # Pick a random high triangle and remove one of its edges
            if len(high_tri_indices) > 0:
                tri_idx = high_tri_indices[random.randint(0, len(high_tri_indices) - 1)]
                tri_u, tri_v, tri_w = tri_array[tri_idx]
                # Pick one of the three edges to remove
                edge_choice = random.randint(0, 2)
                if edge_choice == 0:
                    rem_u, rem_v = tri_u, tri_v
                elif edge_choice == 1:
                    rem_u, rem_v = tri_v, tri_w
                else:
                    rem_u, rem_v = tri_w, tri_u
            else:
                rem_u, rem_v = random.choice(current_edges_list)
        else:
            rem_u, rem_v = random.choice(current_edges_list)
        
        # Pick a non-edge
        added = False
        for _ in range(10):
            add_u = random.randint(0, n - 1)
            add_v = random.randint(0, n - 1)
            if add_u == add_v:
                continue
            if add_u > add_v:
                add_u, add_v = add_v, add_u
            if not A[add_u, add_v]:
                break
        else:
            continue
            
        # Perform the swap
        A_new = A.copy()
        A_new[rem_u, rem_v] = False
        A_new[rem_v, rem_u] = False
        A_new[add_u, add_v] = True
        A_new[add_v, add_u] = True
        
        new_score = compute_score(A_new)
        
        # Acceptance criteria
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            A = A_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = A.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            if random.random() < 0.01:
                T = 5.0

    # Final polish: Hill Climbing
    A_final = best_A.copy()
    score_final = best_score
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Get current edges for A_final
        current_edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if A_final[u, v]:
                    current_edges_list.append((u, v))
        
        # Targeted improvement: try to remove edges from worst triangles
        d = A_final.sum(axis=1)
        u, v, w = tri_array[:, 0], tri_array[:, 1], tri_array[:, 2]
        active = A_final[u, v] & A_final[v, w] & A_final[w, u]
        
        if np.any(active):
            sums = d[u] + d[v] + d[w]
            active_sums = sums[active]
            max_sum = np.max(active_sums)
            threshold = max_sum - 1
            high_tri_indices = np.where(active & (sums >= threshold))[0]
            
            for tri_idx in high_tri_indices:
                if time.time() - start_time > time_limit:
                    break
                tri_u, tri_v, tri_w = tri_array[tri_idx]
                # Try removing each edge of this triangle
                for rem_u, rem_v in [(tri_u, tri_v), (tri_v, tri_w), (tri_w, tri_u)]:
                    # Try adding a random non-edge
                    for _ in range(20):
                        add_u = random.randint(0, n - 1)
                        add_v = random.randint(0, n - 1)
                        if add_u == add_v:
                            continue
                        if add_u > add_v:
                            add_u, add_v = add_v, add_u
                        if A_final[add_u, add_v]:
                            continue
                        if (add_u, add_v) == (rem_u, rem_v):
                            continue
                        
                        A_new = A_final.copy()
                        A_new[rem_u, rem_v] = False
                        A_new[rem_v, rem_u] = False
                        A_new[add_u, add_v] = True
                        A_new[add_v, add_u] = True
                        
                        new_score = compute_score(A_new)
                        if new_score < score_final:
                            A_final = A_new
                            score_final = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # If no targeted improvement, try random swaps
        if not improved:
            for _ in range(50):
                rem_u, rem_v = random.choice(current_edges_list)
                for _ in range(10):
                    add_u = random.randint(0, n - 1)
                    add_v = random.randint(0, n - 1)
                    if add_u == add_v:
                        continue
                    if add_u > add_v:
                        add_u, add_v = add_v, add_u
                    if A_final[add_u, add_v]:
                        continue
                    
                    A_new = A_final.copy()
                    A_new[rem_u, rem_v] = False
                    A_new[rem_v, rem_u] = False
                    A_new[add_u, add_v] = True
                    A_new[add_v, add_u] = True
                    
                    new_score = compute_score(A_new)
                    if new_score < score_final:
                        A_final = A_new
                        score_final = new_score
                        improved = True
                        break
                if improved:
                    break

    # Pick the best
    score_best_A = compute_score(best_A)
    score_A_final = compute_score(A_final)
    
    if score_A_final < score_best_A:
        final_A = A_final
    else:
        final_A = best_A
        
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if final_A[u, v]:
                edges.append([u, v])
                
    # Verify edge count
    if len(edges) != 65:
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        
    return {'n': n, 'edges': edges}
