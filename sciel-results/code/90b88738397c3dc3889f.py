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
    # tri_array shape: (560, 3)
    
    # Helper to compute score from adjacency matrix
    def compute_score(A):
        # A is 16x16 boolean
        # Get degrees
        d = A.sum(axis=1)
        
        # Check which triangles are active
        # For each triangle (u, v, w), check A[u,v], A[v,w], A[w,u]
        # We can use advanced indexing
        u, v, w = tri_array[:, 0], tri_array[:, 1], tri_array[:, 2]
        
        # Boolean mask for active triangles
        active = A[u, v] & A[v, w] & A[w, u]
        
        if not np.any(active):
            return 0 # No triangles, score 0 (but we have 65 edges, so triangles exist)
            
        # Calculate sums for active triangles
        # d[u] + d[v] + d[w]
        sums = d[u] + d[v] + d[w]
        
        # Max sum among active
        return np.max(sums[active])

    # Initial Construction: K_{8,8} + 1 edge
    # Vertices 0-7 and 8-15
    A = np.zeros((n, n), dtype=bool)
    for u in range(8):
        for v in range(8, 16):
            A[u, v] = True
            A[v, u] = True
    # Add one edge inside part 0
    A[0, 1] = True
    A[1, 0] = True
    
    current_edges = 65
    current_score = compute_score(A)
    best_A = A.copy()
    best_score = current_score
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # To speed up, we can maintain list of current edges and non-edges
    # But for n=16, iterating over all 120 edges is cheap.
    
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
            
        rem_u, rem_v = random.choice(current_edges_list)
        
        # Pick a non-edge
        # We can pick randomly from all non-edges
        # To be efficient, we can just try random pairs until we find a non-edge
        # Or precompute non-edges
        # For simplicity, let's just pick random u, v and check
        # If we hit an existing edge, we reject the move (or pick another)
        # Better: maintain a set of non-edges? 
        # n=16 is small. Let's just pick random pair.
        
        added = False
        for _ in range(10): # Try up to 10 times to find a non-edge
            add_u = random.randint(0, n - 1)
            add_v = random.randint(0, n - 1)
            if add_u == add_v:
                continue
            if add_u > add_v:
                add_u, add_v = add_v, add_u
            if not A[add_u, add_v]:
                break
        else:
            # Failed to find non-edge (shouldn't happen with 65/120 edges)
            continue
            
        # Perform the swap on a copy? No, modify in place and revert if rejected.
        # Or compute new score on a copy. Copying 16x16 bool is cheap.
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
            # Restart with higher T? Or just keep going.
            # Let's add a small chance to reheat
            if random.random() < 0.01:
                T = 5.0

    # Final polish: Hill Climbing
    # Do a few passes of only accepting improvements
    # Since we already did SA, this might not help much, but it's cheap.
    # Let's do a simple local search on best_A
    A_final = best_A.copy()
    score_final = best_score
    
    # Get current edges for A_final
    current_edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if A_final[u, v]:
                current_edges_list.append((u, v))
                
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(len(current_edges_list)):
            rem_u, rem_v = current_edges_list[i]
            # Try to swap with a random non-edge
            for _ in range(20):
                add_u = random.randint(0, n - 1)
                add_v = random.randint(0, n - 1)
                if add_u == add_v or (add_u, add_v) == (rem_u, rem_v) or (add_u, add_v) == (rem_v, rem_u):
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
                    # Update current_edges_list? 
                    # It's getting complicated to maintain the list dynamically.
                    # Let's just break and restart the loop?
                    # Or just accept and continue. The list is stale but we are just checking swaps.
                    # Actually, if we change the graph, current_edges_list is invalid.
                    # Let's just do a few random restarts of the polish instead.
                    break
            if improved:
                break

    # Construct final edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v]: # Use best_A found during SA, or A_final?
                # Let's use the best found in SA, as polish might have degraded if not careful.
                # Actually, let's evaluate both.
                pass
                
    # Let's just use best_A from SA, it's robust.
    # Or use A_final if it improved.
    # Let's re-evaluate best_A and A_final and pick the best.
    
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
        # Fallback to K_{8,8}+1 if something went wrong
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        
    return {'n': n, 'edges': edges}
