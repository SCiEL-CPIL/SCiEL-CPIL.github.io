import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Map edge index to (u, v) for fast lookup
    edge_to_uv = {}
    uv_to_edge = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_uv[i] = (u, v)
        uv_to_edge[(u, v)] = i
        uv_to_edge[(v, u)] = i

    # Helper to check validity of a graph represented by adjacency matrix
    # Returns True if no pair of vertices has > 2 common neighbours
    def is_valid(adj_mat):
        # C = adj @ adj.T gives counts of common neighbours for all pairs
        # Using integer arithmetic for speed if possible, but float is fine for n=20
        common_counts = adj_mat @ adj_mat.T
        # Check off-diagonal elements
        np.fill_diagonal(common_counts, 0)
        return np.max(common_counts) <= 2

    # Helper to get current edge count
    def get_edge_count(adj_mat):
        return int(np.sum(adj_mat)) // 2

    # Initialize best solution using a fast greedy method
    # This provides a strong starting point for SA
    def greedy_init():
        adj = np.zeros((n, n), dtype=np.uint8)
        # Shuffle edges for randomized greedy
        shuffled = all_edges.copy()
        random.shuffle(shuffled)
        
        # We need to track common neighbors incrementally or just recompute
        # For n=20, recomputing C = A @ A.T is ~8000 ops. 
        # 190 edges * 8000 ops = 1.5M ops. Very fast.
        
        # To be even faster, we can use the incremental check logic from the seed
        # but for simplicity in init, we just do a few greedy passes
        
        improved = True
        while improved:
            improved = False
            C = adj @ adj.T
            np.fill_diagonal(C, 0)
            
            for u, v in shuffled:
                if adj[u, v] == 1:
                    continue
                
                # Check if adding (u,v) is valid
                # C[i, v] increases by A[i, u]
                # C[u, j] increases by A[v, j]
                
                # If C[i, v] == 2 and A[i, u] == 1, invalid
                # If C[u, j] == 2 and A[v, j] == 1, invalid
                
                valid = True
                # Check column v
                idx_v = np.where(C[:, v] == 2)[0]
                for i in idx_v:
                    if adj[i, u] == 1:
                        valid = False
                        break
                
                if valid:
                    # Check row u
                    idx_u = np.where(C[u, :] == 2)[0]
                    for j in idx_u:
                        if adj[v, j] == 1:
                            valid = False
                            break
                
                if valid:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    improved = True
        
        return adj

    best_adj = greedy_init()
    best_score = get_edge_count(best_adj)
    
    # Simulated Annealing
    # State: current_adj
    curr_adj = best_adj.copy()
    curr_score = best_score
    
    # Temperature parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Kick mechanism
    no_improve_counter = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with higher temperature
            T = 5.0
            no_improve_counter = 0
            # Optional: Perturb state
            # curr_adj = best_adj.copy()
            # curr_score = best_score
            # But let's just cool down and restart T if stuck
        
        # Propose a move
        # 70% Add, 30% Remove
        if random.random() < 0.7:
            # Try to add a random edge
            if curr_score < num_edges:
                # Find missing edges
                # Instead of scanning all, pick random indices
                # We need to ensure we don't pick an existing edge
                # Simple approach: pick random edge, if exists, skip or retry
                # Better: maintain a list of missing edges? 
                # For n=20, scanning 190 edges is fast.
                
                # Pick a random edge
                idx = random.randint(0, num_edges - 1)
                u, v = edge_to_uv[idx]
                
                if curr_adj[u, v] == 0:
                    # Check validity
                    C = curr_adj @ curr_adj.T
                    np.fill_diagonal(C, 0)
                    
                    valid = True
                    idx_v = np.where(C[:, v] == 2)[0]
                    for i in idx_v:
                        if curr_adj[i, u] == 1:
                            valid = False
                            break
                    
                    if valid:
                        idx_u = np.where(C[u, :] == 2)[0]
                        for j in idx_u:
                            if curr_adj[v, j] == 1:
                                valid = False
                                break
                    
                    if valid:
                        # Accept
                        curr_adj[u, v] = 1
                        curr_adj[v, u] = 1
                        curr_score += 1
                        no_improve_counter = 0
                        
                        if curr_score > best_score:
                            best_score = curr_score
                            best_adj = curr_adj.copy()
                    else:
                        # Reject, count as no improvement? 
                        # Usually, a rejected move doesn't count as "no improvement" in the same way,
                        # but if we can't add anything, we are stuck.
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1
            else:
                no_improve_counter += 1
        else:
            # Try to remove a random edge
            if curr_score > 0:
                idx = random.randint(0, num_edges - 1)
                u, v = edge_to_uv[idx]
                
                if curr_adj[u, v] == 1:
                    # Removing an edge always results in a valid graph (fewer constraints)
                    # Acceptance: Delta E = -1
                    # Prob = exp(-1 / T)
                    prob = np.exp(-1.0 / T)
                    if random.random() < prob:
                        curr_adj[u, v] = 0
                        curr_adj[v, u] = 0
                        curr_score -= 1
                        # Do not update best here as score decreased
                        no_improve_counter += 1
                    else:
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1
            
        # Kick Mechanism
        if no_improve_counter > max_no_improve:
            # Perform a kick: remove some random edges to escape local optimum
            # Remove 5-10 random edges
            num_removals = random.randint(5, 10)
            for _ in range(num_removals):
                if curr_score == 0:
                    break
                idx = random.randint(0, num_edges - 1)
                u, v = edge_to_uv[idx]
                if curr_adj[u, v] == 1:
                    curr_adj[u, v] = 0
                    curr_adj[v, u] = 0
                    curr_score -= 1
            
            # Reset temperature
            T = max(T, 5.0)
            no_improve_counter = 0
            
        # Cool down
        T *= cooling_rate

    # Final Greedy Fill on the best solution found
    if time.time() < deadline:
        final_adj = best_adj.copy()
        final_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            C = final_adj @ final_adj.T
            np.fill_diagonal(C, 0)
            
            # Try to add any missing edge
            shuffled_final = all_edges.copy()
            random.shuffle(shuffled_final)
            
            for u, v in shuffled_final:
                if final_adj[u, v] == 1:
                    continue
                    
                valid = True
                idx_v = np.where(C[:, v] == 2)[0]
                for i in idx_v:
                    if final_adj[i, u] == 1:
                        valid = False
                        break
                
                if valid:
                    idx_u = np.where(C[u, :] == 2)[0]
                    for j in idx_u:
                        if final_adj[v, j] == 1:
                            valid = False
                            break
                
                if valid:
                    final_adj[u, v] = 1
                    final_adj[v, u] = 1
                    final_score += 1
                    improved = True
                    break # Restart loop with new C
        
        if final_score > best_score:
            best_score = final_score
            best_adj = final_adj

    # Convert back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return edges
