import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = q
    best_edges = seed_edges
    
    # Initialize adjacency matrix
    adj = np.zeros((42, 42), dtype=bool) # Pre-allocate max size
    for i, j in seed_edges:
        adj[i, j] = True
        adj[j, i] = True
        
    current_n = q
    
    # Precompute K4s and Ind4s and their incidence lists
    def get_k4s_ind4s_and_incidence(adj, n):
        k4s = []
        ind4s = []
        for combo in combinations(range(n), 4):
            is_clique = True
            is_ind = True
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = combo[i], combo[j]
                    if adj[u, v]:
                        is_ind = False
                    else:
                        is_clique = False
                    if not is_clique and not is_ind:
                        break
                if not is_clique and not is_ind:
                    break
            if is_clique:
                k4s.append(combo)
            if is_ind:
                ind4s.append(combo)
        
        # Build incidence lists
        # k4_inc[i] = list of indices of k4s containing i
        k4_inc = [[] for _ in range(n)]
        for idx, k4 in enumerate(k4s):
            for v in k4:
                k4_inc[v].append(idx)
                
        ind4_inc = [[] for _ in range(n)]
        for idx, i4 in enumerate(ind4s):
            for v in i4:
                ind4_inc[v].append(idx)
                
        return k4s, ind4s, k4_inc, ind4_inc

    k4s, ind4s, k4_inc, ind4_inc = get_k4s_ind4s_and_incidence(adj, current_n)
    
    # Try to add vertices
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 20:
            break
            
        found_valid = False
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize x randomly
            x = np.random.choice([0, 1], size=current_n).astype(bool)
            
            # Initialize counts
            k4_counts = np.zeros(len(k4s), dtype=np.int8)
            ind4_counts = np.zeros(len(ind4s), dtype=np.int8)
            
            # Calculate initial counts
            # For each K4, count how many vertices are in x
            for idx, k4 in enumerate(k4s):
                k4_counts[idx] = np.sum(x[list(k4)])
            for idx, i4 in enumerate(ind4s):
                # Count non-neighbors: 1 - x[v]
                ind4_counts[idx] = np.sum(1 - x[list(i4)])
                
            # Calculate initial violations
            current_violations = np.sum(k4_counts == 4) + np.sum(ind4_counts == 4)
            
            if current_violations == 0:
                found_valid = True
                break
            
            # Simulated Annealing on x
            temperature = 1.0
            cooling_rate = 0.999
            steps = 10000 # Can be high now
            
            for step in range(steps):
                if time.time() > deadline - 5:
                    break
                if current_violations == 0:
                    found_valid = True
                    break
                    
                # Flip a random bit
                idx_v = np.random.randint(0, current_n)
                old_val = x[idx_v]
                new_val = not old_val
                
                # Update counts and violations
                delta_violations = 0
                
                # Update K4s containing idx_v
                for k4_idx in k4_inc[idx_v]:
                    old_count = k4_counts[k4_idx]
                    if old_val:
                        # 1 -> 0
                        new_count = old_count - 1
                        if old_count == 4:
                            delta_violations -= 1
                    else:
                        # 0 -> 1
                        new_count = old_count + 1
                        if new_count == 4:
                            delta_violations += 1
                    k4_counts[k4_idx] = new_count
                    
                # Update Ind4s containing idx_v
                for i4_idx in ind4_inc[idx_v]:
                    old_count = ind4_counts[i4_idx]
                    if old_val:
                        # 1 -> 0 (x becomes 0, so non-neighbor count increases)
                        new_count = old_count + 1
                        if new_count == 4:
                            delta_violations += 1
                    else:
                        # 0 -> 1 (x becomes 1, so non-neighbor count decreases)
                        new_count = old_count - 1
                        if old_count == 4:
                            delta_violations -= 1
                    ind4_counts[i4_idx] = new_count
                
                new_violations = current_violations + delta_violations
                
                # Accept?
                if delta_violations < 0 or (temperature > 0.01 and random.random() < np.exp(-delta_violations / temperature)):
                    x[idx_v] = new_val
                    current_violations = new_violations
                else:
                    # Revert
                    # We already updated counts. Need to revert counts?
                    # Or just revert x and re-calculate?
                    # Reverting counts is symmetric.
                    for k4_idx in k4_inc[idx_v]:
                        if new_val:
                            # We set to 1, revert to 0
                            k4_counts[k4_idx] -= 1
                        else:
                            # We set to 0, revert to 1
                            k4_counts[k4_idx] += 1
                            
                    for i4_idx in ind4_inc[idx_v]:
                        if new_val:
                            # We set to 1 (non-neighbor count -1), revert
                            ind4_counts[i4_idx] += 1
                        else:
                            # We set to 0 (non-neighbor count +1), revert
                            ind4_counts[i4_idx] -= 1
                    
                    # x remains old_val
                    # current_violations remains old
                
                temperature *= cooling_rate
            
            if current_violations == 0:
                found_valid = True
                break
        
        if found_valid:
            # Add vertex target_n - 1
            new_idx = target_n - 1
            for i in range(current_n):
                if x[i]:
                    adj[i, new_idx] = True
                    adj[new_idx, i] = True
            
            current_n = target_n
            best_n = current_n
            # Update best_edges
            best_edges = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if adj[i, j]:
                        best_edges.append((i, j))
            
            # Update k4s, ind4s, and incidence
            k4s, ind4s, k4_inc, ind4_inc = get_k4s_ind4s_and_incidence(adj, current_n)
        else:
            # If we can't add this vertex, stop
            break
            
    return best_n, best_edges
