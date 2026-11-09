import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140
    
    n = 16
    target_edges = 65
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    best_score = float('inf')
    best_edge_set = None
    
    def compute_score(edge_set):
        """Compute the max triangle degree sum for a given edge set."""
        deg = [0]*n
        for (u,v) in edge_set:
            deg[u] += 1
            deg[v] += 1
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) in edge_set or (j,i) in edge_set:
                    for k in range(j+1, n):
                        if ((i,k) in edge_set or (k,i) in edge_set) and ((j,k) in edge_set or (k,j) in edge_set):
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def compute_penalty(edge_set, penalty_weight=10.0):
        """
        Penalty-based objective: sum of squares of triangle degree sums exceeding a threshold.
        This aggressively penalizes high-degree triangles to flatten the distribution.
        """
        deg = [0]*n
        for (u,v) in edge_set:
            deg[u] += 1
            deg[v] += 1
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) in edge_set or (j,i) in edge_set:
                    for k in range(j+1, n):
                        if ((i,k) in edge_set or (k,i) in edge_set) and ((j,k) in edge_set or (k,j) in edge_set):
                            s = deg[i] + deg[j] + deg[k]
                            if s > 20:  # Threshold: penalize triangles with sum > 20
                                penalty += (s - 20) ** 2
        return penalty_weight * penalty
    
    def spectral_init():
        """
        Use spectral heuristic: construct graph by taking top 65 edges
        from a Laplacian-based eigenvector of a random perturbation.
        This naturally balances degrees and triangle formation.
        """
        # Create a random perturbation matrix
        np.random.seed(random.randint(0, 2**31))
        W = np.random.randn(n, n)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
        # Compute Laplacian
        deg_w = W.sum(axis=1)
        L = np.diag(deg_w) - W
        
        # Get eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Use the second eigenvector (Fiedler vector) for balanced partitioning
            # But we want to use it to score edges
            # Alternative: use the top eigenvector of the adjacency-like matrix
            # Let's use a different approach: score edges by product of eigenvector values
            
            # Get the first non-trivial eigenvector
            fiedler = eigenvectors[:, 1]
            
            # Score each edge by |fiedler[u] * fiedler[v]|
            # This tends to create balanced structures
            edge_scores = {}
            for (u,v) in all_edges:
                score = abs(fiedler[u] * fiedler[v])
                edge_scores[(u,v)] = score
            
            # Sort edges by score descending and take top 65
            sorted_edges = sorted(edge_scores.keys(), key=lambda e: edge_scores[e], reverse=True)
            edge_set = set(sorted_edges[:target_edges])
            
        except:
            # Fallback to K_{8,8} + 1
            edge_set = set()
            for u in range(8):
                for v in range(8, 16):
                    edge_set.add((u, v))
            edge_set.add((0, 1))
        
        return edge_set
    
    def random_init():
        """Random initialization with K_{8,8} base plus perturbations."""
        edge_set = set()
        for u in range(8):
            for v in range(8, 16):
                edge_set.add((u, v))
        
        # Add one random edge within a part
        if random.random() < 0.5:
            u = random.randint(0, 7)
            v = random.randint(0, 7)
            if u != v:
                edge_set.add((u, v))
        else:
            u = random.randint(8, 15)
            v = random.randint(8, 15)
            if u != v:
                edge_set.add((u, v))
        
        # Perturb: do 3 random edge swaps
        for _ in range(3):
            e_remove = random.choice(list(edge_set))
            e_add = random.choice(all_edges)
            if e_add not in edge_set and e_add != e_remove:
                edge_set.remove(e_remove)
                edge_set.add(e_add)
        
        return edge_set
    
    def local_search_penalty(edge_set_copy, time_limit):
        """
        Local search using penalty-based objective to flatten triangle degree sums.
        Returns (best_score, best_edge_set, final_score).
        """
        nonlocal best_score, best_edge_set
        
        es_local = set(edge_set_copy)
        current_score = compute_score(es_local)
        current_penalty = compute_penalty(es_local)
        
        # Track best by raw score
        local_best_score = current_score
        local_best_set = set(es_local)
        
        # Simulated annealing with penalty
        temp = 5.0
        cooling = 0.9995
        min_temp = 0.05
        
        while time.time() < time_limit:
            # Try a random edge swap
            e_remove = random.choice(list(es_local))
            e_add = random.choice(all_edges)
            if e_add in es_local or e_add == e_remove:
                continue
            
            # Apply swap
            es_local.remove(e_remove)
            es_local.add(e_add)
            
            new_score = compute_score(es_local)
            new_penalty = compute_penalty(es_local)
            
            # Combined objective: primary is score, secondary is penalty
            # We accept if score improves, or if score is same but penalty improves
            # Otherwise use simulated annealing on the penalty
            if new_score < current_score:
                current_score = new_score
                current_penalty = new_penalty
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_set = set(es_local)
                    if current_score < best_score:
                        best_score = current_score
                        best_edge_set = set(es_local)
            elif new_score == current_score and new_penalty < current_penalty:
                current_penalty = new_penalty
            else:
                # Use annealing on penalty difference
                delta_penalty = new_penalty - current_penalty
                if random.random() < math.exp(-delta_penalty / max(temp, min_temp)):
                    current_score = new_score
                    current_penalty = new_penalty
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_set = set(es_local)
                        if current_score < best_score:
                            best_score = current_score
                            best_edge_set = set(es_local)
                else:
                    # Revert
                    es_local.remove(e_add)
                    es_local.add(e_remove)
            
            # Cool down
            temp = max(temp * cooling, min_temp)
            
            # Occasionally try a 2-swap move
            if random.random() < 0.15:
                e_rem1 = random.choice(list(es_local))
                e_rem2 = random.choice(list(es_local))
                if e_rem1 == e_rem2:
                    continue
                e_add1 = random.choice(all_edges)
                e_add2 = random.choice(all_edges)
                if (e_add1 in es_local or e_add2 in es_local or 
                    e_add1 == e_add2 or 
                    e_add1 == e_rem1 or e_add1 == e_rem2 or
                    e_add2 == e_rem1 or e_add2 == e_rem2):
                    continue
                
                # Apply 2-swap
                old_es = set(es_local)
                es_local.remove(e_rem1)
                es_local.remove(e_rem2)
                es_local.add(e_add1)
                es_local.add(e_add2)
                
                new_score = compute_score(es_local)
                new_penalty = compute_penalty(es_local)
                
                if new_score < current_score:
                    current_score = new_score
                    current_penalty = new_penalty
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_set = set(es_local)
                        if current_score < best_score:
                            best_score = current_score
                            best_edge_set = set(es_local)
                elif new_score == current_score and new_penalty < current_penalty:
                    current_penalty = new_penalty
                else:
                    delta_penalty = new_penalty - current_penalty
                    if random.random() < math.exp(-delta_penalty / max(temp, min_temp)):
                        current_score = new_score
                        current_penalty = new_penalty
                        if current_score < local_best_score:
                            local_best_score = current_score
                            local_best_set = set(es_local)
                            if current_score < best_score:
                                best_score = current_score
                                best_edge_set = set(es_local)
                    else:
                        # Revert
                        es_local.clear()
                        es_local.update(old_es)
        
        return local_best_score, local_best_set
    
    # Main search loop with multiple initialization strategies
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        
        # Alternate between spectral and random initialization
        if restart % 3 == 0:
            es_init = spectral_init()
        else:
            es_init = random_init()
        
        # Ensure we have exactly 65 edges
        while len(es_init) < target_edges:
            e = random.choice(all_edges)
            es_init.add(e)
        while len(es_init) > target_edges:
            es_init.remove(random.choice(list(es_init)))
        
        # Compute initial score
        init_score = compute_score(es_init)
        
        if init_score < best_score:
            best_score = init_score
            best_edge_set = set(es_init)
        
        # Run local search with penalty
        time_left = deadline - 15 - time.time()
        if time_left < 2:
            break
        
        # Allocate time based on remaining budget
        search_time = time.time() + min(time_left * 0.6, 8)
        final_score, final_set = local_search_penalty(es_init, search_time)
        
        # Update global best
        if final_score < best_score:
            best_score = final_score
            best_edge_set = final_set
    
    if best_edge_set is None:
        # Fallback
        best_edge_set = set()
        for u in range(8):
            for v in range(8, 16):
                best_edge_set.add((u, v))
        best_edge_set.add((0, 1))
    
    edge_list = [[u, v] for (u, v) in best_edge_set]
    return {'n': n, 'edges': edge_list}
