import time
import random
import numpy as np
from itertools import combinations

def count_violations(adj, n):
    """Count K5 cliques and independent sets of size 5."""
    if n < 5:
        return 0
    k5_count = 0
    ind5_count = 0
    for combo in combinations(range(n), 5):
        is_clique = True
        is_ind = True
        for i in range(5):
            for j in range(i + 1, 5):
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
            k5_count += 1
        if is_ind:
            ind5_count += 1
    return k5_count + ind5_count

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = q
    best_edges = seed_edges
    
    # Try to find larger valid graphs
    for target_n in range(q + 1, 43):
        if time.time() > deadline - 20:
            break
        
        found_valid = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize adjacency matrix
            adj = np.zeros((target_n, target_n), dtype=bool)
            
            # Copy Paley edges
            for i, j in seed_edges:
                adj[i, j] = True
                adj[j, i] = True
            
            # Add random edges for new vertices and between new vertices
            for i in range(q, target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            # Also add some random edges between old and new vertices
            for i in range(q):
                for j in range(q, target_n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            # Simulated annealing
            current_violations = count_violations(adj, target_n)
            
            if current_violations == 0:
                found_valid = True
                break
            
            # SA parameters
            temperature = 1.0
            cooling_rate = 0.9995
            steps = 2000
            
            for step in range(steps):
                if time.time() > deadline - 5:
                    break
                
                if current_violations == 0:
                    found_valid = True
                    break
                
                # Pick a random edge to flip
                i, j = np.random.randint(0, target_n, 2)
                if i == j:
                    continue
                
                # Flip the edge
                old_val = adj[i, j]
                adj[i, j] = not old_val
                adj[j, i] = not old_val
                
                new_violations = count_violations(adj, target_n)
                delta = new_violations - current_violations
                
                # Accept if better, or with probability exp(-delta/T) if worse
                if delta < 0 or (temperature > 0.01 and random.random() < np.exp(-delta / temperature)):
                    current_violations = new_violations
                else:
                    # Revert
                    adj[i, j] = old_val
                    adj[j, i] = old_val
                
                temperature *= cooling_rate
            
            if current_violations == 0:
                found_valid = True
                break
        
        if found_valid:
            best_n = target_n
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj[i, j]:
                        edges.append((i, j))
            best_edges = edges
        else:
            # If we can't find a valid graph for this size, stop
            break
    
    return best_n, best_edges
