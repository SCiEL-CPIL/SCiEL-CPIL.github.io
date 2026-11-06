import numpy as np
import time
import random

def construct():
    n = 20
    start_time = time.time()
    time_limit = 140.0

    def is_valid(A):
        # Precompute pair intersections
        # pair_inter[i, j] is a boolean vector of common neighbors of i and j
        # We only need i < j
        pair_inter = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_inter[(i, j)] = A[i] & A[j]
        
        for i in range(n):
            for j in range(i + 1, n):
                pi = pair_inter[(i, j)]
                for k in range(j + 1, n):
                    if np.sum(pi & A[k]) >= 3:
                        return False
        return True

    def get_violation(A):
        # Similar to is_valid but returns violation
        pair_inter = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_inter[(i, j)] = A[i] & A[j]
        
        for i in range(n):
            for j in range(i + 1, n):
                pi = pair_inter[(i, j)]
                for k in range(j + 1, n):
                    common = pi & A[k]
                    if np.sum(common) >= 3:
                        common_indices = np.where(common)[0]
                        c = random.choice(common_indices.tolist())
                        return (i, j, k, c)
        return None

    def count_edges(A):
        return np.sum(A) // 2

    best_edges = 0
    best_A = None

    while time.time() - start_time < time_limit:
        # Initialize
        p = random.uniform(0.2, 0.6)
        A = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    A[i, j] = True
                    A[j, i] = True
        
        # Repair
        repair_steps = 0
        max_repair_steps = 1000
        while repair_steps < max_repair_steps:
            if time.time() - start_time > time_limit:
                break
            viol = get_violation(A)
            if viol is None:
                break
            i, j, k, c = viol
            edge_to_remove = random.choice([(i, c), (j, c), (k, c)])
            A[edge_to_remove[0], edge_to_remove[1]] = False
            A[edge_to_remove[1], edge_to_remove[0]] = False
            repair_steps += 1
            
        if time.time() - start_time > time_limit:
            break
            
        if repair_steps >= max_repair_steps:
            continue
            
        # Local Search
        stuck_counter = 0
        while time.time() - start_time < time_limit:
            success = False
            attempts = 0
            while attempts < 100:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j or A[i, j]:
                    attempts += 1
                    continue
                
                A[i, j] = True
                A[j, i] = True
                
                if is_valid(A):
                    success = True
                    stuck_counter = 0
                    break
                else:
                    A[i, j] = False
                    A[j, i] = False
                    attempts += 1
            
            if not success:
                stuck_counter += 1
                if stuck_counter > 50:
                    # Perturbation
                    edges = np.where(A)
                    if len(edges[0]) > 0:
                        idx = random.randint(0, len(edges[0]) - 1)
                        u = edges[0][idx]
                        v = edges[1][idx]
                        A[u, v] = False
                        A[v, u] = False
                        stuck_counter = 0
                    else:
                        break
            else:
                current_edges = count_edges(A)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()

    if best_A is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j]:
                result.append([i, j])
    return result
