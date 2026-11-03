import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    # Generate all weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    vectors = []
    
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.float32)
    N = len(V) # 5280
    
    # Build conflict graph
    # Conflict if dot product > 2
    dots = V @ V.T
    conflict_mask = dots > 2.0
    np.fill_diagonal(conflict_mask, False)
    
    # Get neighbors for each node
    adj = [np.where(conflict_mask[i])[0] for i in range(N)]
    
    # Greedy Initialization
    x = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if not np.any(x[adj[i]]):
            x[i] = 1
            
    best_x = x.copy()
    best_size = np.sum(x)
    
    # Simulated Quantum Annealing
    M = 1000.0
    T = 10.0
    Delta = 10.0
    cooling_T = 0.99995
    cooling_D = 0.99995
    
    steps = 5000000
    
    x = best_x.copy()
    neighbor_sum = np.zeros(N, dtype=np.int32)
    for i in range(N):
        neighbor_sum[i] = np.sum(x[adj[i]])
        
    current_energy = -np.sum(x) + M * 0.5 * np.sum(x * neighbor_sum)
    best_energy = current_energy
    best_x = x.copy()
    best_size = np.sum(x)
    
    # SQA Loop
    # Use numpy random for speed
    # Generate random indices in batches? No, sequential dependency.
    # But we can use np.random.randint
    
    for step in range(steps):
        T *= cooling_T
        Delta *= cooling_D
        
        i = np.random.randint(N)
        
        if x[i] == 1:
            delta_E = -1.0 + M * neighbor_sum[i]
        else:
            delta_E = 1.0 - M * neighbor_sum[i]
            
        accept = False
        if delta_E < 0:
            accept = True
        else:
            denom = T + Delta
            if denom > 1e-9:
                prob = np.exp(-delta_E / denom)
            else:
                prob = 0.0
            if np.random.random() < prob:
                accept = True
                
        if accept:
            x[i] = 1 - x[i]
            
            if x[i] == 1:
                neighbor_sum[adj[i]] += 1
            else:
                neighbor_sum[adj[i]] -= 1
            
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
                best_size = np.sum(x)
                
    indices = np.where(best_x == 1)[0]
    result_vectors = V[indices].astype(np.int64)
    
    return result_vectors
