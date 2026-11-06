import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Simulated Annealing on Vertex Permutations with Spectral Initialization.
    1. Initialize ordering using spectral centrality and degree heuristics.
    2. Use Swap and Insertion moves to explore the permutation space.
    3. Fast greedy simulation using NumPy boolean masks.
    4. Adaptive restart mechanism to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Pre-compute neighbor masks? adj_bool rows are already masks.
    
    def simulate_greedy(order):
        selected = 0
        available = np.ones(n, dtype=bool)
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            # Vectorized neighbor update
            available[adj_bool[v]] = False
            
            # Early exit
            if not available.any():
                break
                
        return selected

    # 1. Initialization: Spectral + Degree
    # Power iteration for principal eigenvector
    spec_vec = np.ones(n) / np.sqrt(n)
    for _ in range(10):
        spec_vec = adj_bool @ spec_vec
        norm = np.linalg.norm(spec_vec)
        if norm > 0:
            spec_vec /= norm
        else:
            break
            
    # Normalize spectral vector to [0, 1] range for combination
    spec_min = np.min(spec_vec)
    spec_max = np.max(spec_vec)
    if spec_max > spec_min:
        spec_norm = (spec_vec - spec_min) / (spec_max - spec_min)
    else:
        spec_norm = np.zeros(n)
        
    # Combine: Low degree is good (high score). 
    # Spectral centrality: Hubs have high spec. We want to pick hubs LAST (low score).
    # So score = -degree - alpha * spec_norm
    # Let's use alpha = 0.5
    init_score = -degree - 0.5 * spec_norm
    init_score += np.random.normal(0, 0.1, n)
    
    current_order = np.argsort(-init_score, kind='stable')
    current_size = simulate_greedy(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # 2. Simulated Annealing
    T = 1.0
    cooling = 0.9995
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Generate Neighbor
        # 80% Swap, 20% Insertion
        if np.random.random() < 0.8:
            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)
            if idx1 == idx2:
                continue
            new_order = current_order.copy()
            new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
        else:
            i = np.random.randint(n)
            v = current_order[i]
            # Use list for efficient insertion
            ord_list = list(current_order)
            ord_list.pop(i)
            j = np.random.randint(n)
            ord_list.insert(j, v)
            new_order = np.array(ord_list)
            
        new_size = simulate_greedy(new_order)
        
        delta = new_size - current_size
        
        if delta > 0:
            current_order = new_order
            current_size = new_size
            no_improve_count = 0
            
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
        else:
            # Accept worse move
            if T > 1e-6:
                # Optimization: avoid exp underflow
                if delta > -10 * T:
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        current_order = new_order
                        current_size = new_size
            no_improve_count += 1
            
        # Cool down
        T *= cooling
        
        # Restart logic
        if no_improve_count > max_no_improve or T < 0.01:
            # Restart from best or new random
            if np.random.random() < 0.5:
                current_order = best_order.copy()
                current_size = best_size
            else:
                # New random/spectral init with higher noise
                init_score = -degree - 0.5 * spec_norm + np.random.normal(0, 0.5, n)
                current_order = np.argsort(-init_score, kind='stable')
                current_size = simulate_greedy(current_order)
            
            T = 0.5
            no_improve_count = 0

    # Convert best_order to scores
    scores = np.zeros(n)
    scores[best_order] = n - np.arange(n)
    
    return scores
