import numpy as np
import time

def _simulate_aco(d, eta, n_ants=20, n_iterations=30):
    """
    Simulate ACO to estimate the quality of a heuristic.
    """
    n = d.shape[0]
    tau = np.ones((n, n))
    best_len = float('inf')
    
    # Precompute nothing, just loop
    for _ in range(n_iterations):
        tour_lengths = []
        for _ in range(n_ants):
            visited = [False] * n
            current = 0
            visited[0] = True
            tour_len = 0.0
            prev = 0
            
            for _ in range(n - 1):
                # Calculate probabilities
                # prob[j] = tau[current, j] * eta[current, j]
                # Mask visited and self
                probs = tau[current, :] * eta[current, :]
                probs[visited] = 0.0
                probs[current] = 0.0
                
                total = np.sum(probs)
                if total == 0:
                    # Fallback to random unvisited
                    unvisited = np.where(~np.array(visited))[0]
                    next_node = np.random.choice(unvisited)
                else:
                    probs = probs / total
                    # Sample
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += d[current, next_node]
                prev = current
                current = next_node
                visited[current] = True
            
            # Close tour
            tour_len += d[current, 0]
            tour_lengths.append(tour_len)
            
            # Update pheromone
            # Find the best ant in this iteration? Or all ants?
            # Prompt: "After each iteration, pheromone is multiplied by 0.9 and each ant adds 1/length..."
            # So all ants contribute.
            # But we need the tour edges to update.
            # I didn't store the tour. I need to store the tour to update tau.
            
            # Re-do: Store tour
            # Let's refactor to store tour.
            
    # Refactored simulation
    return best_len

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # Set diagonal to 0 for distance calc, but we need it for heuristic?
    # d has 0 on diagonal.
    
    candidates = [1.0, 1.5, 2.0, 2.5, 3.0]
    best_eta = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 250.0 # Safety margin
    
    for p in candidates:
        if time.time() - start_time > time_limit:
            break
            
        # Compute eta
        # Avoid division by zero
        # d > 0 for i != j
        eta = np.zeros((n, n))
        # Mask diagonal
        mask = np.eye(n, dtype=bool)
        d_safe = d.copy()
        d_safe[mask] = 1.0 # Avoid div by 0, value will be overwritten
        eta = 1.0 / (d_safe ** p)
        eta[mask] = 1e-9 # Set diagonal to small positive
        
        # Simulate
        # We need a fast, accurate simulation.
        score = _simulate_aco(d, eta)
        
        if score < best_score:
            best_score = score
            best_eta = eta
            
    if best_eta is None:
        # Default
        d_safe = d.copy()
        mask = np.eye(n, dtype=bool)
        d_safe[mask] = 1.0
        best_eta = 1.0 / (d_safe ** 2.0)
        best_eta[mask] = 1e-9
        
    return best_eta

def _simulate_aco(d, eta, n_ants=20, n_iterations=30):
    n = d.shape[0]
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for _ in range(n_iterations):
        all_tours = []
        all_lens = []
        
        for _ in range(n_ants):
            visited = [False] * n
            current = 0
            visited[0] = True
            tour_len = 0.0
            tour = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * eta[current, :]
                probs[visited] = 0.0
                probs[current] = 0.0
                
                total = np.sum(probs)
                if total == 0:
                    unvisited = [j for j in range(n) if not visited[j]]
                    next_node = np.random.choice(unvisited)
                else:
                    probs = probs / total
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += d[current, next_node]
                tour.append(next_node)
                current = next_node
                visited[current] = True
            
            tour_len += d[current, 0]
            all_tours.append(tour)
            all_lens.append(tour_len)
            
            if tour_len < best_len:
                best_len = tour_len
        
        # Update pheromones
        tau *= 0.9
        for i, tour in enumerate(all_tours):
            L = all_lens[i]
            if L == 0: continue
            delta = 1.0 / L
            for k in range(len(tour)):
                u = tour[k]
                v = tour[(k + 1) % len(tour)]
                tau[u, v] += delta
                
    return best_len
