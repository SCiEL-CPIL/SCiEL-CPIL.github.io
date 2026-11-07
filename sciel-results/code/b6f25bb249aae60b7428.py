import numpy as np
import time

def _nearest_neighbor_tour(d, start=0):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    tour = [start]
    for _ in range(n - 1):
        d_row = d[current].copy()
        d_row[visited] = np.inf
        d_row[current] = np.inf
        next_node = np.argmin(d_row)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(tour, d):
    n = len(tour)
    if n <= 2:
        return tour
    
    best_tour = tour[:]
    best_len = sum(d[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                u1 = best_tour[i - 1]
                u2 = best_tour[i]
                u3 = best_tour[j]
                u4 = best_tour[(j + 1) % n]
                
                old_len = d[u1, u2] + d[u3, u4]
                new_len = d[u1, u3] + d[u2, u4]
                
                if new_len < old_len - 1e-10:
                    best_tour = best_tour[:]
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_len = tour_length(best_tour)
                    improved = True
    return best_tour

def _or_opt(tour, d):
    """Or-opt: move single nodes to other positions in the tour."""
    n = len(tour)
    if n <= 3:
        return tour
    
    def tour_length(t):
        return sum(d[t[i], t[(i + 1) % n]] for i in range(n))
    
    best_tour = tour[:]
    best_len = tour_length(best_tour)
    improved = True
    
    while improved:
        improved = False
        for i in range(n):
            u = best_tour[i]
            prev = best_tour[(i - 1) % n]
            nxt = best_tour[(i + 1) % n]
            
            # Remove u
            tour_without_u = best_tour[:i] + best_tour[i+1:]
            
            # Try inserting u after each position j (in the tour without u)
            for j in range(len(tour_without_u)):
                # Skip positions adjacent to where u was removed
                if j == (i - 1) % n or j == i % len(tour_without_u):
                    continue
                    
                a = tour_without_u[j]
                b = tour_without_u[(j + 1) % len(tour_without_u)]
                
                new_tour = tour_without_u[:j+1] + [u] + tour_without_u[j+1:]
                new_len = tour_length(new_tour)
                
                if new_len < best_len - 1e-10:
                    best_tour = new_tour
                    best_len = new_len
                    improved = True
                    break
    
    return best_tour

def _local_search(tour, d):
    """Apply 2-opt then Or-opt iteratively."""
    tour = _two_opt(tour, d)
    tour = _or_opt(tour, d)
    tour = _two_opt(tour, d)
    tour = _or_opt(tour, d)
    return tour

def _aco_sim(d, eta, n_ants=10, n_iter=10, seed=42):
    """Fast ACO simulation to evaluate heuristic quality."""
    np.random.seed(seed)
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    
    best_len = float('inf')
    pheromone = np.ones((n, n))
    pheromone[mask] = 0
    
    for _ in range(n_iter):
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour = [0]
            
            for _ in range(n - 1):
                probs = pheromone[current] * eta[current]
                probs[visited] = 0
                probs[current] = 0
                total = probs.sum()
                if total < 1e-15:
                    unvisited = np.where(~visited)[0]
                    next_node = unvisited[np.random.randint(len(unvisited))]
                else:
                    probs = probs / total
                    next_node = np.random.choice(n, p=probs)
                
                tour.append(next_node)
                visited[next_node] = True
                current = next_node
            
            tour_len = sum(d[tour[k], tour[(k + 1) % n]] for k in range(n))
            if tour_len < best_len:
                best_len = tour_len
            
            pheromone *= 0.9
            for k in range(n):
                u = tour[k]
                v = tour[(k + 1) % n]
                pheromone[u, v] += 1.0 / tour_len
                pheromone[v, u] += 1.0 / tour_len
    
    return best_len

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # 1. Find high-quality tour via multi-restart NN + local search
    best_tour = None
    best_tour_len = float('inf')
    
    # Use many restarts (aggressive, but fast enough for n=50)
    n_restarts = 500
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        start_node = i % n
        tour = _nearest_neighbor_tour(d, start=start_node)
        tour = _local_search(tour, d)
        length = sum(d[tour[k], tour[(k + 1) % n]] for k in range(n))
        if length < best_tour_len:
            best_tour_len = length
            best_tour = tour[:]
    
    if best_tour is None:
        best_tour = _nearest_neighbor_tour(d, start=0)
        best_tour = _local_search(best_tour, d)
        best_tour_len = sum(d[best_tour[k], best_tour[(k + 1) % n]] for k in range(n))
    
    # 2. Generate candidate heuristics with different boost factors
    d_safe = d.copy()
    d_safe[mask] = 1.0
    base_eta = 1.0 / (d_safe ** 2.0)
    
    candidates = []
    for boost in [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        eta = base_eta.copy()
        for k in range(n):
            u = best_tour[k]
            v = best_tour[(k + 1) % n]
            eta[u, v] *= boost
            eta[v, u] *= boost
        eta[mask] = 1e-9
        candidates.append(eta)
    
    # 3. Evaluate candidates with fast ACO simulation
    best_eta = candidates[0]
    best_score = float('inf')
    
    for eta in candidates:
        if time.time() > deadline - 10.0:
            break
        score = _aco_sim(d, eta)
        if score < best_score:
            best_score = score
            best_eta = eta.copy()
    
    return best_eta
