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
    
    def tour_length(t):
        return sum(d[t[i], t[(i + 1) % n]] for i in range(n))

    best_tour = tour[:]
    best_len = tour_length(best_tour)
    
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
        # Try moving 1 node
        for i in range(n):
            u = best_tour[i]
            prev = best_tour[(i - 1) % n]
            next_ = best_tour[(i + 1) % n]
            # Remove u
            old_cost = d[prev, u] + d[u, next_]
            new_tour = best_tour[:]
            new_tour.pop(i)
            # Try inserting u at all positions
            for j in range(n - 1):
                if j == (i - 1) % n or j == i: # Skip original position or adjacent
                    continue
                # Insert u after j
                # The new tour will have length n
                # We need to check the cost change
                # Original: ... j -> j+1 ...
                # New: ... j -> u -> j+1 ...
                # Cost change: d[j, u] + d[u, j+1] - d[j, j+1]
                # But indices shift. Let's just reconstruct and check if it's cheap enough.
                # For N=50, reconstruction is cheap.
                temp_tour = new_tour[:]
                temp_tour.insert(j + 1, u)
                # Ensure we don't create a self-loop or invalid structure
                # Actually, simple check:
                if len(set(temp_tour)) != n: continue
                new_len = tour_length(temp_tour)
                if new_len < best_len - 1e-10:
                    best_tour = temp_tour
                    best_len = new_len
                    improved = True
                    break
        # Try moving 2 nodes (segment of 2)
        for i in range(n):
            u1 = best_tour[i]
            u2 = best_tour[(i + 1) % n]
            prev = best_tour[(i - 1) % n]
            next_ = best_tour[(i + 2) % n]
            
            new_tour = best_tour[:]
            # Remove u1, u2
            # Handle circular wrap
            if i + 1 == n:
                # u1 is last, u2 is first
                new_tour.pop(n-1)
                new_tour.pop(0)
            else:
                new_tour.pop(i+1)
                new_tour.pop(i)
                
            # Try inserting segment (u1, u2) at all positions
            for j in range(n - 2):
                # Insert after j in new_tour (which has length n-2)
                # New length will be n
                temp_tour = new_tour[:]
                temp_tour.insert(j + 1, u1)
                temp_tour.insert(j + 2, u2)
                
                if len(set(temp_tour)) != n: continue
                new_len = tour_length(temp_tour)
                if new_len < best_len - 1e-10:
                    best_tour = temp_tour
                    best_len = new_len
                    improved = True
                    break
    return best_tour

def _greedy_tour_length(d, eta):
    """Fast proxy for ACO performance: Greedy tour using heuristic."""
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    current = 0
    visited[0] = True
    tour_len = 0.0
    
    for _ in range(n - 1):
        # Pick unvisited j that maximizes eta[current, j]
        eta_row = eta[current].copy()
        eta_row[visited] = -np.inf
        eta_row[current] = -np.inf
        next_node = np.argmax(eta_row)
        tour_len += d[current, next_node]
        current = next_node
        visited[current] = True
        
    tour_len += d[current, 0]
    return tour_len

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 300.0
    
    # 1. Find a high quality tour
    best_tour = None
    best_tour_len = float('inf')
    
    # Use multiple restarts for NN
    for i in range(50): # Increased restarts
        if time.time() > deadline - 10.0: break
        start_node = i % n
        tour = _nearest_neighbor_tour(d, start=start_node)
        tour = _two_opt(tour, d)
        tour = _or_opt(tour, d)
        length = sum(d[tour[k], tour[(k + 1) % n]] for k in range(n))
        if length < best_tour_len:
            best_tour_len = length
            best_tour = tour[:]
            
    if best_tour is None:
        best_tour = _nearest_neighbor_tour(d, start=0)
        best_tour = _two_opt(best_tour, d)
        best_tour = _or_opt(best_tour, d)
        best_tour_len = sum(d[best_tour[k], best_tour[(k + 1) % n]] for k in range(n))

    # 2. Generate candidate heuristics
    d_safe = d.copy()
    d_safe[mask] = 1.0
    
    candidates = []
    
    # Expanded parameter search
    p_values = [1.5, 2.0, 2.5, 3.0, 4.0]
    boost_values = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    for p in p_values:
        base_eta = 1.0 / (d_safe ** p)
        for boost in boost_values:
            eta = base_eta.copy()
            if boost > 1.0:
                for k in range(n):
                    u = best_tour[k]
                    v = best_tour[(k + 1) % n]
                    eta[u, v] *= boost
                    eta[v, u] *= boost
            eta[mask] = 1e-9
            candidates.append(eta)
            
    # 3. Evaluate candidates with Greedy Tour Length
    best_eta = candidates[0]
    best_score = float('inf')
    
    for eta in candidates:
        if time.time() > deadline - 5.0: break
        score = _greedy_tour_length(d, eta)
        if score < best_score:
            best_score = score
            best_eta = eta.copy()
            
    return best_eta
