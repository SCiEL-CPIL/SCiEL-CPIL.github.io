import numpy as np
import time
import random

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s kill
    
    n = distance_matrix.shape[0]
    
    # Helper: Calculate tour length
    def tour_length(tour):
        return sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
    
    # Helper: 2-Opt improvement
    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if i == 0 and j == n - 1:
                        continue
                    # Calculate delta for 2-opt move
                    # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%n])
                    # New edges: (tour[i-1], tour[j]) and (tour[i], tour[(j+1)%n])
                    # Note: tour is a list, indices wrap around for the closed loop
                    a = tour[(i - 1) % n]
                    b = tour[i]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    
                    current_cost = distance_matrix[a, b] + distance_matrix[c, d]
                    new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                    
                    if new_cost < current_cost - 1e-10:
                        # Reverse the segment tour[i..j]
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
        return tour

    # Helper: Or-Opt (move a single city)
    def or_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                # Try moving tour[i] to every other position
                city = tour[i]
                tour.pop(i)
                best_pos = i
                best_cost = tour_length(tour) + distance_matrix[tour[-1], city] + distance_matrix[city, tour[0]]
                
                for j in range(n - 1):
                    # Insert city at position j
                    tour.insert(j, city)
                    cost = tour_length(tour)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = j
                    tour.pop(j)
                
                if best_pos != i:
                    tour.pop(i)
                    tour.insert(best_pos, city)
                    improved = True
        return tour

    # Helper: Generate a random NN tour
    def random_nn_tour():
        visited = np.zeros(n, dtype=bool)
        start = random.randint(0, n - 1)
        visited[start] = True
        tour = [start]
        current = start
        
        for _ in range(n - 1):
            dists = distance_matrix[current, :].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
            
        return tour

    # 1. Generate an ensemble of high-quality tours
    ensemble_tours = []
    
    # Add a few deterministic NN tours from different starts
    for start in [0, 1, 2]:
        t = random_nn_tour()
        t = two_opt(t)
        t = or_opt(t)
        ensemble_tours.append(t)
        
    # Add random NN tours until time budget is nearly exhausted or we have enough
    # 50 cities is small, 2-opt is fast. Let's aim for ~20-50 tours.
    max_tours = 50
    tour_count = 3
    
    while tour_count < max_tours:
        if time.time() > deadline - 10:
            break
            
        t = random_nn_tour()
        t = two_opt(t)
        # Or-opt is slower, so maybe apply it less frequently or only if time permits
        if time.time() < deadline - 20:
            t = or_opt(t)
        
        ensemble_tours.append(t)
        tour_count += 1
        
    # 2. Construct Heuristic Matrix
    # Base: Inverse distance
    H = 1.0 / (distance_matrix + 1e-9)
    H[np.diag_indices_from(H)] = 0.0
    
    # Frequency count
    freq = np.zeros((n, n))
    
    for t in ensemble_tours:
        for i in range(n):
            u = t[i]
            v = t[(i + 1) % n]
            freq[u, v] += 1
            freq[v, u] += 1  # Symmetric
    
    # Normalize frequency to [0, 1]
    if freq.max() > 0:
        freq_norm = freq / freq.max()
    else:
        freq_norm = np.zeros((n, n))
        
    # Blend: 60% Frequency, 40% Inverse Distance
    # This ensures that even if an edge isn't in the top tours, short edges are still attractive
    blended = 0.6 * freq_norm + 0.4 * (H / H.max())
    
    # Amplify differences using power law
    # Higher power makes good edges (high freq) much more attractive than bad ones
    power = 2.5
    H_final = blended ** power
    
    # Ensure no zeros (replace with small epsilon) and handle diagonal
    H_final[np.diag_indices_from(H_final)] = 0.0
    H_final = np.maximum(H_final, 1e-9)
    
    # Normalize to a reasonable scale? 
    # The ACO uses P ~ (tau * eta)^alpha. 
    # If eta is very large, it might dominate tau. 
    # If eta is very small, it might be ignored.
    # Scaling H_final so max is around 1.0 or similar might be good, 
    # but relative differences are what matter most.
    # Let's keep the raw power-law values but ensure they are positive.
    
    return H_final
