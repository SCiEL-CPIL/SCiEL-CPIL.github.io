"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO on TSP.
    Combines inverse distance with a boost for edges in a locally optimized tour.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    dm = distance_matrix + 1e-9
    
    # 1. Base heuristic: inverse distance. 
    # Power of 1.5 is often a good balance between greediness and exploration.
    # We use 1.5 instead of 2 to allow slightly more exploration.
    heuristic = 1.0 / (dm ** 1.5)
    
    # 2. Generate initial greedy tour from node 0
    unvisited = set(range(1, n))
    current = 0
    tour = [0]
    
    for _ in range(n - 1):
        # Find nearest unvisited neighbor
        # Using numpy for speed
        distances = distance_matrix[current, list(unvisited)]
        idx = np.argmin(distances)
        best_next = list(unvisited)[idx]
        tour.append(best_next)
        unvisited.remove(best_next)
        current = best_next
    
    # Close the tour
    # tour is a list of nodes, we need edges
    # We will run 2-opt to improve this tour
    
    def tour_length(tour):
        length = 0.0
        for i in range(len(tour)):
            length += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return length
    
    # 3. 2-Opt local search to improve the greedy tour
    # This uses the time budget to find a better set of edges
    improved_tour = tour
    best_length = tour_length(improved_tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Skip if i and j are adjacent in the tour
                if j == i + 1:
                    continue
                # Calculate delta length for 2-opt move
                # Current edges: (i-1, i) and (j, j+1)
                # New edges: (i-1, j) and (i, j+1)
                # Note: indices in tour list
                a = improved_tour[i - 1]
                b = improved_tour[i]
                c = improved_tour[j]
                d = improved_tour[(j + 1) % n]
                
                old_dist = distance_matrix[a, b] + distance_matrix[c, d]
                new_dist = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_dist < old_dist - 1e-10:
                    # Reverse the segment between i and j
                    improved_tour = improved_tour[:i] + improved_tour[i:j+1][::-1] + improved_tour[j+1:]
                    best_length = new_dist + (best_length - old_dist)
                    improved = True
                    # Check time budget
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
    
    # 4. Boost edges in the improved tour
    # We boost both directions to maintain symmetry
    boost_factor = 1.8  # Slightly higher boost for the optimized tour
    
    for i in range(n):
        u = improved_tour[i]
        v = improved_tour[(i + 1) % n]
        heuristic[u, v] *= boost_factor
        heuristic[v, u] *= boost_factor
        
    # 5. Optional: Add a small boost for k-nearest neighbors to provide broader guidance
    # This helps in cases where the 2-opt tour might be too specific
    k = 3
    for i in range(n):
        neighbors = np.argsort(distance_matrix[i])[:k]
        for j in neighbors:
            if j != i:
                heuristic[i, j] *= 1.1
                heuristic[j, i] *= 1.1
                
    # 6. Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 7. Time check
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # If we somehow exceeded time, return the base heuristic
        return 1.0 / (dm ** 1.5)
        
    return heuristic
# EVOLVE-BLOCK-END
