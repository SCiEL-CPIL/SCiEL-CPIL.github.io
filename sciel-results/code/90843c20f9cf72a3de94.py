import numpy as np
import time

def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO TSP.
    Combines inverse-squared distance with a strong structural prior from a 2-opt improved tour.
    """
    start_time = time.time()
    # Deadline guard: 330s limit, but this function should be < 1s.
    # We use 300s as a safe internal limit for the whole evaluation context if needed,
    # but primarily we just ensure we return quickly.
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[1.0]])
        
    dm = np.maximum(distance_matrix, 1e-9)
    
    # 1. Base heuristic: Inverse squared distance
    # Strong preference for short edges.
    heuristic = 1.0 / (dm ** 2)
    
    # 2. Build a high-quality tour using Nearest Neighbor + 2-opt
    # Nearest Neighbor construction
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        # Vectorized find for speed, though N=50 is small
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # 2-opt Improvement
    # Multiple passes to ensure local optimality
    improved = True
    max_passes = 50
    pass_count = 0
    
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Early exit if time is running out (unlikely for N=50)
        if time.time() - start_time > 290:
            break
            
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            
            for j in range(i + 2, n):
                # Skip if j is the last node and i is the first, to avoid crossing the closing edge
                # in a way that doesn't make sense for simple 2-opt on a linear array representation
                # of a circular tour, unless handled carefully.
                # Standard 2-opt: reverse segment tour[i+1 ... j]
                # Edges removed: (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                # Edges added: (tour[i], tour[j]) and (tour[i+1], tour[j+1])
                
                if i == 0 and j == n - 1:
                    continue
                
                c = tour[j]
                d = tour[(j + 1) % n]
                
                len_old = distance_matrix[a, b] + distance_matrix[c, d]
                len_new = distance_matrix[a, c] + distance_matrix[b, d]
                
                if len_new < len_old - 1e-10:
                    # Reverse the segment from i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    break # Restart inner loop? Or break outer? 
                    # Breaking inner and letting outer continue is okay, but restarting 
                    # the whole pass is cleaner for convergence. 
                    # For speed, we just break the inner loop and continue to next i.
                    # Actually, standard practice is to restart the pass or just continue.
                    # Let's break the inner loop and continue the outer loop.
            if improved:
                # We modified the tour, so the indices for subsequent j in this i might be invalid
                # or the logic of "next best move" changes. 
                # To be safe and fast, we can just break and restart the while loop pass.
                break
    
    # 3. Construct Final Heuristic
    # Boost edges from the improved tour
    # A strong boost helps the ACO converge quickly to this good region.
    boost_factor = 15.0
    
    for k in range(n):
        u = tour[k]
        v = tour[(k + 1) % n]
        heuristic[u, v] *= boost_factor
        heuristic[v, u] *= boost_factor
    
    # Ensure all values are positive and finite
    # Replace any non-finite or non-positive values with a small epsilon
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
