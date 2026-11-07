import numpy as np
import time

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    
    # 1. Compute a high-quality tour using Nearest Neighbor + 2-opt
    # This serves as a structural prior.
    
    # Nearest Neighbor
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                if distance_matrix[current, j] < min_dist:
                    min_dist = distance_matrix[current, j]
                    next_node = j
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # 2-opt Improvement
    # We will perform a few passes of 2-opt
    # For N=50, this is very fast.
    improved = True
    passes = 0
    max_passes = 50 # Safety limit
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue # Don't cross the start/end edge in this simple implementation unless we handle wrap-around carefully
                c = tour[j]
                d = tour[j + 1] if j + 1 < n else tour[0]
                
                # Check if swapping (a,b) and (c,d) with (a,c) and (b,d) improves length
                # Current length contribution: d(a,b) + d(c,d)
                # New length contribution: d(a,c) + d(b,d)
                # Note: If i=0, a is start. If j=n-1, d is start.
                # The tour is circular.
                
                # Let's be precise with indices.
                # Tour is a list of nodes.
                # Edge 1: tour[i] -> tour[i+1]
                # Edge 2: tour[j] -> tour[(j+1)%n]
                
                # If we reverse the segment between i+1 and j, the edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]) are replaced by (tour[i], tour[j]) and (tour[i+1], tour[j+1]).
                
                # Let's just calculate lengths directly.
                len_old = distance_matrix[a, b] + distance_matrix[c, d]
                len_new = distance_matrix[a, c] + distance_matrix[b, d]
                
                if len_new < len_old - 1e-10:
                    # Reverse the segment from i+1 to j
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    # Break inner loop to restart scanning? Or just continue?
                    # Restarting is safer for 2-opt convergence.
                    # But for speed, we can just continue.
                    # Let's break to restart the outer loop for a cleaner 2-opt pass.
                    break
            if improved:
                break
    
    # 2. Construct Heuristic Matrix
    # Base heuristic: inverse distance
    # Add small epsilon to avoid division by zero on diagonal if any, though diagonal is not used for moves.
    # The problem states: "Heuristic entries that are not positive become 1e-9".
    # So we should ensure all entries are positive.
    
    # Initialize with 1e-9 or similar, then fill
    # Or just 1 / (d + eps)
    eps = 1e-9
    eta = 1.0 / (distance_matrix + eps)
    
    # Boost edges from the good tour
    # The tour is a list of nodes in order.
    # Edges are (tour[k], tour[(k+1)%n])
    boost_factor = 10.0 # Strong prior
    
    for k in range(n):
        u = tour[k]
        v = tour[(k + 1) % n]
        eta[u, v] *= boost_factor
        eta[v, u] *= boost_factor
    
    # Ensure all values are positive and finite
    # The 1/(d+eps) ensures positivity for d>=0.
    # The boost multiplies positive values.
    
    return eta
