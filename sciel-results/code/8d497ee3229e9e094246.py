import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    # Initialize heuristic matrix
    eta = np.zeros((n, n))
    
    # Precompute distances to depot (node 0)
    d_depot = distance_matrix[0, :]
    
    # 1. Depot to Customer (i=0)
    # Use inverse distance to encourage starting with nearby nodes or just standard NN
    # Actually, for CVRP, starting with a node that allows good clustering is key.
    # 1/d is a safe bet.
    for j in range(1, n):
        eta[0, j] = 1.0 / (distance_matrix[0, j] + 1e-9)
        
    # 2. Customer to Depot (j=0)
    # We want to discourage returning to depot unless necessary.
    # Set to a very low value.
    for i in range(1, n):
        eta[i, 0] = 1e-9

    # 3. Customer to Customer (i,j != 0)
    # Use Savings heuristic: d(i,0) + d(0,j) - d(i,j)
    # This is high when i and j are close to each other relative to their distance to depot.
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                eta[i, j] = 0.0 # Self-loop not allowed/needed
            else:
                savings = d_depot[i] + d_depot[j] - distance_matrix[i, j]
                # Ensure positive
                if savings < 1e-9:
                    eta[i, j] = 1e-9
                else:
                    eta[i, j] = savings

    # Normalize or scale?
    # The ACO algorithm uses tau * eta.
    # Pheromone starts at 1.
    # If eta values are ~0.5-1.0, probabilities are reasonable.
    # If eta values are ~10-20 (for 1/d), they are also reasonable.
    # Savings are typically smaller than 1/d.
    # Let's scale Savings to be comparable to 1/d to avoid the depot edges dominating too much or too little.
    # Max 1/d is roughly 1/min_dist. Min dist is ~0.05? 1/0.05 = 20.
    # Max Savings is roughly 2*max_dist ~ 1.0.
    # So 1/d is ~20x larger.
    # This means the first step (Depot->Customer) will be heavily weighted by heuristic compared to subsequent steps.
    # This is actually good: it locks in a "good" starting cluster center.
    # Subsequent steps rely more on pheromone and relative savings.
    
    # However, if eta[i,j] is 0.5 and eta[0,j] is 2.0, the first step is 4x more likely to be influenced by heuristic.
    # This seems fine.
    
    # One potential issue: If an ant is at i, and all other unvisited j have low savings (e.g. i is isolated),
    # it might be forced to go to depot? No, it picks from feasible.
    # If no feasible customer fits, it returns to depot.
    # The low eta[i,0] ensures that if a customer *does* fit, it is preferred over depot.
    
    return eta
