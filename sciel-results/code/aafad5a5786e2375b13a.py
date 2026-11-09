import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    n = len(prizes)
    d = distance_matrix
    
    # Initialize heuristic with a very small base value to avoid zeros
    # We will build up the heuristic based on tour quality
    heuristic = np.full((n, n), 1e-9)
    
    # Strategy: Find high-value tours using a fast, randomized greedy method
    # and reinforce the edges used in those tours.
    
    # 1. Precompute valid moves from each node i to j given remaining budget
    # But budget depends on path, so we do it on the fly in the greedy search.
    
    best_tour_prize = 0.0
    best_tour_edges = None
    
    # Use a simple random seed for internal search
    rng = np.random.RandomState(42)
    
    # Perform multiple greedy tours to find a good one
    # Since n=50 is small, we can afford a few iterations
    for iter_idx in range(50):
        if time.time() > deadline - 10:
            break
            
        current = 0
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        length = 0.0
        prize_sum = prizes[0]
        path = [0]
        
        while True:
            best_j = -1
            best_score = -np.inf
            
            # Find candidate nodes
            for j in range(1, n):
                if not visited[j]:
                    # Check if we can go to j and return to 0
                    if length + d[current, j] + d[j, 0] <= max_len:
                        # Score: prize / distance to j
                        # Add a small random noise to break ties and explore
                        score = prizes[j] / (d[current, j] + 1e-9) + rng.rand() * 0.01
                        if score > best_score:
                            best_score = score
                            best_j = j
            
            if best_j == -1:
                break
            
            # Move to best_j
            length += d[current, best_j]
            visited[best_j] = True
            prize_sum += prizes[best_j]
            path.append(best_j)
            current = best_j
            
        if prize_sum > best_tour_prize:
            best_tour_prize = prize_sum
            best_tour_edges = path[:]
            
    # Construct heuristic matrix based on the best tour found
    # Edges in the best tour get high attractiveness
    # Other edges get low attractiveness
    # This guides the ACO to find similar tours
    
    if best_tour_edges:
        # Reinforce edges in the best tour
        for i in range(len(best_tour_edges) - 1):
            u = best_tour_edges[i]
            v = best_tour_edges[i+1]
            heuristic[u, v] = 100.0  # High attractiveness
            heuristic[v, u] = 100.0  # Symmetric reinforcement
            
        # Also reinforce edges from last node back to 0? 
        # The problem says "until no node qualifies", and return to 0 is implicit in the constraint.
        # The ACO moves from current to j. The heuristic is for [current, j].
        # So we mainly need to reinforce the forward edges of the tour.
        
        # Additionally, we can compute a prize-based heuristic for all nodes
        # and blend it with the tour-specific reinforcement
        # Let's try a pure tour-reinforcement first, but ensure it's not zero for non-tour edges
        # Actually, if we only set tour edges to 100 and others to 1e-9, the ACO will be very myopic.
        # Let's use a baseline heuristic: prize[j] / d[i,j]
        # and multiply by a large factor if (i,j) is in the best tour.
        
        baseline = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    baseline[i, j] = prizes[j] / (d[i, j] + 1e-9)
        
        # Normalize baseline to [0, 1]
        min_b = baseline.min()
        max_b = baseline.max()
        if max_b > min_b:
            baseline = (baseline - min_b) / (max_b - min_b)
        else:
            baseline = np.ones_like(baseline) * 0.5
            
        # Create final heuristic
        heuristic = baseline * 10.0
        
        # Boost tour edges
        for i in range(len(best_tour_edges) - 1):
            u = best_tour_edges[i]
            v = best_tour_edges[i+1]
            heuristic[u, v] *= 100.0
            heuristic[v, u] *= 100.0
            
    else:
        # Fallback: simple prize/distance heuristic
        heuristic = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    heuristic[i, j] = prizes[j] / (d[i, j] + 1e-9)
        
        min_h = heuristic.min()
        max_h = heuristic.max()
        if max_h > min_h:
            heuristic = (heuristic - min_h) / (max_h - min_h) * 10.0
        else:
            heuristic = np.ones_like(heuristic)
            
    # Ensure no zeros or negative values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
