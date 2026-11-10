import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_aco_sim(d, h, n_ants=10, n_iters=5, seed=42):
    """
    Simulates a fast ACO run to evaluate the heuristic.
    Returns the best tour length found.
    """
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    
    # Pheromone matrix
    tau = np.ones((n, n))
    
    # Heuristic matrix (ensure positive)
    eta = h.copy()
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # Precompute powers if needed? 
    # The problem statement says: probability proportional to pheromone[i, j] * heuristic[i, j]
    # Note: It does NOT mention alpha or beta powers explicitly in the probability formula 
    # "proportional to pheromone[i, j] * heuristic[i, j]".
    # However, standard ACO usually has powers. The prompt says:
    # "picks the next unvisited node with probability proportional to pheromone[i, j] * heuristic[i, j]"
    # This implies alpha=1, beta=1.
    
    best_len = float('inf')
    
    for _ in range(n_iters):
        # Each ant starts at 0
        # We simulate all ants in parallel? Or loop?
        # N=50, Ants=10. Looping is fine.
        
        # To speed up, we can vectorize the selection for all ants at each step?
        # But the visited masks are different per ant.
        # Given N=50 is small, a simple loop is likely faster than complex vectorization overhead 
        # unless we use numba (not available).
        
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour = [0]
            tour_len = 0.0
            
            for _ in range(n - 1):
                # Get probabilities for unvisited nodes
                # prob[i] = tau[current, i] * eta[current, i]
                # Mask visited
                probs = tau[current, :] * eta[current, :]
                probs[visited] = 0.0
                
                # Normalize
                total_prob = probs.sum()
                if total_prob == 0:
                    # Fallback to random unvisited
                    unvisited_idx = np.where(~visited)[0]
                    next_node = rng.choice(unvisited_idx)
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour.append(next_node)
                tour_len += d[current, next_node]
                visited[next_node] = True
                current = next_node
            
            # Close the tour
            tour_len += d[current, 0]
            tour.append(0)
            
            if tour_len < best_len:
                best_len = tour_len
            
            # Update pheromone for this ant
            # "each ant adds 1/length to pheromone[i, j] for each step i to j"
            # The prompt says: "After each iteration, pheromone is multiplied by 0.9 and each ant adds 1/length..."
            # This implies we accumulate pheromone from all ants in an iteration, then decay?
            # Or decay then add?
            # "After each iteration, pheromone is multiplied by 0.9 and each ant adds 1/length to pheromone[i, j]..."
            # Usually: tau = rho * tau + sum(delta_tau). Here rho=0.9.
            # Let's stick to the text: 
            # 1. Multiply by 0.9
            # 2. Add 1/length for each edge
            # This is done "After each iteration".
            # So we should accumulate the delta_tau for all ants in this iteration, 
            # then apply tau = 0.9 * tau + delta_tau.
            
            # Wait, the loop above is per ant. 
            # I need to restructure to collect deltas for the iteration.
            
            # Let's restart the loop structure to be correct with the prompt's iteration definition.
            # The prompt says: "20 ants for 30 iterations".
            # So:
            # for iter in range(30):
            #   for ant in range(20):
            #     construct tour
            #     store delta_tau
            #   tau = 0.9 * tau + sum(delta_tau)
            
            # My current loop structure is wrong for the pheromone update timing.
            # I will fix the simulation function.
            pass
            
    # Since the logic above was flawed regarding pheromone update timing, 
    # I will rewrite the simulation correctly below.
    
    # Correct Simulation
    best_len = float('inf')
    tau = np.ones((n, n))
    
    for _ in range(n_iters):
        delta_tau = np.zeros((n, n))
        
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_nodes = [0]
            tour_len = 0.0
            
            for _ in range(n - 1):
                probs = tau[current, :] * eta[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob == 0:
                    unvisited_idx = np.where(~visited)[0]
                    next_node = rng.choice(unvisited_idx)
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_nodes.append(next_node)
                tour_len += d[current, next_node]
                visited[next_node] = True
                current = next_node
            
            tour_len += d[current, 0]
            tour_nodes.append(0)
            
            if tour_len < best_len:
                best_len = tour_len
            
            # Add to delta_tau
            # "adds 1/length to pheromone[i, j] for each step i to j of its closed tour, in travel direction only"
            dep = 1.0 / tour_len
            for i in range(n):
                u = tour_nodes[i]
                v = tour_nodes[i+1]
                delta_tau[u, v] += dep
        
        # Update pheromone
        tau = 0.9 * tau + delta_tau
        
    return best_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 250.0 # Safety margin
    
    d = distance_matrix.copy()
    n = d.shape[0]
    
    # Generate candidates
    candidates = []
    
    # 1. Inverse distance variants
    candidates.append(1.0 / (d + 1e-9))
    candidates.append(1.0 / (d**1.5 + 1e-9))
    candidates.append(1.0 / (d**2 + 1e-9))
    
    # 2. Exponential decay
    candidates.append(np.exp(-d))
    
    # 3. Neighbor-based boosts on inverse distance
    h_base = 1.0 / (d + 1e-9)
    h_top3 = h_base.copy()
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        top3_idx = np.argpartition(row, 3)[:3]
        h_top3[i, top3_idx] *= 10.0
    candidates.append(h_top3)
    
    h_top5 = h_base.copy()
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        top5_idx = np.argpartition(row, 5)[:5]
        h_top5[i, top5_idx] *= 5.0
    candidates.append(h_top5)

    # Evaluate candidates using Fast ACO Simulation
    # We use a small number of ants and iterations to keep it fast.
    # N=50, Ants=10, Iters=5.
    # 10*5*50 steps = 2500 steps per candidate.
    # 6 candidates * 2500 = 15000 steps.
    # Each step involves a numpy choice and some array ops.
    # This should be well under 1 second per candidate.
    
    best_heuristic = candidates[0]
    best_score = float('inf')
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Use fixed seed for reproducibility in evaluation
        score = _fast_aco_sim(d, h, n_ants=10, n_iters=5, seed=42)
        
        if score < best_score:
            best_score = score
            best_heuristic = h
            
    return best_heuristic
# EVOLVE-BLOCK-END
