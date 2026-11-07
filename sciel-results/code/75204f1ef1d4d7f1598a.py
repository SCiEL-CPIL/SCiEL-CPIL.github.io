import numpy as np
import time
import random
from scipy.sparse.csgraph import minimum_spanning_tree

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix combining inverse distance and MST structure.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    demands = np.array(demands)
    D = np.asarray(distance_matrix)
    
    # 1. Compute MST of the complete graph defined by D
    # minimum_spanning_tree returns a sparse matrix where entry (i,j) is the weight if (i,j) is in MST
    mst_sparse = minimum_spanning_tree(D)
    mst_dense = mst_sparse.toarray()
    
    # Create a binary mask for MST edges
    # Note: MST is undirected, so if (i,j) is in MST, (j,i) is also considered for the graph structure
    # In ACO, we usually have a directed graph or symmetric. Here we treat it as symmetric encouragement.
    mst_mask = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and mst_dense[i, j] > 0:
                mst_mask[i, j] = 1.0

    # 2. Base Heuristic: Inverse distance
    # Handle diagonal and zeros
    base_heuristic = np.full((n, n), 1e-9)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = D[i, j]
                if d > 0:
                    base_heuristic[i, j] = 1.0 / d
                else:
                    base_heuristic[i, j] = 1e-9

    # 3. Search for best combination
    # We blend base_heuristic and mst_mask.
    # heuristic[i,j] = (base_heuristic[i,j] ** gamma) * (1 + alpha * mst_mask[i,j])
    # Or a linear combination in log space?
    # Let's try: eta[i,j] = (1/d)^gamma + beta * mst_mask[i,j]
    # The MST term acts as a constant boost to specific edges.
    
    # Candidate parameters:
    # gamma: exponent for distance
    # beta: weight of the MST boost relative to the max distance heuristic
    
    # Normalize base heuristic to [0, 1] to make beta comparable?
    # Or just scale beta.
    # Let's use: eta[i,j] = (1/d)^gamma * (1 + alpha * mst_mask[i,j])
    # If alpha is large, MST edges are heavily favored.
    
    candidate_gammas = [1.0, 1.5, 2.0]
    candidate_alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    best_score = float('inf')
    best_params = (1.0, 0.0)
    
    def evaluate_params(gamma, alpha):
        """
        Runs a fast greedy proxy to evaluate the heuristic quality.
        """
        # Construct heuristic for this param set
        heuristic = np.power(base_heuristic, gamma)
        # Apply MST boost
        # We want to boost MST edges.
        # heuristic[i,j] *= (1 + alpha * mst_mask[i,j])
        # But careful: if mst_mask is 0, factor is 1. If 1, factor is (1+alpha).
        boost_factor = 1.0 + alpha * mst_mask
        heuristic = heuristic * boost_factor
        
        # Ensure positivity
        heuristic = np.where(heuristic > 0, heuristic, 1e-9)
        
        min_cost = float('inf')
        # Run 5 fast greedy tours
        for _ in range(5):
            current = 0
            current_load = 0.0
            total_distance = 0.0
            unvisited = set(range(1, n))
            
            while True:
                feasible = []
                weights = []
                for j in unvisited:
                    if demands[j] <= (capacity - current_load):
                        feasible.append(j)
                        weights.append(heuristic[current, j])
                
                if not feasible:
                    total_distance += D[current, 0]
                    break
                
                total_w = sum(weights)
                if total_w == 0:
                    next_node = random.choice(feasible)
                else:
                    r = random.random() * total_w
                    cumsum = 0.0
                    next_node = feasible[-1]
                    for idx, node in enumerate(feasible):
                        cumsum += weights[idx]
                        if r <= cumsum:
                            next_node = node
                            break
                
                total_distance += D[current, next_node]
                current_load += demands[next_node]
                unvisited.remove(next_node)
                current = next_node
                
                if not unvisited:
                    total_distance += D[current, 0]
                    break
            
            if total_distance < min_cost:
                min_cost = total_distance
        
        return min_cost

    for gamma in candidate_gammas:
        for alpha in candidate_alphas:
            if time.time() - start_time > 250.0:
                break
            cost = evaluate_params(gamma, alpha)
            if cost < best_score:
                best_score = cost
                best_params = (gamma, alpha)
        
        if time.time() - start_time > 250.0:
            break

    gamma, alpha = best_params
    
    # Construct final heuristic
    final_heuristic = np.power(base_heuristic, gamma)
    boost_factor = 1.0 + alpha * mst_mask
    final_heuristic = final_heuristic * boost_factor
    
    # Safety checks
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    return final_heuristic
