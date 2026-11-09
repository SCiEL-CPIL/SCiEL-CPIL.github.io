import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def _two_opt(tour, dist):
    """
    Deterministic 2-opt local search.
    tour: list of nodes
    dist: distance matrix
    Returns improved tour length.
    """
    n = len(tour) - 1 # tour has n+1 elements, last is first
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            # Edge (i, i+1) and (j, j+1) in tour
            # We reverse segment i+1 to j
            # Indices in tour: tour[i], tour[i+1], ..., tour[j], tour[j+1]
            # New edges: (i, j) and (i+1, j+1)
            # Old edges: (i, i+1) and (j, j+1)
            # Note: tour is 0-indexed list of length n+1
            # i ranges from 1 to n-2 (since i+1 < n)
            # j ranges from i+1 to n-1 (since j+1 <= n)
            
            # Optimization: Only check i=1 to n-2, j=i+1 to n-1
            # Actually, standard 2-opt:
            # For i in 1..n-2
            #   For j in i+1..n-1
            #     delta = d[tour[i], tour[j]] + d[tour[i+1], tour[j+1]] - d[tour[i], tour[i+1]] - d[tour[j], tour[j+1]]
            
            for j in range(i + 1, n):
                # tour[j+1] is valid because j < n => j+1 <= n
                # If j == n-1, j+1 == n, which is tour[0] (start/end)
                
                a = tour[i]
                b = tour[i+1]
                c = tour[j]
                d_node = tour[j+1]
                
                delta = dist[a, c] + dist[b, d_node] - dist[a, b] - dist[c, d_node]
                
                if delta < -1e-10:
                    # Reverse segment from i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
    return sum(dist[tour[i], tour[i+1]] for i in range(n))

def _evaluate_heuristic_greedy_2opt(dist, heuristic):
    """
    Simulates the ACO behavior in a deterministic way:
    1. Greedy construction using the heuristic (simulating high pheromone convergence or just the heuristic bias).
       Since ACO starts with pheromone=1, the first iteration is effectively guided by the heuristic.
       Later iterations are guided by pheromone + heuristic.
       A good heuristic should produce a good initial tour that 2-opt can refine.
       
    We simulate the "ideal" ant path: always pick the most attractive unvisited node (argmax of heuristic).
    This represents the limit of ACO as pheromone influence dominates or heuristic is very strong.
    Then we apply 2-opt to clean up the result.
    """
    n = dist.shape[0]
    # Greedy tour construction
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    
    # Mask diagonal
    h_masked = heuristic.copy()
    np.fill_diagonal(h_masked, -np.inf)
    
    while unvisited:
        # Find best next node
        # attractions = h_masked[current, list(unvisited)]
        # To avoid list conversion overhead, we can just argmax the row and check if visited
        # But since we remove from unvisited, we need to mask visited.
        
        # Efficient way:
        # Create a temp array of -inf for visited, heuristic values for unvisited
        # But n=50 is small.
        
        best_val = -np.inf
        best_node = -1
        
        # Vectorized check
        # Get indices of unvisited
        unvisited_list = list(unvisited)
        vals = h_masked[current, unvisited_list]
        idx = np.argmax(vals)
        best_node = unvisited_list[idx]
        
        tour.append(best_node)
        unvisited.remove(best_node)
        current = best_node
        
    tour.append(0)
    
    # Apply 2-Opt
    final_length = _two_opt(tour, dist)
    
    return final_length

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    d = distance_matrix.copy()
    n = d.shape[0]
    
    # Generate a diverse set of candidate heuristics
    candidates = []
    
    # 1. Basic Inverse Distance Powers
    for power in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        candidates.append(1.0 / (d + 1e-9) ** power)
        
    # 2. Exponential Decays
    for beta in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
        candidates.append(np.exp(-beta * d))
        
    # 3. Offset Inverse
    for offset in [0.01, 0.05, 0.1, 0.2, 0.5]:
        candidates.append(1.0 / (d + offset))
        
    # 4. Neighbor-Boosted Inverse Distance
    # Boost top k neighbors
    h_base = 1.0 / (d + 1e-9)
    
    for k in [2, 3, 5, 10]:
        for boost in [2.0, 5.0, 10.0, 20.0]:
            h_boosted = h_base.copy()
            for i in range(n):
                row = d[i].copy()
                row[i] = np.inf
                # Get top k
                top_k_idx = np.argpartition(row, k)[:k]
                h_boosted[i, top_k_idx] *= boost
            candidates.append(h_boosted)
            
    # 5. Angle-aware heuristic (requires coordinate recovery or approximation)
    # Since we only have distance matrix, we can approximate "straightness" 
    # using the fact that for points in a unit square, the sum of distances to neighbors 
    # might indicate position. 
    # Alternatively, we can use a proxy: 
    # h[i,j] = 1/d[i,j] * (1 / (d[i, k] + d[j, k] - d[i,j]) ) for some k? No, that's complex.
    
    # Let's try a "degree-based" heuristic.
    # Nodes with many close neighbors might be central.
    # h[i,j] = 1/d[i,j] * (degree_score[i] + degree_score[j])?
    
    # Calculate "local density" or "centrality"
    # Sum of inverse distances to all other nodes
    density = np.sum(1.0 / (d + 1e-9), axis=1)
    
    for weight in [0.1, 0.5, 1.0, 2.0]:
        h_density = h_base * (density[:, None] + density[None, :]) ** weight
        candidates.append(h_density)
        
    # 6. Rank-based heuristic
    # Rank distances for each row, smaller rank (closer) -> higher heuristic
    # h[i,j] = 1 / rank[i,j]
    ranks = np.zeros((n, n))
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        # Rank: 1 is closest
        order = np.argsort(row)
        rank_vals = np.empty(n)
        rank_vals[order] = np.arange(1, n+1)
        ranks[i, :] = rank_vals
    
    for power in [1.0, 2.0, 3.0]:
        candidates.append(1.0 / (ranks ** power))
        
    # 7. Hybrid: Inverse distance * Rank penalty
    # h[i,j] = (1/d[i,j]) * (1/rank[i,j])
    candidates.append(h_base / (ranks + 1e-9))
    
    # Filter invalid candidates (non-finite, wrong shape)
    valid_candidates = []
    for h in candidates:
        if h.shape != d.shape:
            continue
        if not np.all(np.isfinite(h)):
            continue
        # Ensure positive
        h_pos = np.maximum(h, 1e-9)
        valid_candidates.append(h_pos)
        
    if not valid_candidates:
        # Fallback
        return np.ones_like(d)
        
    # Evaluate candidates
    best_heuristic = valid_candidates[0]
    best_score = float('inf')
    
    # We can afford to evaluate more candidates now because _evaluate_heuristic_greedy_2opt is fast
    # n=50, 2-opt is O(N^2) * iterations. Greedy is O(N^2).
    # Total per candidate ~ 50^2 * 10 = 25000 ops. Very fast.
    
    for h in valid_candidates:
        if time.time() > deadline:
            break
            
        score = _evaluate_heuristic_greedy_2opt(d, h)
        
        if score < best_score:
            best_score = score
            best_heuristic = h
            
    return best_heuristic
# EVOLVE-BLOCK-END
