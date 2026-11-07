import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9):
    """
    Fast internal ACO simulation matching the contract's parameters (20 ants, 30 iters).
    This is used to evaluate heuristic quality.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    # Pre-compute unvisited masks? No, dynamic.
    # Vectorize probability calculation where possible, but the loop is sequential.
    
    for it in range(n_iterations):
        tour_lengths = np.zeros(n_ants)
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                # Calculate probabilities
                # heuristic and tau are (n, n)
                # We need probs for current row, excluding visited
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[np.random.randint(len(unvisited_indices))]
                else:
                    # Normalize
                    probs = probs / total_prob
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths[ant] = tour_len
            
            # Pheromone update
            deposit = 1.0 / tour_len
            # Update tau for each step in tour_path
            # tour_path is [0, ..., n-1], closed tour means last to 0
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        # Evaporation
        tau *= rho
        
        current_best = np.min(tour_lengths)
        if current_best < best_tour_len:
            best_tour_len = current_best
            
    return best_tour_len

def _two_opt_improve(tour, dist):
    """
    Apply 2-opt improvement to a tour.
    tour: list of node indices
    dist: distance matrix
    Returns improved tour and length.
    """
    n = len(tour)
    if n < 4:
        return tour, sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Check if reversing segment i+1 to j improves tour
                # Current edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, i+1)
                # Wait, standard 2-opt:
                # Remove (i-1, i) and (j, j+1)
                # Add (i-1, j) and (i, j+1)
                # Reverse segment i..j
                
                u1 = tour[(i - 1) % n]
                v1 = tour[i]
                u2 = tour[j]
                v2 = tour[(j + 1) % n]
                
                delta = dist[u1, u2] + dist[v1, v2] - dist[u1, v1] - dist[u2, v2]
                if delta < -1e-10:
                    # Reverse segment i to j
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    
    # Calculate final length
    total_len = 0.0
    for i in range(n):
        total_len += dist[tour[i], tour[(i + 1) % n]]
    return tour, total_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    # Strategy: Graph-theoretic backbone
    # 1. Compute MST
    # 2. Generate a tour from MST (e.g., double-tree or nearest neighbor on MST)
    # 3. Improve with 2-opt
    # 4. Use edges from this high-quality tour as "backbone" with high heuristic value
    # 5. Optionally blend with inverse distance for non-backbone edges
    
    # 1. Compute MST using Prim's algorithm (O(n^2) for dense graph, fine for n=50)
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = safe_dist[0].copy()
    
    mst_edges = []
    for _ in range(n - 1):
        # Find min distance from MST to non-MST
        min_dist[in_mst] = np.inf
        v = np.argmin(min_dist)
        u = np.argmin(safe_dist[v, in_mst]) # This is not quite right for finding the u that connected to v
        # Actually, we need to track which node in MST connected to v.
        # Let's re-implement Prim's properly to get edges.
        pass
    
    # Proper Prim's
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = safe_dist[0].copy()
    parent = np.zeros(n, dtype=int)
    
    for _ in range(n - 1):
        min_dist[in_mst] = np.inf
        v = np.argmin(min_dist)
        in_mst[v] = True
        # Find parent u
        # The parent is the node in MST with min distance to v
        # We can just find it by checking all in_mst nodes
        u_candidates = np.where(in_mst)[0]
        u = u_candidates[np.argmin(safe_dist[u_candidates, v])]
        parent[v] = u
        mst_edges.append((u, v))
        min_dist = np.minimum(min_dist, safe_dist[v, :])
        
    # 2. Generate initial tour
    # Nearest neighbor heuristic starting from 0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    tour = [0]
    
    for _ in range(n - 1):
        # Find nearest unvisited
        dists = safe_dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    # 3. Improve with 2-opt
    tour, tour_len = _two_opt_improve(tour, safe_dist)
    
    # 4. Construct heuristic
    # High value for edges in the improved tour
    # Low value for others
    # We can try a few variations:
    # A) Pure backbone: 10 for tour edges, 1 for others
    # B) Backbone + Inverse Dist: (10 if tour edge else 1) * (1/dist)
    # C) Soft backbone: blend
    
    base_heur = 1.0 / safe_dist
    
    candidates = []
    
    # Candidate A: Pure backbone
    h_backbone = np.ones((n, n))
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        h_backbone[u, v] = 10.0
        h_backbone[v, u] = 10.0 # Symmetric? TSP is usually symmetric, but contract says "travel direction only" for pheromone.
                                # Heuristic is usually symmetric in TSP.
                                # However, the ACO uses directed pheromone.
                                # The heuristic matrix is (n, n).
                                # Let's keep it symmetric for now, as distance is symmetric.
    candidates.append(h_backbone)
    
    # Candidate B: Backbone * Inverse Dist
    h_blend = h_backbone * base_heur
    candidates.append(h_blend)
    
    # Candidate C: Just Inverse Dist (baseline)
    candidates.append(base_heur)
    
    # Candidate D: MST edges only (weaker backbone)
    h_mst = np.ones((n, n))
    for u, v in mst_edges:
        h_mst[u, v] = 5.0
        h_mst[v, u] = 5.0
    candidates.append(h_mst * base_heur)
    
    # Evaluate candidates
    # We have ~290s.
    # _fast_aco_simulate with 20 ants, 30 iters, n=50.
    # Each ant: 49 steps.
    # Total steps per iteration: 20 * 49 = 980.
    # Total steps per sim: 30 * 980 = 29400 steps.
    # Each step involves numpy ops. Should be fast.
    # Let's estimate time.
    
    best_heur = base_heur
    best_score = float('inf')
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Ensure validity
        h = h.copy()
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        
        score = _fast_aco_simulate(safe_dist, h, n_ants=20, n_iterations=30)
        
        if score < best_score:
            best_score = score
            best_heur = h
            
    # If we have time, try to refine the backbone strength
    if time.time() < deadline - 30:
        # Try different backbone strengths
        for strength in [5.0, 20.0, 50.0]:
            h_strong = np.ones((n, n))
            for i in range(n):
                u = tour[i]
                v = tour[(i + 1) % n]
                h_strong[u, v] = strength
                h_strong[v, u] = strength
            h_strong *= base_heur
            h_strong[h_strong <= 0] = 1e-9
            h_strong[~np.isfinite(h_strong)] = 1e-9
            
            score = _fast_aco_simulate(safe_dist, h_strong, n_ants=20, n_iterations=30)
            if score < best_score:
                best_score = score
                best_heur = h_strong
                
    return best_heur
