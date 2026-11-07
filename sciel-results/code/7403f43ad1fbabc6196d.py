import numpy as np
import time
import sys

def _sa_tsp(distance_matrix, n_restarts=20, max_iter=500, t_start=1.0, t_end=0.001, seed=42):
    """
    Runs Multi-Start Simulated Annealing to find a good tour.
    Returns the best tour found (list of nodes) and its length.
    """
    n = distance_matrix.shape[0]
    rng = np.random.default_rng(seed)
    best_tour = None
    best_len = float('inf')
    
    for restart in range(n_restarts):
        # Random initial tour
        tour = list(range(n))
        rng.shuffle(tour)
        
        # Calculate initial length
        current_len = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        
        # Simulated Annealing
        for i in range(max_iter):
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (i / max_iter)
            
            # Pick two random indices to swap (2-opt move)
            i_idx = rng.integers(1, n - 1)
            j_idx = rng.integers(i_idx + 1, n)
            
            # Calculate delta
            # Old edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
            # New edges: (i-1, j), (i, j+1) ... wait, standard 2-opt reversal
            # Let's use a simpler 2-opt: reverse segment between i and j
            # Actually, let's just do a simple swap of two cities to keep it fast, 
            # or standard 2-opt reversal. 2-opt is better for TSP.
            
            # 2-opt: reverse tour[i:j+1]
            # Old edges affected:
            # 1. (tour[i-1], tour[i])
            # 2. (tour[j], tour[(j+1)%n])
            # New edges:
            # 1. (tour[i-1], tour[j])
            # 2. (tour[i], tour[(j+1)%n])
            
            a = tour[i_idx - 1]
            b = tour[i_idx]
            c = tour[j_idx]
            d = tour[(j_idx + 1) % n]
            
            delta = distance_matrix[a, c] + distance_matrix[b, d] - (distance_matrix[a, b] + distance_matrix[c, d])
            
            if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-9)):
                # Accept move
                tour[i_idx:j_idx+1] = reversed(tour[i_idx:j_idx+1])
                current_len += delta
                
                if current_len < best_len:
                    best_len = current_len
                    best_tour = tour.copy()
        
        # Check if this restart's best is better than global best
        # (Already updated inside loop, but ensure we capture the final state if it was the best)
        if current_len < best_len:
            best_len = current_len
            best_tour = tour.copy()
            
    return best_tour, best_len

def _two_opt_improve(tour, distance_matrix, max_passes=10):
    """
    Apply standard 2-opt improvement until no improvement is found.
    """
    n = len(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a = tour[i-1]
                b = tour[i]
                c = tour[j]
                d = tour[(j+1)%n]
                
                delta = distance_matrix[a, c] + distance_matrix[b, d] - (distance_matrix[a, b] + distance_matrix[c, d])
                if delta < 0:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = distance_matrix.shape[0]
    
    # 1. Find a high-quality tour using Multi-Start SA
    # We don't need the absolute best, just a very good one to guide the ACO
    # 50 nodes is small, so SA is very fast.
    best_tour, _ = _sa_tsp(distance_matrix, n_restarts=50, max_iter=200, seed=123)
    
    # 2. Further polish with 2-opt
    best_tour = _two_opt_improve(best_tour, distance_matrix, max_passes=5)
    
    # 3. Construct Heuristic Matrix
    # Strategy: Strong backbone
    # Base heuristic: inverse distance (weak)
    # Backbone heuristic: very high value for edges in the SA tour
    
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    # Weak base: 1/distance
    base_heur = 1.0 / safe_dist
    
    # Strong backbone: 100.0 for edges in the tour
    # The ACO probability is proportional to tau * h.
    # If h is 100 for tour edges and 1 for others, ants are 100x more likely to pick tour edges initially.
    # This effectively "seeds" the pheromone with a strong prior.
    
    heuristic = base_heur.copy()
    
    # Mark tour edges
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        heuristic[u, v] = 100.0
        # Note: The problem states "travel direction only" for pheromone, 
        # but heuristic is symmetric in distance matrix usually, but ACO uses directed edges.
        # The task says "edge-attractiveness matrix ... same shape".
        # The ACO rule: prob proportional to tau[i,j] * heuristic[i,j].
        # Since the tour is a cycle, we should boost both directions? 
        # Actually, the tour is a closed loop. The ant travels i->j.
        # If we only boost i->j for the specific direction in our tour, 
        # an ant traveling j->i might not get the boost.
        # However, TSP tours are symmetric in distance, but the "tour" found by SA is a sequence.
        # Usually, heuristics are symmetric (1/d). 
        # If we make it asymmetric, we might hurt ants that traverse the edge in the "wrong" direction.
        # But a good tour can be traversed in either direction.
        # Let's boost both directions to be safe, or just one?
        # The prompt says "for each step i to j of its closed tour, in travel direction only" for PHEROMONE.
        # For HEURISTIC, it just says "edge-attractiveness matrix".
        # Standard TSP heuristics are symmetric. 
        # If I boost u->v but not v->u, and the optimal tour uses v->u, I lose.
        # But since the tour is a cycle, the set of undirected edges is what matters for the "backbone".
        # Let's boost both u->v and v->u to ensure the structure is attractive regardless of direction.
        heuristic[v, u] = 100.0

    # Sanity checks
    heuristic[heuristic <= 0] = 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    
    return heuristic
