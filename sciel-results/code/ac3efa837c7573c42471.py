import numpy as np
import time

def _nearest_neighbor_tour(d, start=0):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    tour = [start]
    for _ in range(n - 1):
        # Find nearest unvisited
        d_row = d[current].copy()
        d_row[visited] = np.inf
        d_row[current] = np.inf
        next_node = np.argmin(d_row)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(tour, d):
    n = len(tour)
    improved = True
    # Map tour to indices for faster access? No, just use the list.
    # Precompute tour length?
    def tour_length(t):
        length = 0.0
        for i in range(n):
            length += d[t[i], t[(i + 1) % n]]
        return length

    best_tour = tour[:]
    best_len = tour_length(best_tour)
    
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                # Reverse segment i..j
                new_tour = best_tour[:]
                new_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                
                # Calculate delta
                # Edges involved:
                # (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                # Note: indices in tour
                u1 = best_tour[i - 1]
                u2 = best_tour[i]
                u3 = best_tour[j]
                u4 = best_tour[(j + 1) % n]
                
                # Old length parts
                old_len = d[u1, u2] + d[u3, u4]
                # New length parts
                new_len = d[u1, u3] + d[u2, u4]
                
                if new_len < old_len - 1e-10:
                    best_tour = new_tour
                    best_len = tour_length(best_tour)
                    improved = True
    
    return best_tour

def _simulate_aco_fast(d, eta, n_ants=20, n_iterations=30, seed=None):
    """
    Fast simulation of the ACO algorithm to estimate heuristic quality.
    """
    if seed is not None:
        np.random.seed(seed)
        
    n = d.shape[0]
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for _ in range(n_iterations):
        all_tours = []
        all_lens = []
        
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            current = 0
            visited[0] = True
            tour_len = 0.0
            tour = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * eta[current, :]
                probs[visited] = 0.0
                probs[current] = 0.0
                
                total = np.sum(probs)
                if total <= 1e-15:
                    unvisited = np.where(~visited)[0]
                    next_node = unvisited[np.random.randint(len(unvisited))]
                else:
                    cum_probs = np.cumsum(probs)
                    cum_probs /= total
                    r = np.random.rand()
                    next_node = np.searchsorted(cum_probs, r)
                    if next_node >= n:
                        next_node = n - 1
                
                tour_len += d[current, next_node]
                tour.append(next_node)
                current = next_node
                visited[current] = True
            
            tour_len += d[current, 0]
            all_tours.append(tour)
            all_lens.append(tour_len)
            
            if tour_len < best_len:
                best_len = tour_len
        
        tau *= 0.9
        for i, tour in enumerate(all_tours):
            L = all_lens[i]
            if L < 1e-15: continue
            delta = 1.0 / L
            for k in range(len(tour)):
                u = tour[k]
                v = tour[(k + 1) % len(tour)]
                tau[u, v] += delta
                
    return best_len

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # Mask diagonal
    mask = np.eye(n, dtype=bool)
    
    # 1. Compute a high-quality tour using NN + 2-Opt with multiple restarts
    # This is a "solution-based" heuristic bias.
    best_tour = None
    best_tour_len = float('inf')
    
    start_time = time.time()
    time_limit = 20.0 # Time budget for heuristic construction
    
    # Try a few restarts of NN to get a good base tour
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
        # Random start node? Or just node 0?
        # The ACO always starts at 0. But the tour is a cycle, so start node doesn't matter for the cycle itself.
        # However, NN from different starts might yield different 2-Opt basins.
        start_node = restart % n
        tour = _nearest_neighbor_tour(d, start=start_node)
        tour = _two_opt(tour, d)
        
        # Calculate length
        length = 0.0
        for i in range(n):
            length += d[tour[i], tour[(i + 1) % n]]
        
        if length < best_tour_len:
            best_tour_len = length
            best_tour = tour[:]
            
    if best_tour is None:
        best_tour = _nearest_neighbor_tour(d, start=0)
        best_tour = _two_opt(best_tour, d)
        best_tour_len = sum(d[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
        
    # 2. Construct Heuristic Matrix
    # Base: Inverse distance power law
    # Boost: Edges in the best tour get a large boost
    
    # We want eta[i,j] to be large for attractive edges.
    # Let eta[i,j] = base[i,j] + boost * indicator(i,j in tour)
    # Or multiplicative?
    # The ACO uses tau * eta.
    # If we make eta huge for tour edges, ants will strongly prefer them.
    # If we make eta small for others, ants will avoid them.
    
    # Let's try a combination:
    # eta[i,j] = (1/d[i,j]^p) * (1 + A * I[i,j in tour])
    # Or simpler:
    # eta[i,j] = 1/d[i,j]^p
    # eta[tour_edges] *= boost_factor
    
    # What is a good boost factor?
    # If boost is too high, ACO gets stuck in the initial tour and cannot improve if it's not optimal.
    # But 2-Opt is quite strong.
    # Let's try boost_factor = 10.0
    
    p = 1.5
    d_safe = d.copy()
    d_safe[mask] = 1.0
    
    eta = 1.0 / (d_safe ** p)
    eta[mask] = 1e-9
    
    # Boost tour edges
    boost = 5.0
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        eta[u, v] *= boost
        eta[v, u] *= boost # Heuristic is usually symmetric, but ACO uses directed?
        # The task says "edge-attractiveness matrix".
        # ACO step: prob proportional to tau[i,j] * eta[i,j].
        # tau is updated for i->j.
        # If eta is symmetric, it's fine.
        
    # 3. Validate and possibly refine using a quick simulation?
    # The previous program did a search over p.
    # Here we have a strong structural bias.
    # We can quickly test this eta and maybe a few variations of boost.
    
    candidates = []
    
    # Candidate 1: Boost 5.0
    candidates.append(eta.copy())
    
    # Candidate 2: Boost 10.0
    eta2 = 1.0 / (d_safe ** p)
    eta2[mask] = 1e-9
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        eta2[u, v] *= 10.0
        eta2[v, u] *= 10.0
    candidates.append(eta2)
    
    # Candidate 3: Boost 2.0
    eta3 = 1.0 / (d_safe ** p)
    eta3[mask] = 1e-9
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        eta3[u, v] *= 2.0
        eta3[v, u] *= 2.0
    candidates.append(eta3)
    
    # Candidate 4: Pure distance (baseline)
    eta4 = 1.0 / (d_safe ** p)
    eta4[mask] = 1e-9
    candidates.append(eta4)
    
    # Evaluate candidates with a fast ACO simulation
    # We need to be careful with time.
    # One simulation takes ~0.1-0.5s?
    # Let's measure one.
    
    t0 = time.time()
    _simulate_aco_fast(d, candidates[0], seed=42)
    sim_time = time.time() - t0
    
    # How many simulations can we run?
    # Remaining time = 280 - (time.time() - start_time)
    # Let's aim for 10 simulations total.
    max_sims = max(1, int(10.0 / max(sim_time, 0.1)))
    max_sims = min(max_sims, 20) # Cap at 20
    
    best_eta = candidates[0]
    best_score = float('inf')
    
    # Use a few seeds to average out noise?
    # If max_sims is 10 and we have 4 candidates, we can do 2-3 seeds each.
    seeds_per_candidate = max(1, max_sims // len(candidates))
    
    for c_idx, eta_c in enumerate(candidates):
        if time.time() - start_time > 280.0:
            break
            
        scores = []
        for s in range(seeds_per_candidate):
            seed_val = 1000 + s * 12345 + c_idx * 777
            score = _simulate_aco_fast(d, eta_c, seed=seed_val)
            scores.append(score)
            
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_eta = eta_c.copy()
            
    return best_eta
