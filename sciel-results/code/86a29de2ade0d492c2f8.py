import numpy as np
import time
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

def _run_aco_fast(D, eta, n_ants=20, n_iter=30, seed=42):
    """Run ACO with given heuristic eta, return best tour length."""
    n = D.shape[0]
    rng = np.random.RandomState(seed)
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for it in range(n_iter):
        all_paths = []
        all_lens = []
        
        for ant in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path = [0]
            tour_len = 0.0
            
            for step in range(1, n):
                available = np.where(~visited)[0]
                probs = tau[current, available] * eta[current, available]
                total = probs.sum()
                
                if total <= 0:
                    next_node = available[0]
                else:
                    probs = probs / total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    next_node = available[np.searchsorted(cumsum, r)]
                
                tour_len += D[current, next_node]
                current = next_node
                visited[current] = True
                path.append(current)
            
            tour_len += D[current, 0]
            path.append(0)
            
            if tour_len < best_len:
                best_len = tour_len
            all_paths.append(path)
            all_lens.append(tour_len)
        
        tau *= 0.9
        for ant_idx in range(n_ants):
            l = all_lens[ant_idx]
            if l <= 0:
                continue
            reward = 1.0 / l
            path = all_paths[ant_idx]
            for k in range(len(path) - 1):
                i = path[k]
                j = path[k+1]
                tau[i, j] += reward
    
    return best_len

def _two_opt(tour, D):
    """Perform 2-opt optimization on a tour."""
    n = len(tour) - 1 # excluding the return to start
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[j+1] if j+1 < len(tour) else tour[0]
                
                # Check for non-crossing
                if a == c or a == d or b == c or b == d:
                    continue
                
                # Calculate length change
                old_dist = D[a, b] + D[c, d]
                new_dist = D[a, c] + D[b, d]
                
                if new_dist < old_dist - 1e-10:
                    # Reverse segment between i and j
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
    return tour

def _compute_structural_heuristic(D):
    """
    Compute a heuristic matrix based on structural importance.
    1. Find MST.
    2. Find a good tour using MST + Greedy/2-opt.
    3. Boost edges in the good tour.
    4. Combine with inverse distance.
    """
    n = D.shape[0]
    
    # 1. Base geometric heuristic: Inverse distance
    # Use a slightly higher power to favor local moves, but not too much
    D_safe = np.where(D > 0, D, 1.0)
    eta_geo = 1.0 / np.power(D_safe, 1.5)
    eta_geo[D == 0] = 1e-9
    
    # 2. Find a high-quality reference tour to identify "good" edges
    # Start with nearest neighbor
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    ref_tour = [0]
    
    for _ in range(n - 1):
        dists = D[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        ref_tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    ref_tour.append(0)
    
    # Improve with 2-opt
    ref_tour = _two_opt(ref_tour.copy(), D)
    
    # 3. Identify edges in the reference tour
    eta_struct = np.zeros((n, n))
    for i in range(len(ref_tour) - 1):
        u, v = ref_tour[i], ref_tour[i+1]
        eta_struct[u, v] = 1.0
        eta_struct[v, u] = 1.0 # ACO is usually symmetric in heuristic, but let's keep it directed if needed? 
                               # Standard ACO uses symmetric heuristic. The problem says "edge-attractiveness matrix".
                               # "each ant adds 1/length to pheromone[i, j] ... in travel direction only".
                               # The heuristic is usually symmetric for TSP.
    
    # 4. Combine
    # We want to strongly boost the structural edges.
    # Let's try: eta = eta_geo + boost * eta_struct
    # If an edge is in the good tour, it gets a significant boost.
    # However, if we boost too much, we might force the ACO to follow the reference tour exactly,
    # which might be suboptimal if the reference tour is not perfect.
    # A softer boost: multiply by a factor.
    
    # Let's use a multiplicative boost for edges in the tour, and additive for others?
    # Or just: eta = eta_geo * (1 + alpha * eta_struct)
    # This preserves the geometric preference but amplifies good edges.
    
    alpha = 5.0 # Boost factor
    eta_final = eta_geo * (1.0 + alpha * eta_struct)
    
    # Ensure no zeros
    eta_final[eta_final < 1e-9] = 1e-9
    
    return eta_final

def _make_eta_simple(D, p):
    n = D.shape[0]
    eta = np.ones((n, n))
    mask = D > 0
    eta[mask] = 1.0 / np.power(D[mask], p)
    eta[~mask] = 1e-9
    return eta

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    
    if time.time() > _DEADLINE:
        return _make_eta_simple(D, 1.0)
    
    # Try the structural heuristic
    try:
        eta_struct = _compute_structural_heuristic(D)
        # Validate
        if not np.all(np.isfinite(eta_struct)):
            raise ValueError("Non-finite values")
        if eta_struct.shape != D.shape:
            raise ValueError("Wrong shape")
        if np.any(eta_struct <= 0):
            # Clamp to 1e-9 as per rules
            eta_struct = np.where(eta_struct > 0, eta_struct, 1e-9)
            
        # Evaluate this heuristic
        score_struct = _run_aco_fast(D, eta_struct, seed=42)
        
        # Also evaluate a standard geometric heuristic for comparison
        # Maybe the structural one is overfitting or too restrictive?
        eta_geo = _make_eta_simple(D, 1.5)
        score_geo = _run_aco_fast(D, eta_geo, seed=42)
        
        if score_struct < score_geo:
            return eta_struct
        else:
            return eta_geo
            
    except Exception as e:
        # Fallback to simple geometric
        pass
    
    return _make_eta_simple(D, 1.5)
