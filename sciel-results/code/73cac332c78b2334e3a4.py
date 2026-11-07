import numpy as np
import time

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
                j = path[k + 1]
                tau[i, j] += reward

    return best_len

def _two_opt(tour, dist_matrix, deadline):
    """Improve a tour using 2-opt heuristic. Returns the improved tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            if time.time() > deadline:
                return tour
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]

                if j == i + 1:
                    continue

                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]

                if new_cost < current_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour

def _nn_tour(D, deadline):
    """Generate a nearest-neighbor tour starting from node 0."""
    n = D.shape[0]
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0

    for _ in range(n - 1):
        if time.time() > deadline:
            break
        min_dist = float('inf')
        next_node = -1
        # Vectorized nearest neighbor
        dists = D[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node

    return tour if len(tour) == n else None

def _random_2opt_tour(D, deadline, rng):
    """
    Generate a tour by starting with a random permutation (or NN) 
    and applying 2-opt. This helps find diverse local optima.
    """
    n = D.shape[0]
    # Start with random permutation
    tour = list(rng.permutation(n))
    
    # Apply 2-opt
    tour = _two_opt(tour, D, deadline)
    return tour

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]

    if time.time() > _DEADLINE - 5.0:
        # Fallback: simple inverse distance
        eta = 1.0 / (D + 1e-9)
        eta[D == 0] = 1e-9
        return eta

    search_deadline = min(time.time() + 90.0, _DEADLINE - 5.0)
    
    # Strategy: Build a "Topological Consensus" Heuristic
    # 1. Find multiple good tours using randomized 2-opt.
    # 2. Count edge frequencies in these tours.
    # 3. The heuristic is proportional to frequency + small base.
    # This steers ACO towards edges that are part of many good solutions,
    # effectively encoding the "shape" of the optimal tour rather than just local geometry.
    
    edge_counts = np.zeros((n, n))
    num_tours_to_find = 50 # Target number of diverse tours
    
    # First, get a very good reference tour (NN + 2-opt)
    ref_tour = _nn_tour(D, search_deadline - 10)
    if ref_tour:
        ref_tour = _two_opt(ref_tour, D, search_deadline - 10)
        # Add reference tour edges with high weight
        for i in range(n):
            u = ref_tour[i]
            v = ref_tour[(i + 1) % n]
            edge_counts[u, v] += 10.0
            edge_counts[v, u] += 10.0

    # Generate additional diverse tours using random start + 2-opt
    rng = np.random.RandomState(123)
    tours_found = 1
    
    while tours_found < num_tours_to_find:
        if time.time() > search_deadline - 10.0:
            break
        
        # Random restart
        tour = _random_2opt_tour(D, search_deadline - 15.0, rng)
        
        if tour:
            # Check if this tour is distinct from previous ones (simple check: length)
            # We just add it to the consensus, assuming diversity comes from random start
            for i in range(n):
                u = tour[i]
                v = tour[(i + 1) % n]
                edge_counts[u, v] += 1.0
                edge_counts[v, u] += 1.0
            tours_found += 1

    # Construct Heuristic Matrix
    # Base heuristic: Inverse distance to ensure some geometric awareness
    # Topological boost: Add the edge counts
    # eta[i,j] = base[i,j] + boost * edge_counts[i,j]
    
    # Normalize edge counts to be comparable to base heuristic scale
    # Base heuristic max is 1/d_min. 
    # Let's use a weighted sum.
    
    eta_base = 1.0 / (D + 1e-9)
    eta_base[D == 0] = 1e-9
    
    # Scale edge counts. Max count is roughly 10 + 50 = 60.
    # We want the topological signal to be strong but not overwhelming.
    # Let's map counts to [0, 1] and add to base.
    max_count = np.max(edge_counts)
    if max_count > 0:
        eta_topo = edge_counts / max_count
    else:
        eta_topo = np.zeros_like(edge_counts)
        
    # Combine: 0.5 * Geometric + 0.5 * Topological
    # This ensures we don't ignore long edges if they are part of the consensus,
    # but also don't pick short edges that lead to bad global structure.
    eta = 0.5 * eta_base + 0.5 * eta_topo
    
    # Ensure diagonal is low (don't pick self)
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure positivity
    eta[eta <= 0] = 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    
    # Optional: If we have time, verify this heuristic is better than a pure geometric one
    # by running a quick ACO test.
    if time.time() < _DEADLINE - 30.0:
        # Test candidate 1: Our consensus heuristic
        score_consensus = _run_aco_fast(D, eta, seed=42)
        
        # Test candidate 2: Pure geometric (1/d^2)
        eta_geo = 1.0 / np.power(D + 1e-9, 2)
        np.fill_diagonal(eta_geo, 1e-9)
        eta_geo[eta_geo <= 0] = 1e-9
        score_geo = _run_aco_fast(D, eta_geo, seed=42)
        
        if score_geo < score_consensus:
            eta = eta_geo
            # If geometric is better, maybe we should try to refine the mix?
            # But for now, stick to the better one.
            # Actually, let's try a mix of Consensus and Geometric with different weights
            # if we have time.
            if time.time() < _DEADLINE - 20.0:
                best_score = score_geo
                best_eta = eta_geo
                for w in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:
                    if time.time() > _DEADLINE - 15.0:
                        break
                    eta_mix = w * eta_base + (1-w) * eta_topo
                    np.fill_diagonal(eta_mix, 1e-9)
                    eta_mix[eta_mix <= 0] = 1e-9
                    s = _run_aco_fast(D, eta_mix, seed=42)
                    if s < best_score:
                        best_score = s
                        best_eta = eta_mix
                eta = best_eta

    return eta
