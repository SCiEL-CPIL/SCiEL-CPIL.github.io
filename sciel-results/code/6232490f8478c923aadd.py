import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy:
    1. Generate a set of candidate heuristic matrices based on parametric forms
       combining prize, distance to current node, and distance to depot.
    2. Evaluate each candidate using a fast greedy tour simulation.
       The assumption is that a heuristic that guides a greedy ant to a good tour
       is also a good heuristic for the full ACO algorithm.
    3. Return the best-performing heuristic matrix.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones((n, n), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin
    
    d = distance_matrix.copy()
    d_to_depot = d[:, 0].copy()
    eps = 1e-9
    
    # Precompute safe distances
    d_safe = d + eps
    d_to_depot_safe = d_to_depot + eps
    
    # Feasibility mask: can we go from i to j and back to 0 within max_len?
    # This is dynamic based on current length, but for a static heuristic,
    # we can at least filter out nodes that are individually too far from depot
    # or too far from any node in a way that makes them impossible.
    # However, the ACO loop handles the dynamic length constraint.
    # We can use a static "potential" feasibility: d[j, 0] <= max_len.
    
    # Candidate heuristic generation
    candidates = []
    
    # Form 1: Standard prize/distance
    # eta_ij = prize_j^a / (d_ij^b * d_j0^c)
    a_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    b_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    c_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Precompute prize powers
    prize_powers = {a: (prizes + eps) ** a for a in a_vals}
    
    # Precompute distance powers
    d_powers_b = {b: d_safe ** b for b in b_vals}
    d_powers_c = {c: d_to_depot_safe ** c for c in c_vals}
    
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                if time.time() > deadline:
                    break
                # eta_ij = prize_j^a / (d_ij^b * d_j0^c)
                # Note: d_j0^c is a vector of size n, broadcast to columns
                denom = d_powers_b[b] * (d_powers_c[c][None, :])
                eta = prize_powers[a][None, :] / denom
                eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
                candidates.append(eta)
    
    # Form 2: Detour-based
    # eta_ij = prize_j^a / max(d_ij + d_j0 - d_i0, eps)^b
    # This penalizes detours more accurately
    detour = d + d_to_depot_safe[None, :] - d_to_depot_safe[:, None]
    detour_safe = np.maximum(detour, eps)
    
    b_detour_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    a_detour_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for a in a_detour_vals:
        for b in b_detour_vals:
            if time.time() > deadline:
                break
            eta = prize_powers[a][None, :] / (detour_safe ** b)
            eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
            candidates.append(eta)
    
    # Form 3: Hybrid with centrality (inverse distance sum)
    # centrality_j = sum(1/d_jk)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    mask[0, :] = False
    inv_d = 1.0 / d_safe
    centrality = np.sum(inv_d * mask, axis=0)
    mean_c = np.mean(centrality[1:]) if n > 1 else 1.0
    centrality_norm = centrality / mean_c if mean_c > 0 else np.ones(n)
    centrality_norm = np.where(np.isfinite(centrality_norm) & (centrality_norm > 0), centrality_norm, 1.0)
    
    g_vals = [0.5, 1.0, 1.5]
    for a in [1.0, 2.0]:
        for b in [1.0, 1.5]:
            for g in g_vals:
                if time.time() > deadline:
                    break
                eta = (prize_powers[a][None, :] * (centrality_norm ** g)[None, :]) / (d_safe ** b)
                eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
                candidates.append(eta)
    
    if not candidates:
        # Fallback
        return (prizes[None, :] / (d_safe * d_to_depot_safe[None, :])).astype(np.float64)

    # Evaluation function: Fast Greedy Simulation
    def evaluate_heuristic(eta, n_tours=20):
        """
        Estimates the quality of the heuristic by running n_tours greedy tours.
        At each step, the ant moves to the unvisited feasible node j that maximizes eta[i, j].
        This approximates the behavior of ACO when pheromone is uniform (initial state).
        """
        best_total = 0
        for _ in range(n_tours):
            visited = [0]
            current = 0
            length = 0.0
            total_prize = prizes[0]
            
            while True:
                best_j = -1
                best_eta_val = -1
                best_new_length = 0
                
                # Find the best candidate
                for j in range(1, n):
                    if j in visited:
                        continue
                    new_length = length + d[current, j] + d[j, 0]
                    if new_length <= max_len:
                        # Greedy choice based on heuristic
                        val = eta[current, j]
                        if val > best_eta_val:
                            best_eta_val = val
                            best_j = j
                            best_new_length = new_length
                
                if best_j == -1:
                    break
                
                # Move
                length = best_new_length
                total_prize += prizes[best_j]
                visited.append(best_j)
                current = best_j
            
            if total_prize > best_total:
                best_total = total_prize
                
        return best_total

    # Evaluate candidates
    best_score = -1
    best_eta = candidates[0]
    
    # To save time, we can evaluate a subset first, or all if fast enough.
    # n=50, tours=20, steps~50. 50*20*50 = 50k ops per candidate.
    # ~50-100 candidates. Total ~5M ops. Very fast in NumPy/Python.
    
    for eta in candidates:
        if time.time() > deadline:
            break
        
        score = evaluate_heuristic(eta)
        if score > best_score:
            best_score = score
            best_eta = eta
    
    # Final check: ensure no non-finite values
    best_eta = np.where(np.isfinite(best_eta) & (best_eta > 0), best_eta, 1e-9)
    
    return best_eta
