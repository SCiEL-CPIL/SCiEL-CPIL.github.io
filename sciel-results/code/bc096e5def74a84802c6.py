import numpy as np
import time
import sys

# Global deadline management
_T0 = time.time()
_DEADLINE = _T0 + 300.0
_TIME_MARGIN = 15.0

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for ACO-based CVRP.
    Combines inverse-distance and Clarke-Wright savings, tuned via mini-ACO simulation.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic safety checks
    if dm.shape != (n, n):
        # Fallback to simple inverse distance if shape is weird
        d_safe = np.where(dm > 0, dm, 1e-9)
        H = 1.0 / d_safe
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
        np.fill_diagonal(H, 1e-9)
        return H
        
    if not np.isfinite(dm).all() or not np.isfinite(coords).all() or not np.isfinite(D).all():
        d_safe = np.where(dm > 0, dm, 1e-9)
        H = 1.0 / d_safe
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
        np.fill_diagonal(H, 1e-9)
        return H

    # Pre-compute distance matrix from coordinates for consistency if dm is not Euclidean?
    # The task says distance_matrix is provided, but usually it's Euclidean. 
    # We will use the provided distance_matrix for the heuristic calculation to be safe,
    # but we might use coords for geometry checks if needed.
    
    # Helper: Build heuristic matrix given parameters
    def build_heuristic(alpha, savings_weight, depot_boost_weight=0.0):
        """
        alpha: exponent for inverse distance (1/d^alpha)
        savings_weight: multiplier for Clarke-Wright savings bonus
        depot_boost_weight: multiplier for depot proximity bonus
        """
        d_safe = np.where(dm > 0, dm, 1e-9)
        
        # Base term: 1 / d^alpha
        base = 1.0 / (d_safe ** alpha)
        
        # Savings term: (d0i + d0j - dij) / (d0i + d0j)
        d0 = dm[0, :]
        d0_rows = d0[:, np.newaxis]
        d0_cols = d0[np.newaxis, :]
        
        num = d0_rows + d0_cols - dm
        den = d0_rows + d0_cols
        den_safe = np.where(den > 0, den, 1e-9)
        
        savings_ratio = num / den_safe
        savings_ratio = np.clip(savings_ratio, 0.0, 1.0)
        
        # Mask for customer-customer edges (i>0, j>0)
        mask_customer = np.ones((n, n), dtype=bool)
        mask_customer[0, :] = False
        mask_customer[:, 0] = False
        
        H = base.copy()
        
        # Apply savings bonus to customer-customer edges
        if savings_weight > 0:
            bonus_factor = 1.0 + savings_weight * savings_ratio
            H[mask_customer] = base[mask_customer] * bonus_factor[mask_customer]
        
        # Apply depot boost if needed (encourages visiting nodes close to depot early/late?)
        # Actually, depot edges are handled by base. A boost might make ants prefer
        # returning to depot or starting with close nodes. Let's stick to savings for now.
        
        # Ensure positivity and finiteness
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
        np.fill_diagonal(H, 1e-9)
        
        return H

    # Helper: Run a quick ACO simulation to evaluate heuristic quality
    def simulate_aco(H, n_ants=10, n_iter=10, max_steps=100):
        """
        Runs a simplified ACO to estimate the solution quality.
        Returns the best cost found (lower is better).
        """
        P = np.ones((n, n))
        logH = np.log(H)
        best_cost_iter = np.inf
        best_overall = np.inf
        
        for _ in range(n_iter):
            it_costs = []
            logP = np.log(P)
            
            for _a in range(n_ants):
                node = 0
                rem = cap
                cost = 0.0
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                steps = 0
                
                while True:
                    # Find feasible neighbors
                    feas = []
                    for j in range(1, n):
                        if not visited[j] and D[j] <= rem + 1e-9:
                            feas.append(j)
                    
                    if not feas:
                        cost += dm[node, 0]
                        break
                    
                    # Calculate probabilities: P[i,j] * H[i,j]
                    # Use log-sum-exp for stability
                    vals = np.array([logP[node, j] + logH[node, j] for j in feas])
                    m = vals.max()
                    w = np.exp(vals - m)
                    s = w.sum()
                    
                    if s <= 0 or not np.isfinite(s):
                        nxt = feas[0]
                    else:
                        r = np.random.random() * s
                        acc = 0.0
                        nxt = feas[-1]
                        for k, j in enumerate(feas):
                            acc += w[k]
                            if r <= acc:
                                nxt = j
                                break
                    
                    cost += dm[node, nxt]
                    rem -= D[nxt]
                    visited[nxt] = True
                    node = nxt
                    steps += 1
                    
                    if steps >= max_steps:
                        cost += dm[node, 0]
                        break
                        
            it_costs.append(cost)
            
            best_in_iter = min(it_costs)
            if best_in_iter < best_overall:
                best_overall = best_in_iter
                
            # Update pheromone
            P *= 0.9
            P += 1.0 / best_in_iter
            
            # Time check inside simulation loop to avoid hanging
            if time.time() > _DEADLINE - _TIME_MARGIN - 5:
                break
                
        return best_overall

    # Time management
    remaining_time = _DEADLINE - _TIME_MARGIN - time.time()
    
    if remaining_time < 5.0:
        # Not enough time to tune, use default good parameters
        # alpha=1.0 (inverse dist), savings_weight=1.0
        return build_heuristic(1.0, 1.0)

    # Grid search over parameters
    # alpha: exponent for 1/d^alpha. 1.0 is standard. 2.0 is more aggressive.
    # savings_weight: how much to boost savings edges. 0.0 to 2.0.
    
    best_score = np.inf
    best_params = (1.0, 1.0)
    
    # Coarse grid
    alpha_vals = [0.8, 1.0, 1.2, 1.5]
    savings_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    n_ants_sim = 8
    n_iter_sim = 8
    
    for a in alpha_vals:
        for s in savings_vals:
            if time.time() > _DEADLINE - _TIME_MARGIN - 10:
                break
            H = build_heuristic(a, s)
            score = simulate_aco(H, n_ants_sim, n_iter_sim, 80)
            if score < best_score:
                best_score = score
                best_params = (a, s)
        if time.time() > _DEADLINE - _TIME_MARGIN - 10:
            break
            
    # Fine tuning around best coarse params
    if time.time() < _DEADLINE - _TIME_MARGIN - 15:
        a_opt, s_opt = best_params
        
        # Try smaller steps
        for da in [-0.1, 0.1]:
            for ds in [-0.2, 0.2]:
                na = a_opt + da
                ns = s_opt + ds
                if na < 0.1 or na > 3.0 or ns < 0.0 or ns > 4.0:
                    continue
                H = build_heuristic(na, ns)
                score = simulate_aco(H, n_ants_sim, n_iter_sim, 80)
                if score < best_score:
                    best_score = score
                    best_params = (na, ns)
                    
    # Final construction with best params
    a_final, s_final = best_params
    H_final = build_heuristic(a_final, s_final)
    
    return H_final
