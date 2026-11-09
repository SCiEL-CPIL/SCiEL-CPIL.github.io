import numpy as np
import time

def _fast_aco(prizes, dist, max_len, h_mat, seed, n_ant=20, n_iter=30, 
              feas=None, d0=None, n=0):
    """
    Fast ACO simulation to evaluate a heuristic matrix.
    Returns the best total prize collected by any ant.
    """
    np.random.seed(seed)
    n = int(n)
    if feas is None:
        feas = []
        for i in range(n):
            # Feasible next nodes j: dist[i,j] + dist[j,0] <= max_len
            row = dist[i, :] + d0
            mask = row <= max_len
            mask[i] = False
            feas.append(np.nonzero(mask)[0])
    
    ph = np.ones((n, n), dtype=np.float64)
    best_global = 0.0
    h_mat_c = np.ascontiguousarray(h_mat)
    d0_c = np.ascontiguousarray(d0)

    for _ in range(n_iter):
        best_iter = 0.0
        for _a in range(n_ant):
            # Start at node 0
            cur = 0
            length = 0.0
            visited = np.zeros(n, dtype=bool)
            
            # First move
            cand = feas[0]
            if cand.size == 0:
                break
            
            # Prune by budget: dist[0, j] + dist[j, 0] <= max_len
            # Note: feas[0] already ensures dist[0,j]+dist[j,0] <= max_len
            # But we also need to ensure we don't exceed max_len immediately.
            # The definition of feas[i] is j such that d[i,j] + d[j,0] <= max_len.
            # So for i=0, this is exactly the constraint for the first step.
            
            # Calculate weights
            w = ph[0, cand] * h_mat_c[0, cand]
            w = np.maximum(w, 0.0)
            tot_w = w.sum()
            if tot_w <= 0 or not np.isfinite(tot_w):
                break
            
            # Roulette wheel selection
            r = np.random.random() * tot_w
            cum = np.cumsum(w)
            pos = np.searchsorted(cum, r, side='right')
            if pos >= cand.size:
                pos = cand.size - 1
            nxt = cand[pos]
            
            length += dist[0, nxt]
            visited[nxt] = True
            cur = nxt

            # Subsequent moves
            while True:
                cand = feas[cur]
                if cand.size == 0:
                    break
                
                # Prune by remaining budget: length + d[cur, j] + d[j, 0] <= max_len
                m = (length + dist[cur, cand] + d0_c[cand]) <= max_len
                cand = cand[m]
                if cand.size == 0:
                    break
                
                w = ph[cur, cand] * h_mat_c[cur, cand]
                w = np.maximum(w, 0.0)
                tot_w = w.sum()
                if tot_w <= 0 or not np.isfinite(tot_w):
                    break
                
                r = np.random.random() * tot_w
                cum = np.cumsum(w)
                pos = np.searchsorted(cum, r, side='right')
                if pos >= cand.size:
                    pos = cand.size - 1
                nxt = cand[pos]
                
                length += dist[cur, nxt]
                visited[nxt] = True
                cur = nxt

            tot_prize = prizes[visited].sum()
            if tot_prize > best_iter:
                best_iter = tot_prize
        
        if best_iter > best_global:
            best_global = best_iter
        
        # Pheromone update
        ph = ph * 0.9 + best_iter / 10.0

    return best_global

def _make_h(prizes, dist, max_len, alpha, beta, gamma, delta, centrality):
    """
    Construct heuristic matrix with parameters.
    eta_ij = (prize_j^alpha * (1+centrality_j)^delta) / (d_ij + beta * d_j0)^gamma
    """
    d0 = dist[0, :]
    eps = 1e-9
    
    # Node-based terms
    prize_term = np.power(np.maximum(prizes, eps), alpha)
    centrality_term = np.power(1.0 + centrality, delta)
    node_score = prize_term * centrality_term
    
    # Edge-based denominator
    # dist is (n, n), d0 is (n,)
    # We want d_ij + beta * d_j0
    # d_j0 is the distance from j to 0, which is dist[j, 0] or d0[j]
    denom_base = dist + beta * d0[None, :]
    denom = np.power(np.maximum(denom_base, eps), gamma)
    
    eta = node_score[None, :] / denom
    
    # Handle self-loops and invalid entries
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    # Ensure diagonal is 0 or very small to avoid self-selection if logic allows
    np.fill_diagonal(eta, 1e-9)
    
    return eta

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    Uses instance-specific parameter tuning via fast ACO simulation.
    """
    t_start = time.time()
    deadline = t_start + 320.0 # Safety margin for 330s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    ml = float(max_len)
    n = dist.shape[0]
    d0 = dist[0, :]
    
    # Precompute feasibility lists
    feas = []
    for i in range(n):
        row = dist[i, :] + d0
        mask = row <= ml
        mask[i] = False
        feas.append(np.nonzero(mask)[0])
        
    # Precompute centrality
    # Centrality: sum of inverse distances to other non-depot nodes
    mask_c = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask_c, False)
    mask_c[0, :] = False
    mask_c[:, 0] = False # Exclude depot as target too? Usually centrality is about connectivity.
    
    d_safe = dist + 1e-9
    inv_d = 1.0 / d_safe
    inv_d_masked = inv_d * mask_c
    centrality = np.sum(inv_d_masked, axis=0)
    
    # Normalize centrality
    non_depot_c = centrality[1:]
    if len(non_depot_c) > 0 and np.mean(non_depot_c) > 0:
        centrality = centrality / np.mean(non_depot_c)
    else:
        centrality = np.ones(n)
        
    # Evaluation function
    seeds = [42, 7, 123, 999, 314]
    
    def eval_params(alpha, beta, gamma, delta):
        H = _make_h(prizes, dist, ml, alpha, beta, gamma, delta, centrality)
        score = 0.0
        for s in seeds:
            score += _fast_aco(prizes, dist, ml, H, s, feas=feas, d0=d0, n=n)
        return score / len(seeds), H

    best_score = -np.inf
    best_H = None
    best_params = (1.0, 1.0, 1.0, 1.0)
    
    # Stage 1: Coarse Grid Search
    # alpha: prize sensitivity
    # beta: depot proximity weight in denominator
    # gamma: distance sensitivity
    # delta: centrality sensitivity
    
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    gammas = [0.5, 1.0, 1.5, 2.0]
    deltas = [0.0, 0.5, 1.0] # 0.0 disables centrality
    
    for a in alphas:
        if time.time() > deadline: break
        for b in betas:
            if time.time() > deadline: break
            for g in gammas:
                if time.time() > deadline: break
                for d in deltas:
                    try:
                        sc, H = eval_params(a, b, g, d)
                        if sc > best_score:
                            best_score = sc
                            best_H = H
                            best_params = (a, b, g, d)
                    except Exception:
                        pass
                        
    if best_H is None:
        # Fallback to standard heuristic
        best_H = _make_h(prizes, dist, ml, 1.0, 1.0, 1.0, 1.0, centrality)
        best_params = (1.0, 1.0, 1.0, 1.0)

    # Stage 2: Local Refinement
    if time.time() < deadline:
        a0, b0, g0, d0p = best_params
        # Small steps around best
        step_a = 0.25
        step_b = 0.25
        step_g = 0.25
        step_d = 0.25
        
        for da in [-step_a, 0, step_a]:
            if time.time() > deadline: break
            for db in [-step_b, 0, step_b]:
                if time.time() > deadline: break
                for dg in [-step_g, 0, step_g]:
                    if time.time() > deadline: break
                    for dd in [-step_d, 0, step_d]:
                        a = a0 + da
                        b = b0 + db
                        g = g0 + dg
                        d = d0p + dd
                        
                        if a <= 0 or g <= 0 or b < 0 or d < 0:
                            continue
                            
                        try:
                            sc, H = eval_params(a, b, g, d)
                            if sc > best_score:
                                best_score = sc
                                best_H = H
                                best_params = (a, b, g, d)
                        except Exception:
                            pass

    # Final sanity check
    if not np.all(np.isfinite(best_H)) or best_H.shape != (n, n):
        # Fallback
        best_H = _make_h(prizes, dist, ml, 1.0, 1.0, 1.0, 1.0, centrality)
        
    # Ensure positive and finite
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    
    return best_H
