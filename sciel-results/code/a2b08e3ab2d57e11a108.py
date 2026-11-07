"""Optimized ACO heuristic search with faster simulation and broader candidate space."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0  # Safety margin (evaluator kills at 330s)
    
    N = len(prizes)
    
    # Precompute useful arrays
    d_j0 = distance_matrix[:, 0].copy()
    prizes_arr = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    
    # Precompute eligibility: for each (i, j), whether going i->j->0 fits in max_len
    # total_len[i,j] = d[i,j] + d[j,0] (for first move from 0)
    # But for general i, it's path_len + d[i,j] + d[j,0] <= max_len
    # We can precompute d[i,j] + d[j,0] for quick checking
    d_ij_plus_j0 = dist + d_j0[None, :]
    
    def run_aco_fast(eta, seed, n_ants=20, n_iters=30):
        """Fast ACO simulation using precomputed eligibility where possible."""
        rng = np.random.RandomState(seed)
        tau = np.ones((N, N))
        
        best_prize_global = 0.0
        
        for it in range(n_iters):
            best_prize_iter = 0.0
            
            for ant in range(n_ants):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    # Find eligible nodes vectorized
                    remaining = ~visited
                    remaining[0] = False  # never go back to 0 as a "next" node in this formulation
                    # Actually the problem says move to unvisited node j, and j can be any unvisited
                    # But node 0 is already visited, so it's excluded
                    # Eligibility: path_len + d[current, j] + d[j, 0] <= max_len
                    # i.e., d[current, j] + d[j, 0] <= max_len - path_len
                    threshold = max_len - path_len
                    eligible_mask = remaining & (d_ij_plus_j0[current, :] <= threshold + 1e-12)
                    
                    eligible_idx = np.where(eligible_mask)[0]
                    
                    if len(eligible_idx) == 0:
                        break
                    
                    # Compute probabilities
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    if total_w <= 0:
                        break
                    
                    probs = weights / total_w
                    r = rng.random_sample()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1
                    
                    next_node = eligible_idx[idx]
                    path_len += dist[current, next_node]
                    prize_total += prizes_arr[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global
    
    # Heuristic constructors
    def make_heuristic_alpha(alpha, p, r):
        """prize^r / (d_ij + alpha * d_j0)^p"""
        denom = dist + alpha * d_j0[None, :]
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) / np.power(denom, p)
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_sum(p, q, r, s):
        """prize^p / (d_ij^q + d_j0^r)^s"""
        denom = np.power(np.maximum(dist, 1e-12), q) + np.power(np.maximum(d_j0, 1e-12)[None, :], r)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], p) / np.power(denom, s)
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_mul(p, q, r):
        """prize^r / (d_ij^p * d_j0^q)"""
        denom = np.power(np.maximum(dist, 1e-12), p) * np.power(np.maximum(d_j0, 1e-12)[None, :], q)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_log(alpha, p, r):
        """prize^r / (log(1 + d_ij) + alpha * log(1 + d_j0))^p"""
        denom = np.log1p(np.maximum(dist, 0)) + alpha * np.log1p(np.maximum(d_j0, 0))[None, :]
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) / np.power(denom, p)
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_exp(alpha, p, r):
        """prize^r * exp(-alpha * (d_ij + d_j0)^p)"""
        val = dist + d_j0[None, :]
        val = np.maximum(val, 0)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) * np.exp(-alpha * np.power(val, p))
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_norm(alpha, p, r):
        """prize^r / (d_ij^p + alpha * d_j0^p)"""
        denom = np.power(np.maximum(dist, 1e-12), p) + alpha * np.power(np.maximum(d_j0, 1e-12)[None, :], p)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes_arr, 1e-12)[None, :], r) / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    # Generate a diverse set of candidates
    candidates = []
    
    # Type: alpha - prize^r / (d_ij + alpha * d_j0)^p
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        for p in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
            for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
                candidates.append(('alpha', alpha, p, r))
    
    # Type: sum - prize^p / (d_ij^q + d_j0^r)^s
    for p in [0.5, 1.0, 1.5, 2.0]:
        for q in [0.5, 1.0, 1.5, 2.0]:
            for r in [0.5, 1.0, 1.5, 2.0]:
                for s in [0.5, 1.0, 1.5]:
                    candidates.append(('sum', p, q, r, s))
    
    # Type: mul - prize^r / (d_ij^p * d_j0^q)
    for p in [0.5, 1.0, 1.5, 2.0]:
        for q in [0.5, 1.0, 1.5, 2.0]:
            for r in [0.5, 1.0, 1.5, 2.0]:
                candidates.append(('mul', p, q, r))
    
    # Type: log - prize^r / (log(1+d_ij) + alpha*log(1+d_j0))^p
    for alpha in [0.5, 1.0, 2.0]:
        for p in [0.5, 1.0, 1.5, 2.0]:
            for r in [0.5, 1.0, 1.5, 2.0]:
                candidates.append(('log', alpha, p, r))
    
    # Type: exp - prize^r * exp(-alpha*(d_ij+d_j0)^p)
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for p in [0.5, 1.0, 1.5, 2.0]:
            for r in [0.5, 1.0, 1.5, 2.0]:
                candidates.append(('exp', alpha, p, r))
    
    # Type: norm - prize^r / (d_ij^p + alpha*d_j0^p)
    for alpha in [0.5, 1.0, 2.0]:
        for p in [0.5, 1.0, 1.5, 2.0]:
            for r in [0.5, 1.0, 1.5, 2.0]:
                candidates.append(('norm', alpha, p, r))
    
    def get_eta(cand):
        ctype = cand[0]
        if ctype == 'alpha':
            _, alpha, p, r = cand
            return make_heuristic_alpha(alpha, p, r)
        elif ctype == 'sum':
            _, p, q, r, s = cand
            return make_heuristic_sum(p, q, r, s)
        elif ctype == 'mul':
            _, p, q, r = cand
            return make_heuristic_mul(p, q, r)
        elif ctype == 'log':
            _, alpha, p, r = cand
            return make_heuristic_log(alpha, p, r)
        elif ctype == 'exp':
            _, alpha, p, r = cand
            return make_heuristic_exp(alpha, p, r)
        elif ctype == 'norm':
            _, alpha, p, r = cand
            return make_heuristic_norm(alpha, p, r)
        else:
            return make_heuristic_alpha(1.0, 1.0, 1.0)
    
    # Phase 1: Quick screen with seed 0
    quick_results = []
    n_cands = len(candidates)
    
    for i, cand in enumerate(candidates):
        if time.time() - start_time > time_limit - 100:
            break
        eta = get_eta(cand)
        s = run_aco_fast(eta, 0)
        quick_results.append((s, cand, eta))
    
    # Sort by quick score descending
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Evaluate top candidates with multiple seeds
    top_k = min(15, len(quick_results))
    best_score = -1
    best_eta = quick_results[0][2] if quick_results else make_heuristic_alpha(1.0, 1.0, 1.0)
    best_cand = quick_results[0][1] if quick_results else ('alpha', 1.0, 1.0, 1.0)
    
    for s_quick, cand, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 30:
            break
        scores = [s_quick]
        for seed in [1, 2, 3, 4, 5]:
            if time.time() - start_time > time_limit - 25:
                break
            s = run_aco_fast(eta, seed)
            scores.append(s)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_eta = eta
            best_cand = cand
    
    # Phase 3: Local refinement around best candidate
    if time.time() - start_time < time_limit - 35:
        best_type = best_cand[0]
        refinement_candidates = []
        
        if best_type == 'alpha':
            _, balpha, bp, br = best_cand
            for dalpha in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                for dp in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                    for dr in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                        nalpha = balpha + dalpha
                        np_val = bp + dp
                        nr_val = br + dr
                        if nalpha > 0.05 and np_val > 0.05 and nr_val > 0.05:
                            refinement_candidates.append(('alpha', nalpha, np_val, nr_val))
        
        elif best_type == 'sum':
            _, bp, bq, br, bs = best_cand
            for dp in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                for dq in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                    for dr in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                        np_val = bp + dp
                        nq_val = bq + dq
                        nr_val = br + dr
                        if np_val > 0.05 and nq_val > 0.05 and nr_val > 0.05:
                            refinement_candidates.append(('sum', np_val, nq_val, nr_val, bs))
        
        elif best_type == 'mul':
            _, bp, bq, br = best_cand
            for dp in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                for dq in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                    for dr in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                        np_val = bp + dp
                        nq_val = bq + dq
                        nr_val = br + dr
                        if np_val > 0.05 and nq_val > 0.05 and nr_val > 0.05:
                            refinement_candidates.append(('mul', np_val, nq_val, nr_val))
        
        elif best_type == 'log':
            _, balpha, bp, br = best_cand
            for dalpha in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                for dp in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                    for dr in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                        nalpha = balpha + dalpha
                        np_val = bp + dp
                        nr_val = br + dr
                        if nalpha > 0.05 and np_val > 0.05 and nr_val > 0.05:
                            refinement_candidates.append(('log', nalpha, np_val, nr_val))
        
        elif best_type == 'exp':
            _, balpha, bp, br = best_cand
            for dalpha in [-0.3, -0.1, 0.1, 0.3, 0.5]:
                for dp in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                    for dr in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                        nalpha = balpha + dalpha
                        np_val = bp + dp
                        nr_val = br + dr
                        if nalpha > 0.05 and np_val > 0.05 and nr_val > 0.05:
                            refinement_candidates.append(('exp', nalpha, np_val, nr_val))
        
        elif best_type == 'norm':
            _, balpha, bp, br = best_cand
            for dalpha in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                for dp in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                    for dr in [-0.15, -0.05, 0.05, 0.15, 0.3]:
                        nalpha = balpha + dalpha
                        np_val = bp + dp
                        nr_val = br + dr
                        if nalpha > 0.05 and np_val > 0.05 and nr_val > 0.05:
                            refinement_candidates.append(('norm', nalpha, np_val, nr_val))
        
        # Evaluate refinement candidates
        for cand in refinement_candidates:
            if time.time() - start_time > time_limit - 20:
                break
            eta = get_eta(cand)
            s = run_aco_fast(eta, 0)
            if s > best_score * 0.95:  # Only pursue promising ones
                scores = [s]
                for seed in [1, 2, 3]:
                    if time.time() - start_time > time_limit - 15:
                        break
                    s2 = run_aco_fast(eta, seed)
                    scores.append(s2)
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_eta = eta
    
    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != dist.shape:
        denom = dist + d_j0[None, :]
        denom = np.maximum(denom, 1e-12)
        best_eta = prizes_arr[None, :] / denom
        best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    # Ensure no non-positive values
    best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    return best_eta

# EVOLVE-BLOCK-END
