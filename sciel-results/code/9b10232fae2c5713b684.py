"""Combined approach: opportunity-cost aware heuristic with fast ACO simulation and multi-phase search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 325.0  # Safety margin (evaluator kills at 330s)
    
    N = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    
    # Precompute distances from each node to node 0
    d_j0 = dist[:, 0].copy()
    
    # Precompute eligibility: for each (i, j), whether going i->j->0 fits in max_len
    # total_len = path_len + d[i,j] + d[j,0] <= max_len
    # For the first move from node 0: d[0,j] + d[j,0] <= max_len
    # For subsequent moves, it depends on path_len, so we can't fully precompute
    # But we can precompute d[i,j] + d[j,0] for quick checking
    total_cost = dist + d_j0[None, :]  # (N, N): d[i,j] + d[j,0]
    
    # Fast ACO simulation
    def run_aco_fast(eta, seed, n_ants=20, n_iters=30):
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
                    # Find eligible nodes: unvisited and path_len + d[current,j] + d[j,0] <= max_len
                    remaining = max_len - path_len
                    eligible_mask = (~visited) & (total_cost[current, :] <= remaining)
                    eligible = np.where(eligible_mask)[0]
                    
                    if len(eligible) == 0:
                        break
                    
                    # Compute probabilities
                    probs = tau[current, eligible] * eta[current, eligible]
                    total_prob = probs.sum()
                    if total_prob <= 0:
                        break
                    
                    probs = probs / total_prob
                    r = rng.random_sample()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible):
                        idx = len(eligible) - 1
                    
                    next_node = eligible[idx]
                    path_len += dist[current, next_node]
                    prize_total += prizes[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global

    # Heuristic families:
    # Family A: prize^p / (d_ij^q + d_j0^r)^s  (distance-based)
    # Family B: prize^r / (d_ij * d_j0^q)       (multiplicative distance)
    # Family C: prize^r / (d_ij + alpha * d_j0)^p (weighted sum)
    # Family D: prize^p * slack^w / (d_ij + d_j0)^s  (opportunity cost / slack-based)
    #   where slack[j] = max_len - d[0,j] - d[j,0]  (remaining budget after visiting j from start and returning)
    #   This penalizes nodes that consume high fraction of total budget
    
    # Precompute slack for family D
    # slack[j] = max_len - d[0,j] - d[j,0]  (how much budget remains after a round trip via j)
    slack = np.maximum(max_len - dist[0, :] - d_j0, 0.0)  # (N,)
    
    def make_heuristic_A(p, q, r, s):
        """prize^p / (d_ij^q + d_j0^r)^s"""
        denom = np.power(np.maximum(dist, 1e-12), q) + np.power(np.maximum(d_j0, 1e-12)[None, :], r)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes, 1e-12)[None, :], p) / np.power(denom, s)
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_B(p, q, r):
        """prize^r / (d_ij^p * d_j0^q)"""
        denom = np.power(np.maximum(dist, 1e-12), p) * np.power(np.maximum(d_j0, 1e-12)[None, :], q)
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes, 1e-12)[None, :], r) / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_C(alpha, p, r):
        """prize^r / (d_ij + alpha * d_j0)^p"""
        denom = dist + alpha * d_j0[None, :]
        denom = np.maximum(denom, 1e-12)
        eta = np.power(np.maximum(prizes, 1e-12)[None, :], r) / np.power(denom, p)
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_D(p, w, s, alpha=1.0):
        """prize^p * slack^w / (d_ij + alpha*d_j0)^s
        Opportunity cost: reward high prize AND high remaining slack, penalize high distance
        """
        slack_safe = np.maximum(slack, 1e-12)
        denom = dist + alpha * d_j0[None, :]
        denom = np.maximum(denom, 1e-12)
        eta = (np.power(np.maximum(prizes, 1e-12)[None, :], p) * 
               np.power(slack_safe[None, :], w) / 
               np.power(denom, s))
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    def make_heuristic_E(p, w, s, alpha, beta):
        """prize^p * (max_len - d_j0)^w / (d_ij + alpha*d_j0 + beta)^s
        Uses remaining budget after returning from j
        """
        rem_budget = np.maximum(max_len - d_j0, 1e-12)
        denom = dist + alpha * d_j0[None, :] + beta
        denom = np.maximum(denom, 1e-12)
        eta = (np.power(np.maximum(prizes, 1e-12)[None, :], p) * 
               np.power(rem_budget[None, :], w) / 
               np.power(denom, s))
        eta = np.where(eta > 0, eta, 1e-9)
        return eta
    
    # Generate candidates across all families
    candidates = []
    
    # Family A: prize^p / (d_ij^q + d_j0^r)^s
    for p in [1.0, 1.5, 2.0]:
        for q in [1.0, 1.5, 2.0]:
            for r in [1.0, 1.5, 2.0]:
                for s in [1.0, 1.5, 2.0]:
                    candidates.append(('A', p, q, r, s))
    
    # Family B: prize^r / (d_ij^p * d_j0^q)
    for p in [1.0, 1.5, 2.0]:
        for q in [1.0, 1.5, 2.0]:
            for r in [1.0, 1.5, 2.0]:
                candidates.append(('B', p, q, r))
    
    # Family C: prize^r / (d_ij + alpha * d_j0)^p
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for p in [1.0, 1.5, 2.0, 2.5]:
            for r in [1.0, 1.5, 2.0]:
                candidates.append(('C', alpha, p, r))
    
    # Family D: prize^p * slack^w / (d_ij + alpha*d_j0)^s
    for p in [1.0, 1.5, 2.0]:
        for w in [0.5, 1.0, 1.5, 2.0]:
            for s in [1.0, 1.5, 2.0]:
                candidates.append(('D', p, w, s, 1.0))
    
    # Family E: prize^p * (max_len - d_j0)^w / (d_ij + alpha*d_j0 + beta)^s
    for p in [1.0, 1.5, 2.0]:
        for w in [0.5, 1.0, 1.5]:
            for s in [1.0, 1.5, 2.0]:
                for alpha in [0.5, 1.0, 2.0]:
                    candidates.append(('E', p, w, s, alpha, 0.0))
    
    def get_eta(cand):
        ctype = cand[0]
        if ctype == 'A':
            _, p, q, r, s = cand
            return make_heuristic_A(p, q, r, s)
        elif ctype == 'B':
            _, p, q, r = cand
            return make_heuristic_B(p, q, r)
        elif ctype == 'C':
            _, alpha, p, r = cand
            return make_heuristic_C(alpha, p, r)
        elif ctype == 'D':
            _, p, w, s, alpha = cand
            return make_heuristic_D(p, w, s, alpha)
        elif ctype == 'E':
            _, p, w, s, alpha, beta = cand
            return make_heuristic_E(p, w, s, alpha, beta)
        else:
            return make_heuristic_C(1.0, 1.0, 1.0)
    
    # Phase 1: Quick screen with seed 0
    quick_results = []
    for cand in candidates:
        if time.time() - start_time > time_limit - 80:
            break
        eta = get_eta(cand)
        s = run_aco_fast(eta, 0)
        quick_results.append((s, cand, eta))
    
    # Sort by quick score descending
    quick_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Evaluate top candidates with multiple seeds
    top_k = min(12, len(quick_results))
    best_score = -1
    best_eta = quick_results[0][2] if quick_results else make_heuristic_C(1.0, 1.0, 1.0)
    best_cand = quick_results[0][1] if quick_results else ('C', 1.0, 1.0, 1.0)
    
    for s_quick, cand, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 25:
            break
        scores = [s_quick]  # seed 0 already computed
        for seed in [1, 2, 3, 4, 5]:
            if time.time() - start_time > time_limit - 20:
                break
            s = run_aco_fast(eta, seed)
            scores.append(s)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_eta = eta
            best_cand = cand
    
    # Phase 3: Local refinement around best candidate
    if time.time() - start_time < time_limit - 30:
        best_type = best_cand[0]
        refinement_candidates = []
        
        if best_type == 'A':
            _, bp, bq, br, bs = best_cand
            for dp in [-0.2, 0.2]:
                for dq in [-0.2, 0.2]:
                    for dr in [-0.2, 0.2]:
                        for ds in [-0.2, 0.2]:
                            np_val = bp + dp
                            nq_val = bq + dq
                            nr_val = br + dr
                            ns_val = bs + ds
                            if np_val > 0.1 and nq_val > 0.1 and nr_val > 0.1 and ns_val > 0.1:
                                refinement_candidates.append(('A', np_val, nq_val, nr_val, ns_val))
        
        elif best_type == 'B':
            _, bp, bq, br = best_cand
            for dp in [-0.2, 0.2]:
                for dq in [-0.2, 0.2]:
                    for dr in [-0.2, 0.2]:
                        np_val = bp + dp
                        nq_val = bq + dq
                        nr_val = br + dr
                        if np_val > 0.1 and nq_val > 0.1 and nr_val > 0.1:
                            refinement_candidates.append(('B', np_val, nq_val, nr_val))
        
        elif best_type == 'C':
            _, balpha, bp, br = best_cand
            for dalpha in [-0.3, 0.3]:
                for dp in [-0.2, 0.2]:
                    for dr in [-0.2, 0.2]:
                        nalpha = balpha + dalpha
                        np_val = bp + dp
                        nr_val = br + dr
                        if nalpha > 0.1 and np_val > 0.1 and nr_val > 0.1:
                            refinement_candidates.append(('C', nalpha, np_val, nr_val))
        
        elif best_type == 'D':
            _, bp, bw, bs, balpha = best_cand
            for dp in [-0.2, 0.2]:
                for dw in [-0.2, 0.2]:
                    for ds in [-0.2, 0.2]:
                        np_val = bp + dp
                        nw_val = bw + dw
                        ns_val = bs + ds
                        if np_val > 0.1 and nw_val > 0.1 and ns_val > 0.1:
                            refinement_candidates.append(('D', np_val, nw_val, ns_val, balpha))
        
        elif best_type == 'E':
            _, bp, bw, bs, balpha, bbeta = best_cand
            for dp in [-0.2, 0.2]:
                for dw in [-0.2, 0.2]:
                    for ds in [-0.2, 0.2]:
                        for dalpha in [-0.3, 0.3]:
                            np_val = bp + dp
                            nw_val = bw + dw
                            ns_val = bs + ds
                            nalpha = balpha + dalpha
                            if np_val > 0.1 and nw_val > 0.1 and ns_val > 0.1 and nalpha > 0.1:
                                refinement_candidates.append(('E', np_val, nw_val, ns_val, nalpha, bbeta))
        
        for cand in refinement_candidates:
            if time.time() - start_time > time_limit - 20:
                break
            eta = get_eta(cand)
            s = run_aco_fast(eta, 0)
            if s > best_score:
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
                    best_cand = cand
    
    # Phase 4: If time remains, try a few more diverse candidates
    if time.time() - start_time < time_limit - 25:
        # Try some hand-crafted diverse heuristics
        diverse = [
            ('C', 2.0, 2.5, 2.0),
            ('C', 3.0, 1.5, 1.5),
            ('D', 2.0, 1.5, 1.5, 1.0),
            ('D', 1.5, 2.0, 2.0, 1.0),
            ('E', 2.0, 1.0, 1.5, 1.0, 0.1),
            ('E', 1.5, 1.5, 2.0, 2.0, 0.0),
        ]
        for cand in diverse:
            if time.time() - start_time > time_limit - 20:
                break
            if cand in [r[1] for r in quick_results[:top_k]]:
                continue
            eta = get_eta(cand)
            s = run_aco_fast(eta, 0)
            if s > best_score:
                scores = [s]
                for seed in [1, 2]:
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
        best_eta = prizes[None, :] / denom
        best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    # Ensure no non-positive values
    best_eta = np.where(best_eta > 0, best_eta, 1e-9)
    
    return best_eta

# EVOLVE-BLOCK-END
