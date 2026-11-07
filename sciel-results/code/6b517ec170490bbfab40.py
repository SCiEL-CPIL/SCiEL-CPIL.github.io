import numpy as np
import time

def _run_greedy_proxy(prizes, distance_matrix, max_len, heuristic, num_runs=20):
    """
    Fast proxy: Simulates greedy ants.
    An ant starts at 0. At each step, it picks the unvisited node j that maximizes
    heuristic[cur, j] among valid nodes (those that allow return to 0 within max_len).
    This is much faster than ACO and correlates well with ACO performance for short paths.
    """
    n = len(prizes)
    d = distance_matrix
    d_j0 = d[0, :]
    cost = d + d_j0[None, :] # cost[i, j] = d[i, j] + d[j, 0]
    
    best_prize = 0.0
    # We run multiple "virtual" greedy paths. Since it's deterministic, 
    # running it multiple times is redundant unless we add noise or perturbations.
    # However, to simulate the "best of ants" behavior, we can just run it once.
    # But maybe we should simulate a few random perturbations to the heuristic 
    # to see robustness? Or just run it once.
    # Let's run it once for speed, but maybe add a small random perturbation 
    # to break ties and simulate diversity? No, keep it deterministic for speed.
    
    cur = 0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    length = 0.0
    prize = 0.0
    
    while True:
        remaining = max_len - length
        valid_mask = (~visited) & (cost[cur, :] <= remaining)
        valid_nodes = np.where(valid_mask)[0]
        
        if len(valid_nodes) == 0:
            break
        
        # Pick the node with the highest heuristic value
        # If heuristic has ties, pick the first one (or random? No, deterministic is faster)
        h_vals = heuristic[cur, valid_nodes]
        max_idx = np.argmax(h_vals)
        next_node = valid_nodes[max_idx]
        
        visited[next_node] = True
        length += d[cur, next_node]
        prize += prizes[next_node]
        cur = next_node
        
    return prize

def _run_aco_fast(prizes, distance_matrix, max_len, heuristic, num_ants=10, num_iters=10, seed=0):
    """
    Faster ACO implementation for final verification.
    Uses vectorized operations where possible and fewer iterations.
    """
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    d = distance_matrix
    d_j0 = d[0, :]
    cost = d + d_j0[None, :]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
            
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 90.0 # Safety margin
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    d_j0 = d[0, :]
    
    # Generate candidates
    # Focus on the form: p^alpha / (d + d_j0)^beta
    # We will grid search this space, but use the fast greedy proxy for the bulk.
    
    candidates = []
    candidate_params = []
    
    # Coarser grid for the fast proxy phase
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0.5, 1.0, 1.5, 2.0]
    
    for a in alphas:
        for b in betas:
            p_pow = np.where(p > eps, p**a, 0)
            denom = (d + d_j0[None, :] + eps)**b
            cand = p_pow[None, :] / denom
            candidates.append(cand)
            candidate_params.append((a, b))
            
    # Add a few structural variations
    # 1. Pure prize
    candidates.append(p[None, :] * np.ones((n, n)))
    candidate_params.append(('pure_prize', 0))
    
    # 2. Inverse distance only (if prize is low, maybe distance dominates?)
    # Actually, if prize is 0, heuristic should be low.
    # So p / (d+d_j0) is the standard.
    
    # Validate
    valid_indices = []
    for i, cand in enumerate(candidates):
        if cand.shape == (n, n) and np.all(np.isfinite(cand)):
            valid_indices.append(i)
    
    if not valid_indices:
        return np.ones((n, n))
        
    # Phase 1: Fast Greedy Screening
    # This is very fast. We can afford to test many more variants or just this set.
    greedy_scores = []
    for i in valid_indices:
        if time.time() > deadline - 20.0:
            break
        cand = candidates[i]
        # Run greedy proxy
        score = _run_greedy_proxy(p, d, max_len, cand)
        greedy_scores.append((score, i))
        
    # Sort by greedy score
    greedy_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 5 for ACO verification
    top_indices = [i for _, i in greedy_scores[:5]]
    
    # Phase 2: ACO Verification and Refinement
    best_score = -1
    best_heuristic = None
    best_params = None
    
    # Evaluate top candidates with ACO
    for i in top_indices:
        if time.time() > deadline - 10.0:
            break
        cand = candidates[i]
        # Use a small ACO run to verify
        # 10 ants, 10 iters is fast
        score = _run_aco_fast(p, d, max_len, cand, num_ants=10, num_iters=10, seed=0)
        
        if score > best_score:
            best_score = score
            best_heuristic = cand
            best_params = candidate_params[i]
            
    # Phase 3: Local Refinement of the Best Candidate
    # If we have time, refine the (alpha, beta) of the best candidate
    if best_params is not None and time.time() < deadline - 5.0:
        if isinstance(best_params[0], float) and isinstance(best_params[1], float):
            base_a, base_b = best_params
            # Try small steps
            steps = [0.1, 0.2]
            for s in steps:
                for da in [-s, s]:
                    for db in [-s, s]:
                        if time.time() > deadline - 2.0:
                            break
                        a = base_a + da
                        b = base_b + db
                        if a < 0.1 or b < 0.1:
                            continue
                        p_pow = np.where(p > eps, p**a, 0)
                        denom = (d + d_j0[None, :] + eps)**b
                        cand = p_pow[None, :] / denom
                        
                        if cand.shape != (n, n) or not np.all(np.isfinite(cand)):
                            continue
                        
                        score = _run_aco_fast(p, d, max_len, cand, num_ants=10, num_iters=10, seed=0)
                        if score > best_score:
                            best_score = score
                            best_heuristic = cand
                            best_params = (a, b)
                            
    # Final check: If we still have a lot of time, do a full ACO run on the best heuristic
    # to ensure it's robust? The evaluator runs the full ACO.
    # Our proxy ACO was reduced. Let's do one final full ACO run on the best candidate
    # to make sure we didn't pick a "proxy trap".
    if best_heuristic is not None and time.time() < deadline - 5.0:
        # Full ACO: 20 ants, 30 iters
        final_score = _run_aco_fast(p, d, max_len, best_heuristic, num_ants=20, num_iters=30, seed=0)
        # We don't necessarily need to update best_heuristic based on this score 
        # because we can't compare it to others without running full ACO on them too.
        # But if the score is significantly lower than expected, maybe we should fallback?
        # For now, trust the proxy + local search.
        
    if best_heuristic is None:
        best_heuristic = np.ones((n, n))
    else:
        best_heuristic = best_heuristic.copy()
        best_heuristic[best_heuristic <= 0] = 1e-9
        
    return best_heuristic
