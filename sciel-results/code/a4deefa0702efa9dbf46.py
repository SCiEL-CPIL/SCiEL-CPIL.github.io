import numpy as np
import time

def _run_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    """
    Runs the ACO simulation to evaluate the heuristic.
    """
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    # Ensure positive values as per task description
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :] 
    # Precompute cost matrix for validity check: d[i,j] + d[j,0]
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
                # Valid nodes j: not visited, and length + d[cur, j] + d[j, 0] <= max_len
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
            
        # Pheromone update
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    # Hard deadline of 280s to be safe within 300s limit (evaluator might have overhead)
    deadline = start_time + 280.0
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    d_j0 = d[0, :]
    
    # Strategy: Regret-based heuristic
    # Attractiveness of j from i is high if p[j] is high relative to the "best alternative"
    # that is also reachable from i.
    # This creates a dynamic pressure: if a high prize node is easily reachable from i, 
    # it might not be as attractive if there's an even higher prize node also reachable.
    # However, the standard ACO uses static heuristics. 
    # We compute a static heuristic that approximates this regret logic.
    
    # For each node j, we want to know how "regrettable" it is to NOT visit it early?
    # Or rather, how attractive is it compared to alternatives?
    
    # Let's define the heuristic H[i, j] based on the idea that j is attractive 
    # if it has a high prize and is not "dominated" by another node k 
    # that is closer to i and has a higher prize.
    
    # A simpler static proxy for "regret":
    # For each j, let R[j] be the number of nodes k that are closer to i than j is? No, i varies.
    
    # Let's try a different structural change:
    # The current program uses p[j]^alpha / (d[i,j] + d[j,0])^beta.
    # This assumes a tradeoff between prize and distance.
    # The "Regret" strategy suggests we should look at the distribution of prizes.
    
    # Idea: Heuristic[i, j] = p[j] / (1 + count of nodes k with p[k] > p[j] and d[i,k] < d[i,j])
    # This penalizes moving to a node j if there are "better" (higher prize) nodes that are "closer" (less distance cost).
    # This encourages the ant to grab the "best available" node in its vicinity, reducing regret.
    
    # Precompute:
    # For each pair (i, j), we need to know how many k exist such that:
    # 1. p[k] > p[j]
    # 2. d[i, k] < d[i, j]
    # 3. k is not i and k is not j (and presumably valid, but static heuristic doesn't know visited)
    
    # This is O(N^3) to compute naively, which for N=50 is 125,000 ops, very fast.
    
    H = np.zeros((n, n))
    
    # Sort prizes to make comparisons faster? Or just loops.
    # Let's vectorize as much as possible.
    
    # d is (n, n). p is (n,).
    # For a fixed i, we have distances d[i, :].
    # We want for each j: count of k where p[k] > p[j] and d[i, k] < d[i, j].
    
    # We can do this for all i at once?
    # d[i, k] < d[i, j] means d[i, k] - d[i, j] < 0.
    # This is tricky to vectorize fully without large intermediate arrays (N^3 = 125k is fine, but let's be careful).
    
    # Let's use a loop over i, and vectorize over j and k.
    
    for i in range(n):
        d_i = d[i, :]
        # For each j, we want count of k such that p[k] > p[j] and d_i[k] < d_i[j]
        # We can create a matrix M_jk where M_jk = 1 if (p[k] > p[j]) and (d_i[k] < d_i[j])
        # p is shape (n,). p[j] is scalar in inner loop, but we can broadcast.
        
        # p_greater = p > p[None, :]  # Shape (n, n) -> p[k] > p[j] for all k, j
        # d_less = d_i[:, None] < d_i[None, :] # Shape (n, n) -> d_i[k] < d_i[j] for all k, j
        
        # Wait, d_i is 1D.
        # d_i[k] < d_i[j]
        # Let's construct the condition matrices.
        
        # p[k] > p[j]
        # k is column, j is row? Or k is row, j is column?
        # Let's say H[i, j] depends on count over k.
        # So for fixed i, we want a vector of counts for each j.
        
        # p_greater = (p > p[None, :]).T # Shape (n, n) where [k, j] is True if p[k] > p[j]
        # d_less = (d_i[:, None] < d_i[None, :]).T # Shape (n, n) where [k, j] is True if d_i[k] < d_i[j]
        
        # Actually, let's keep j as the index we want to update.
        # For each j, count k.
        
        # p[k] > p[j]
        # Let's use broadcasting.
        # p is (n,). 
        # p_j = p[None, :] # (1, n)
        # p_k = p[:, None] # (n, 1)
        # cond_p = p_k > p_j # (n, n) -> [k, j] is True if p[k] > p[j]
        
        # d_i is (n,).
        # d_j = d_i[None, :] # (1, n)
        # d_k = d_i[:, None] # (n, 1)
        # cond_d = d_k < d_j # (n, n) -> [k, j] is True if d_i[k] < d_i[j]
        
        # count = (cond_p & cond_d).sum(axis=0) # Sum over k (axis 0), result is vector of length n for j.
        
        cond_p = (p[:, None] > p[None, :]) # Shape (n, n), index [k, j]
        cond_d = (d_i[:, None] < d_i[None, :]) # Shape (n, n), index [k, j]
        
        counts = (cond_p & cond_d).sum(axis=0) # Shape (n,), index j
        
        # H[i, j] = p[j] / (1 + counts[j])
        # If counts[j] is high, it means many better nodes are closer, so j is less attractive (regret of not picking them is high? 
        # Or regret of picking j is high because you could have picked better ones closer? 
        # Yes, if you pick j, you "miss" the closer, better nodes. So attractiveness should be low.)
        
        # We also need to handle the case where j is unreachable? 
        # The heuristic is static, so it doesn't know about max_len directly in the "reachability" sense 
        # other than the cost d[i,j] + d[j,0]. 
        # But the ACO agent checks validity. The heuristic just needs to be a good guide.
        
        # Let's add a distance penalty to the denominator to ensure closer nodes are generally preferred 
        # if prizes are similar, or to scale the regret.
        # H[i, j] = p[j] / ((1 + counts[j]) * (d[i, j] + eps)^gamma)
        
        # Let's try gamma = 0.5 or 1.0.
        # Since the regret term already accounts for "closeness" relative to better nodes, 
        # maybe a mild distance penalty is enough.
        
        denom = (1.0 + counts) * (d_i + eps)**0.5
        H[i, :] = p / denom
        
        # Ensure node 0 has 0 attractiveness? 
        # Node 0 has prize 0, so H[i, 0] will be 0. 
        # The task says prize 0 at node 0. 
        # The ACO logic handles visited nodes, so H[i, 0] doesn't matter much if 0 is visited, 
        # but for i=0, H[0, 0] is self-loop.
        
    # Validate H
    if H.shape != (n, n) or not np.all(np.isfinite(H)):
        # Fallback
        H = p[None, :] / (d + d_j0[None, :] + eps)
        H[H <= 0] = 1e-9
        return H
    
    # The "Regret" heuristic might be too aggressive or weak.
    # Let's blend it with a standard heuristic to ensure stability.
    # H_std = p[None, :] / (d + d_j0[None, :] + eps)
    
    # Let's evaluate a few variants of the regret heuristic or blends.
    # Variant 1: Pure Regret
    # Variant 2: Regret * Standard
    # Variant 3: Regret + Standard
    
    candidates = []
    
    # Candidate 1: Pure Regret
    candidates.append(H.copy())
    
    # Candidate 2: Regret normalized by max prize?
    # H is already scaled by p[j].
    
    # Candidate 3: Blend with standard distance heuristic
    H_std = p[None, :] / (d + d_j0[None, :] + eps)
    H_std[H_std <= 0] = 1e-9
    
    # Normalize H and H_std to [0, 1] range for blending?
    H_max = H.max()
    H_std_max = H_std.max()
    if H_max > 0: H_norm = H / H_max
    else: H_norm = H
    if H_std_max > 0: H_std_norm = H_std / H_std_max
    else: H_std_norm = H_std
    
    for alpha in [0.5, 0.8, 1.0, 1.2]:
        blend = alpha * H_norm + (1 - alpha) * H_std_norm
        candidates.append(blend)
        
    # Candidate 4: Inverse Regret? (Attractiveness proportional to regret?)
    # No, the logic was: high regret (many better closer nodes) -> low attractiveness.
    # So H = p / (1+counts) is correct.
    
    # Let's also try a variant where we consider the "return cost" in the regret.
    # Regret based on total cost d[i,k] + d[k,0] vs d[i,j] + d[j,0].
    # If a better node k has a LOWER total cost, it's more regrettable to skip it.
    
    H_regret_return = np.zeros((n, n))
    total_cost = d + d_j0[None, :]
    
    for i in range(n):
        tc_i = total_cost[i, :]
        cond_p = (p[:, None] > p[None, :])
        cond_d = (tc_i[:, None] < tc_i[None, :])
        counts = (cond_p & cond_d).sum(axis=0)
        denom = (1.0 + counts) * (tc_i + eps)**0.5
        H_regret_return[i, :] = p / denom
        
    candidates.append(H_regret_return.copy())
    
    # Evaluate candidates
    best_score = -1
    best_heuristic = None
    
    # Use a quick ACO run to select the best candidate
    # We have ~280s. 5 candidates. 50s each is fine.
    # Actually, let's use 20 ants, 30 iters for the final selection to be robust.
    # 5 * 30 iters * 20 ants * 50 nodes... 
    # 30*20 = 600 ant runs.
    # Each ant run is ~50 steps. 30,000 steps.
    # Python loop overhead might be significant.
    # Let's time one run first.
    
    t0 = time.time()
    test_score = _run_aco(p, d, max_len, candidates[0], num_ants=5, num_iters=5, seed=0)
    t1 = time.time()
    run_time = t1 - t0
    
    # Estimate time for full run
    # 5 ants, 5 iters took run_time.
    # 20 ants, 30 iters is 4x ants, 6x iters = 24x time.
    est_full_time = run_time * 24
    
    # If est_full_time > 50s, reduce iterations/ants for selection
    if est_full_time > 50:
        sel_ants = 10
        sel_iters = 15
    else:
        sel_ants = 20
        sel_iters = 30
        
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Ensure valid
        if cand.shape != (n, n) or not np.all(np.isfinite(cand)):
            continue
            
        cand_clean = cand.copy()
        cand_clean[cand_clean <= 0] = 1e-9
        
        score = _run_aco(p, d, max_len, cand_clean, num_ants=sel_ants, num_iters=sel_iters, seed=0)
        
        if score > best_score:
            best_score = score
            best_heuristic = cand_clean
            
    if best_heuristic is None:
        # Fallback
        best_heuristic = H_std.copy()
        best_heuristic[best_heuristic <= 0] = 1e-9
        
    return best_heuristic
