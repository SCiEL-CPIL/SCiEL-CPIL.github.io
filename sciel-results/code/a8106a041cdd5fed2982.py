import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0
MARGIN = 10.0

def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()

def _greedy_value(order, weights, prizes, C):
    """
    Simulate greedy packing given an order of items.
    Returns the total prize of the feasible set.
    """
    n_constraints = weights.shape[0]
    rem = C.copy()
    val = 0.0
    for i in order:
        if np.all(weights[:, i] <= rem):
            rem -= weights[:, i]
            val += prizes[i]
    return val

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Ensure shapes
    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes.ravel()
    
    if weights.ndim == 1:
        w = weights.reshape(1, -1)
    elif weights.ndim == 2:
        w = weights
    else:
        w = weights.reshape(5, -1)

    n = p.shape[0]
    m = w.shape[0]
    
    if n == 0:
        return np.ones(0)

    cap = 25.0
    C = np.full(m, cap, dtype=float)

    # 1. Compute Shadow Prices (Lagrangian Dual)
    lam = np.zeros(m)
    best_dual_val = -1.0
    best_lam = lam.copy()
    
    n_sub = 50
    step_init = 10.0
    
    for it in range(n_sub):
        if _deadline_left() < MARGIN + 5.0:
            break
        shadow_cost = w.T @ lam
        rc = p - shadow_cost
        
        order = np.argsort(-rc, kind='stable')
        val = _greedy_value(order, w, p, C)
        
        if val > best_dual_val:
            best_dual_val = val
            best_lam = lam.copy()
        
        rem = C.copy()
        used = np.zeros(m)
        for i in order:
            if rc[i] <= 0:
                continue
            if np.all(w[:, i] <= rem):
                rem -= w[:, i]
                used += w[:, i]
        
        slack = C - used
        step = step_init / (1.0 + it * 0.5)
        lam = np.maximum(0.0, lam + step * slack)

    lam = best_lam
    shadow_cost = w.T @ lam
    rc = p - shadow_cost
    
    # 2. Prepare Base Signals
    rc_pos = np.where(rc > 0, rc, 0.0)
    rc_max = rc_pos.max()
    if rc_max > 0:
        rc_norm = rc_pos / rc_max
    else:
        rc_norm = np.ones(n)
        
    sum_w = w.sum(axis=0)
    safe_sum_w = np.maximum(sum_w, 1e-9)
    density = p / safe_sum_w
    d_max = density.max()
    if d_max > 0:
        d_norm = density / d_max
    else:
        d_norm = np.ones(n)

    max_w = w.max(axis=0)
    safe_max_w = np.maximum(max_w, 1e-9)
    bot_density = p / safe_max_w
    b_max = bot_density.max()
    if b_max > 0:
        b_norm = bot_density / b_max
    else:
        b_norm = np.ones(n)
        
    safe_weights = np.maximum(w, 1e-9)
    slack_ratios = (cap - w) / safe_weights
    slack_ratios = np.maximum(slack_ratios, 0.0)
    bottleneck_slack = slack_ratios.min(axis=0)
    bs_max = bottleneck_slack.max()
    if bs_max > 0:
        bs_norm = bottleneck_slack / bs_max
    else:
        bs_norm = np.ones(n)
        
    # Signal 5: Multi-Start Greedy Frequency
    weighted_counts = np.zeros(n, dtype=float)
    
    strategies = []
    strategies.append(np.argsort(-density, kind='stable'))
    strategies.append(np.argsort(-bot_density, kind='stable'))
    strategies.append(np.argsort(-p, kind='stable'))
    strategies.append(np.argsort(-rc, kind='stable'))
    
    combo_slack = (bottleneck_slack + 1e-9) * p
    strategies.append(np.argsort(-combo_slack, kind='stable'))
    
    rng = np.random.RandomState(42)
    strategies.append(rng.permutation(n))
    
    for order in strategies:
        if _deadline_left() < MARGIN + 2.0:
            break
        val = _greedy_value(order, w, p, C)
        if val > 0:
            rem = C.copy()
            for i in order:
                if np.all(w[:, i] <= rem):
                    rem -= w[:, i]
                    weighted_counts[i] += val
    
    wc_max = weighted_counts.max()
    if wc_max > 0:
        wc_norm = weighted_counts / wc_max
    else:
        wc_norm = np.ones(n)

    # 3. Fit/Select Best Heuristic Combination
    candidates = []
    candidates.append(rc_norm)
    candidates.append(d_norm)
    candidates.append(bs_norm)
    candidates.append(wc_norm)
    candidates.append(rc_norm * wc_norm)
    candidates.append(rc_norm * wc_norm * bs_norm)
    candidates.append((rc_norm + wc_norm) * bs_norm)
    candidates.append((rc_norm ** 1.5) * (wc_norm ** 1.0) * (bs_norm ** 0.5))
    candidates.append((wc_norm ** 1.5) * (bs_norm ** 1.0))
    candidates.append(rc_norm * d_norm * bs_norm)
    candidates.append(np.sqrt(rc_norm * wc_norm + 1e-9))
    
    # NEW: Structural Change - Simulate ACO-like behavior to evaluate heuristics
    # The previous method evaluated heuristics via simple greedy sort.
    # This method evaluates them via a mini-ACO simulation (or random sampling guided by heuristic)
    # to better predict how the actual ACO solver will perform.
    
    # We will evaluate each candidate heuristic by running a short, stochastic search
    # that mimics the ACO dynamics (probability proportional to heuristic) 
    # but without full pheromone update complexity, or just simple random sampling 
    # weighted by heuristic to see if it finds better packs than greedy.
    
    # However, to keep it fast and structural, let's use a "Soft-Greedy" or "Simulated Annealing" style 
    # acceptance or just multiple random restarts guided by the heuristic.
    
    # Let's try: For each heuristic, run K random restarts where we pick items 
    # with probability proportional to heuristic, but only if they fit.
    # This tests if the heuristic leads to good packs in a stochastic setting, 
    # which is closer to ACO than pure greedy.
    
    best_score = -1.0
    best_heur = candidates[0]
    
    # Use a fixed seed for these internal simulations to ensure reproducibility
    sim_rng = np.random.RandomState(123)
    
    num_sim_restarts = 20
    sim_items_to_try = n * 2 # Try to pick up to 2*n items (will stop when none fit)
    
    for idx, h in enumerate(candidates):
        if _deadline_left() < MARGIN + 2.0:
            break
            
        # Evaluate this heuristic via stochastic sampling
        best_sim_val = 0.0
        
        # Normalize heuristic for probability calculation
        h_pos = np.maximum(h, 1e-9)
        h_sum = h_pos.sum()
        if h_sum == 0:
            h_probs = np.ones(n) / n
        else:
            h_probs = h_pos / h_sum
            
        for _ in range(num_sim_restarts):
            rem = C.copy()
            val = 0.0
            available = np.ones(n, dtype=bool)
            
            # Simple stochastic greedy: at each step, pick an item from available 
            # ones with prob proportional to heuristic, if it fits.
            # To make it efficient, we don't do full ACO, just a weighted random choice 
            # from fitting items.
            
            # Optimization: Pre-compute which items fit? No, rem changes.
            # For n=100, checking all available items each step is O(n^2 * m) per restart.
            # 20 restarts * 100 steps * 100 items * 5 constraints = 10^6 ops. Fast.
            
            steps = 0
            while steps < sim_items_to_try:
                # Find items that fit
                fits_mask = np.all(w.T <= rem, axis=1) & available
                if not np.any(fits_mask):
                    break
                
                # Get indices of fitting items
                fitting_indices = np.where(fits_mask)[0]
                
                # Get probabilities for these items
                probs = h_probs[fitting_indices]
                probs_sum = probs.sum()
                if probs_sum == 0:
                    break
                probs_norm = probs / probs_sum
                
                # Pick one
                chosen_idx = sim_rng.choice(fitting_indices, p=probs_norm)
                
                # Add it
                rem -= w[:, chosen_idx]
                val += p[chosen_idx]
                available[chosen_idx] = False
                steps += 1
                
            if val > best_sim_val:
                best_sim_val = val
                
        # Also compare with pure greedy for this heuristic
        greedy_order = np.argsort(-h, kind='stable')
        greedy_val = _greedy_value(greedy_order, w, p, C)
        
        # Take the max of stochastic and greedy? Or just stochastic?
        # Stochastic is a better proxy for ACO. But greedy is a lower bound.
        # Let's use the max to be safe, or just the stochastic mean?
        # Let's use the max of the best stochastic run and the greedy run.
        eval_score = max(best_sim_val, greedy_val)
        
        if eval_score > best_score:
            best_score = eval_score
            best_heur = h
            
    # Ensure output is valid
    final_heur = np.asarray(best_heur, dtype=float)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1.0)
    final_heur = np.maximum(final_heur, 1e-9)
    
    f_max = final_heur.max()
    if f_max > 0:
        final_heur = final_heur / f_max
    
    final_heur = np.maximum(final_heur, 1e-9)
    
    return final_heur
