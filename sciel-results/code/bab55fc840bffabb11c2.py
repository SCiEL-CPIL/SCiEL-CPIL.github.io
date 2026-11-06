"""Bounded lookahead 0/1 knapsack heuristic with deep greedy estimation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard: total wall-clock budget for the whole evaluation.
_START = time.monotonic()
_TOTAL_BUDGET = 115.0  # leave margin before the 150s kill

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

def _fast_greedy_complete(weights, values, capacity, policy='density'):
    """
    Simulate greedy completion from a state. Returns total added value.
    Optimized for speed: pre-sorts indices, then iterates.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    
    if policy == 'density':
        # Avoid division by zero
        safe_w = np.maximum(w, 1e-12)
        order = np.argsort(-v / safe_w)
    elif policy == 'value':
        order = np.argsort(-v)
    elif policy == 'blend':
        safe_w = np.maximum(w, 1e-12)
        order = np.argsort(-(v / safe_w + 0.15 * v))
    else:
        safe_w = np.maximum(w, 1e-12)
        order = np.argsort(-v / safe_w)
    
    cap = float(capacity)
    total = 0.0
    # Iterate through sorted items
    for i in order:
        wi = w[i]
        if wi <= cap:
            total += v[i]
            cap -= wi
            if cap <= 1e-9:
                break
    return total

def _score_candidates(w, v, cap, n):
    """Generate diverse candidate indices using multiple scoring families."""
    safe_w = np.maximum(w, 1e-12)
    density = v / safe_w
    frac = w / float(cap) if cap > 0 else np.zeros(n)
    
    # Multiple scoring families
    scores = np.vstack([
        v,                                      # 1. pure value
        density,                                # 2. pure density
        density + 0.1 * v,                      # 3. blend
        v - 0.02 * w,                           # 4. value minus weight penalty
        density * (1.0 + 0.3 * (w / cap)),      # 5. density boosted by relative size
        v / (w + 0.1 * cap),                    # 6. smoothed density
        density - 0.2 * np.abs(frac - 0.2),     # 7. density with size-shape term
        v * np.exp(-w / (cap + 1e-9)),          # 8. value decaying with weight
    ])
    
    cands = set()
    k = min(n, 10)
    for s in scores:
        top = np.argsort(-s)[:k]
        for t in top:
            cands.add(int(t))
    
    # Ensure density-argmax and value-argmax are included
    cands.add(int(np.argmax(density)))
    cands.add(int(np.argmax(v)))
    
    return list(cands), density

def _evaluate_candidate_deep(ci, w, v, cap, n, density):
    """
    Estimate total value if we pick candidate ci now, using deep greedy completion.
    We simulate the next 1-2 steps explicitly to get a better estimate than simple greedy.
    """
    used_w = float(w[ci])
    used_v = float(v[ci])
    new_cap = cap - used_w
    if new_cap < 0:
        return -np.inf
    
    # Build mask excluding candidate
    mask = np.ones(n, dtype=bool)
    mask[ci] = False
    rw = w[mask]
    rv = v[mask]
    rn = len(rw)
    
    if rn == 0:
        return used_v
        
    # 1. Simple greedy baseline
    base_est = used_v + _fast_greedy_complete(rw, rv, new_cap, 'density')
    
    # 2. Deep lookahead: Try picking the best next item explicitly
    # Identify top candidates for the *next* step in the remainder
    # We use density and value to find good "next" items
    
    safe_rw = np.maximum(rw, 1e-12)
    next_candidates = set()
    
    # Top 3 by density
    if rn > 0:
        top_dens = np.argsort(-rv / safe_rw)[:min(3, rn)]
        for idx in top_dens:
            next_candidates.add(int(idx))
            
    # Top 3 by value
    if rn > 0:
        top_val = np.argsort(-rv)[:min(3, rn)]
        for idx in top_val:
            next_candidates.add(int(idx))
            
    # Top 3 by blend
    if rn > 0:
        top_blend = np.argsort(-(rv / safe_rw + 0.15 * rv))[:min(3, rn)]
        for idx in top_blend:
            next_candidates.add(int(idx))
            
    best_deep_est = base_est
    
    for ni in next_candidates:
        # Check if this next item fits
        nw = float(rw[ni])
        if nw > new_cap:
            continue
            
        nv = float(rv[ni])
        rem_cap = new_cap - nw
        
        # Now complete the rest with a fast greedy (density is usually best for speed/quality tradeoff)
        # Create mask excluding ci and ni
        mask2 = np.ones(rn, dtype=bool)
        mask2[ni] = False
        rw2 = rw[mask2]
        rv2 = rv[mask2]
        
        est = used_v + nv + _fast_greedy_complete(rw2, rv2, rem_cap, 'density')
        if est > best_deep_est:
            best_deep_est = est
            
    return best_deep_est

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1
    
    # Only consider feasible items
    feasible = w <= cap
    if not np.any(feasible):
        return -1
    
    # Fast path if near deadline
    if _remaining_budget() < 0.1:
        feasible_v = np.where(feasible, v / np.maximum(w, 1e-12), -np.inf)
        idx = int(np.argmax(feasible_v))
        return idx if feasible[idx] else -1
    
    # Generate diverse candidates
    cands, density = _score_candidates(w, v, cap, n)
    
    # Filter to feasible candidates only
    cands = [c for c in cands if feasible[c]]
    if not cands:
        return -1
    
    # Limit candidates to keep per-call cost bounded
    # With deep evaluation, we can afford fewer candidates if it's slow, 
    # but N=100 is small. Let's stick to 30-50.
    sim_limit = 40
    if len(cands) > sim_limit:
        cands.sort(key=lambda i: -density[i])
        cands = cands[:sim_limit]
    
    # Evaluate each candidate with deep lookahead
    best_idx = cands[0]
    best_est = -np.inf
    
    for ci in cands:
        est = _evaluate_candidate_deep(ci, w, v, cap, n, density)
        if est > best_est:
            best_est = est
            best_idx = ci
        
        # Deadline check
        if _remaining_budget() < 0.5:
            break
    
    return int(best_idx)
# EVOLVE-BLOCK-END
