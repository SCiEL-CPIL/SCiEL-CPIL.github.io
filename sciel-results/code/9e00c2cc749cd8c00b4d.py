import numpy as np
import time

_START = None
_CALL_COUNT = 0
_DEADLINE_MARGIN = 10.0  # seconds before hard limit to fall back

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _rollout_value(capacity, weights, values, exclude_idx):
    # simulate greedy density rollout starting with item exclude_idx already taken
    n = len(weights)
    if exclude_idx is None or exclude_idx < 0:
        total = 0.0
        cap = capacity
    else:
        if weights[exclude_idx] > capacity:
            return -1.0
        total = float(values[exclude_idx])
        cap = capacity - float(weights[exclude_idx])
    avail = np.ones(n, dtype=bool)
    if exclude_idx is not None and 0 <= exclude_idx < n:
        avail[exclude_idx] = False
    # precompute density
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    while cap > 0:
        feasible = avail & (weights <= cap)
        if not feasible.any():
            break
        cand = np.where(feasible)[0]
        # pick max density
        d = density[feasible]
        i = cand[int(np.argmax(d))]
        total += float(values[i])
        cap -= float(weights[i])
        avail[i] = False
    return total

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _CALL_COUNT
    if remaining_capacity is None:
        return -1
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    if n == 0 or cap <= 0:
        return -1
    # feasible check
    feasible = weights <= cap
    if not feasible.any():
        return -1

    now = _now()
    elapsed = now - _START
    _CALL_COUNT += 1
    
    # fast fallback heuristic
    def fast_index():
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        # among feasible
        fd = np.where(feasible, density, -np.inf)
        return int(np.argmax(fd))

    if elapsed > (150.0 - _DEADLINE_MARGIN):
        # hard safety: just return fast
        return fast_index()

    # search: candidate first picks
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    
    # Annealing parameters
    # T starts high and decreases as _CALL_COUNT increases
    # We assume ~100 calls per instance (100 items max)
    # Let's scale T such that it goes from 1.0 to 0.0 over 100 calls
    # T = max(0, 1 - _CALL_COUNT / 100)
    T = max(0.0, 1.0 - (_CALL_COUNT - 1) / 100.0)
    
    # If T is very low, we might just want the best few
    # But we also want to keep the deterministic candidates to ensure we don't miss the obvious best
    cand_indices = set()
    
    # Always include the top deterministic candidates
    fd = np.where(feasible, density, -np.inf)
    cand_indices.add(int(np.argmax(fd)))                       # density
    fv = np.where(feasible, values, -np.inf)
    cand_indices.add(int(np.argmax(fv)))                       # value
    fw = np.where(feasible, -weights, np.inf)
    cand_indices.add(int(np.argmin(np.where(feasible, weights, np.inf))))  # lightest
    # value - small penalty
    fp = np.where(feasible, values - 0.1*weights, -np.inf)
    cand_indices.add(int(np.argmax(fp)))
    
    feas_idx = np.where(feasible)[0]
    feas_values = values[feasible]
    
    # If we have time and T > 0, add stochastic candidates
    # We need to be careful with the time budget.
    # Each _rollout_value takes O(N^2) worst case? No, O(N log N) or O(N) if we are smart.
    # Current _rollout_value is O(N^2) because it scans for max density in a loop.
    # With N=100, N^2 = 10000 ops. Very fast.
    # So we can afford more candidates.
    
    # Let's add candidates based on a softmax over values? Or just random?
    # The prompt suggests "different optimization or search method".
    # Simulated annealing typically involves accepting worse solutions with probability exp(-delta/T).
    # Here we are picking the *first* item. 
    # Let's interpret "annealing" as: 
    # Early in the build: pick from a wider distribution of good items.
    # Late in the build: pick from the very best items.
    
    # We can rank feasible items by a score (e.g., density) and pick from the top K, 
    # where K decreases as T decreases?
    # Or pick from all feasible items with probability proportional to exp(score/T)?
    
    # Let's try: pick 5 random feasible items, but bias towards higher value/density.
    # Score = density
    # P(i) proportional to exp(density[i] / T)
    
    if T > 0.01 and len(feas_idx) > 0:
        # Use density as the score
        scores = density[feasible]
        # Normalize scores to avoid overflow/underflow in exp
        # max_score = np.max(scores)
        # exp_scores = np.exp((scores - max_score) / T)
        
        # Simpler: just pick random items from the top 50% density?
        # Let's stick to the "different strategy" directive.
        # Current: fixed set of heuristics + random.
        # New: Stochastic selection where the "temperature" controls the spread.
        
        # Let's add 3 candidates sampled from a distribution based on density.
        # If T is high, it's uniform. If T is low, it's argmax.
        
        # To keep it fast and simple:
        # If T is high, pick 3 random from all feasible.
        # If T is low, pick 3 from top 10 density.
        
        # Let's do a linear interpolation of the "pool size" for random selection.
        # Pool size goes from N to 1 as T goes from 1 to 0.
        pool_size = max(1, int(len(feas_idx) * T))
        
        # Sort feasible indices by density descending
        sort_order = np.argsort(-density[feasible])
        top_pool = feas_idx[sort_order[:pool_size]]
        
        # Pick 2 random from this pool
        if len(top_pool) > 0:
            picks = np.random.choice(top_pool, size=min(2, len(top_pool)), replace=False)
            for p in picks:
                cand_indices.add(int(p))
        
    best_idx = -1
    best_val = -np.inf
    # limit candidates to keep budget
    cands = list(cand_indices)
    for ci in cands:
        v = _rollout_value(cap, weights, values, ci)
        if v > best_val:
            best_val = v
            best_idx = ci
        # re-check time
        if time.time() - _START > (150.0 - _DEADLINE_MARGIN):
            break
    if best_idx < 0:
        best_idx = fast_index()
    return int(best_idx)
