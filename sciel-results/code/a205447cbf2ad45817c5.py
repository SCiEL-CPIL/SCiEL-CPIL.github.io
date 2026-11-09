"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 135  # Safety margin before 150s kill


def _simulate_bin_packing_fast(items, capacity, param_type, param_value):
    """
    Fast simulation using pre-allocated arrays and direct scoring.
    Returns: number of bins used
    """
    n = len(items)
    # Pre-allocate max possible bins
    max_bins = n + 1
    remaining = np.zeros(max_bins, dtype=np.float64)
    n_bins = 0
    
    for item in items:
        if n_bins == 0:
            remaining[0] = capacity - item
            n_bins = 1
            continue
        
        valid = remaining[:n_bins] >= item
        valid_count = np.count_nonzero(valid)
        
        if valid_count == 0:
            remaining[n_bins] = capacity - item
            n_bins += 1
            continue
        
        # Compute scores for valid bins only
        valid_remaining = remaining[:n_bins][valid]
        gap = valid_remaining - item
        
        if param_type == 'alpha':
            scores = -gap ** param_value
        elif param_type == 'target':
            scores = -((gap) - param_value) ** 2
        elif param_type == 'norm_alpha':
            # Use max(item, 1.0) to avoid division by tiny values
            safe_item = max(item, 1.0)
            ratio = gap / safe_item
            scores = -ratio ** param_value
        elif param_type == 'combined':
            # score = -(gap^alpha) + beta * (remaining/capacity)
            # param_value is (alpha, beta)
            alpha, beta = param_value
            fill_frac = valid_remaining / capacity
            scores = -(gap ** alpha) + beta * fill_frac
        elif param_type == 'content_bf':
            # Best fit with content bonus: score = -(gap) + beta * (capacity - remaining)
            # This encourages placing in bins that already have content (higher fill)
            # param_value is beta
            beta = param_value
            content = capacity - valid_remaining
            scores = -gap + beta * content
        elif param_type == 'soft_min':
            # Smooth best-fit: score = -soft_min(gaps, t)
            # param_value is t (temperature)
            t = param_value
            # soft_min approximates min, so -soft_min approximates -min
            # We want to maximize score, so we want to minimize gap.
            # score = -soft_min(gap, t)
            # Using log-sum-exp for stability: soft_min(x, t) = -t * log(sum(exp(-x/t)))
            # But we want -soft_min, so score = t * log(sum(exp(-gap/t)))
            # This is a smooth approximation of min(gap).
            # We want to maximize this, which means minimizing gap.
            # Let's verify: if gap = [1, 2], t=1.
            # exp(-1)=0.36, exp(-2)=0.13. sum=0.49. log(0.49)=-0.71. score = -0.71.
            # If gap = [1, 100]. exp(-1)=0.36, exp(-100)~0. sum=0.36. log(0.36)=-1.02. score=-1.02.
            # The score is higher (less negative) for the tighter fit. Correct.
            # However, for large gaps, exp(-gap/t) underflows.
            # Let's use a stable version.
            # score = t * log(sum(exp(-gap/t)))
            # Let m = min(gap).
            # score = t * ( -m/t + log(sum(exp(-(gap-m)/t))) )
            # score = -m + t * log(sum(exp(-(gap-m)/t)))
            m = np.min(gap)
            exp_terms = np.exp(-(gap - m) / t)
            scores = -m + t * np.log(np.sum(exp_terms))
            # Wait, the score should be a vector of scores for each bin?
            # No, the task says "scores each bin". The item goes to the HIGHEST scoring bin.
            # If I use a global soft-min, it returns a single scalar.
            # The interface requires an array of scores, one per bin.
            # So I can't use a global soft-min for the selection mechanism directly unless I interpret it differently.
            # The prompt says: "score each bin ... item goes to the highest-scoring bin".
            # So I need a per-bin score.
            # Maybe the "soft min" inspiration implies a different scoring function per bin?
            # Or maybe it means using a soft-min in the *evaluation* of the strategy? No, the strategy is the scoring function.
            # Let's stick to per-bin scores.
            # Perhaps a per-bin score that approximates "how good is this bin relative to others"?
            # That's hard without knowing the distribution of other bins.
            # Let's drop 'soft_min' as a per-bin scoring function for the priority() call,
            # or implement it as a fallback if it doesn't make sense.
            # Actually, let's just return -gap for now in this branch to be safe, or remove it.
            scores = -gap
        else:
            scores = -gap
        
        best_local = np.argmax(scores)
        valid_idx = np.where(valid)[0]
        best_global = valid_idx[best_local]
        remaining[best_global] -= item
    
    return n_bins


def _l1_lower_bound(items, capacity):
    total = np.sum(items)
    max_item = np.max(items)
    lb1 = int(np.ceil(total / capacity))
    lb2 = int(np.ceil(max_item))
    return max(lb1, lb2)


def _generate_sequences(n_sequences=5, n_items=1000, capacity=100):
    """Generate Weibull sequences for testing."""
    sequences = []
    for seed in range(n_sequences):
        rng = np.random.RandomState(seed)
        items = rng.weibull(2, size=n_items) * 20
        items = items[items <= capacity]
        while len(items) < n_items:
            new_items = rng.weibull(2, size=n_items) * 20
            new_items = new_items[new_items <= capacity]
            items = np.concatenate([items, new_items])[:n_items]
        sequences.append(items)
    return sequences


def _evaluate_params(sequences, capacity, param_type, param_value):
    """Evaluate a parameter set across all sequences. Returns mean excess %."""
    total_excess = 0.0
    for seq in sequences:
        bins_used = _simulate_bin_packing_fast(seq, capacity, param_type, param_value)
        lb = _l1_lower_bound(seq, capacity)
        excess = (bins_used - lb) / lb * 100
        total_excess += excess
    return total_excess / len(sequences)


def _fit_parameters():
    """
    Search over scoring function families with coarse grid + local refinement.
    """
    sequences = _generate_sequences()
    capacity = 100
    
    best_score = float('inf')
    best_params = ('alpha', 1.0)
    
    # === Phase 1: Coarse grid search ===
    
    # Alpha family: score = -(remaining - item)^alpha
    alphas_coarse = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for alpha in alphas_coarse:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'alpha', alpha)
        if score < best_score:
            best_score = score
            best_params = ('alpha', alpha)
    
    # Target family: score = -((remaining - item) - t)^2
    targets_coarse = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0]
    for t in targets_coarse:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'target', t)
        if score < best_score:
            best_score = score
            best_params = ('target', t)
    
    # Norm alpha family
    norm_alphas = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    for na in norm_alphas:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'norm_alpha', na)
        if score < best_score:
            best_score = score
            best_params = ('norm_alpha', na)
    
    # Combined family: score = -(gap^alpha) + beta * (remaining/capacity)
    # This blends tight-fit preference with a fill-level bonus
    for alpha_c in [0.5, 0.75, 1.0, 1.5, 2.0]:
        if time.time() > _deadline - 20:
            break
        for beta_c in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            if time.time() > _deadline - 20:
                break
            pv = (alpha_c, beta_c)
            score = _evaluate_params(sequences, capacity, 'combined', pv)
            if score < best_score:
                best_score = score
                best_params = ('combined', pv)
    
    # Content-BF family: score = -gap + beta * content
    # content = capacity - remaining
    # This is a direct implementation of the "content bonus" idea.
    # If beta is large, it strongly prefers bins with more content.
    # If beta is 0, it's pure best-fit.
    # Let's search for beta.
    betas_cb = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    for beta in betas_cb:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'content_bf', beta)
        if score < best_score:
            best_score = score
            best_params = ('content_bf', beta)
    
    # === Phase 2: Local refinement around best ===
    if time.time() < _deadline - 15:
        ptype, pval = best_params
        if ptype == 'alpha':
            # Refine alpha in [pval*0.5, pval*1.5] with finer steps
            lo = max(0.05, pval * 0.5)
            hi = pval * 1.5 + 0.5
            n_steps = 12
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                a = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'alpha', a)
                if score < best_score:
                    best_score = score
                    best_params = ('alpha', a)
        elif ptype == 'target':
            lo = max(0.0, pval - 10.0)
            hi = pval + 10.0
            n_steps = 12
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                t = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'target', t)
                if score < best_score:
                    best_score = score
                    best_params = ('target', t)
        elif ptype == 'norm_alpha':
            lo = max(0.1, pval * 0.5)
            hi = pval * 1.5 + 0.5
            n_steps = 12
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                na = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'norm_alpha', na)
                if score < best_score:
                    best_score = score
                    best_params = ('norm_alpha', na)
        elif ptype == 'combined':
            alpha_c, beta_c = pval
            for da in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                if time.time() > _deadline - 10:
                    break
                for db in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    if time.time() > _deadline - 10:
                        break
                    a2 = max(0.1, alpha_c + da)
                    b2 = max(0.0, beta_c + db)
                    pv = (a2, b2)
                    score = _evaluate_params(sequences, capacity, 'combined', pv)
                    if score < best_score:
                        best_score = score
                        best_params = ('combined', pv)
        elif ptype == 'content_bf':
            # Refine beta
            lo = max(0.0, pval * 0.5)
            hi = pval * 1.5 + 0.5
            n_steps = 12
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                b = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'content_bf', b)
                if score < best_score:
                    best_score = score
                    best_params = ('content_bf', b)
    
    return best_params, best_score


# Try fitting at module load time, but with a time guard
_start_time = time.time()

try:
    _BEST_PARAMS, _BEST_SCORE = _fit_parameters()
except Exception:
    _BEST_PARAMS = ('alpha', 1.0)
    _BEST_SCORE = float('inf')

# Build the scoring function based on best params
_param_type, param_value = _BEST_PARAMS


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    bins = np.asarray(bins, dtype=np.float64)
    remaining = bins
    valid_mask = remaining >= item
    
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    gap = remaining[valid_mask] - item
    
    if _param_type == 'alpha':
        scores[valid_mask] = -gap ** param_value
    elif _param_type == 'target':
        scores[valid_mask] = -((gap) - param_value) ** 2
    elif _param_type == 'norm_alpha':
        safe_item = max(item, 1.0)
        ratio = gap / safe_item
        scores[valid_mask] = -ratio ** param_value
    elif _param_type == 'combined':
        alpha, beta = param_value
        fill_frac = remaining[valid_mask] / 100.0
        scores[valid_mask] = -(gap ** alpha) + beta * fill_frac
    elif _param_type == 'content_bf':
        beta = param_value
        content = 100.0 - remaining[valid_mask]
        scores[valid_mask] = -gap + beta * content
    else:
        scores[valid_mask] = -gap
    
    return scores
# EVOLVE-BLOCK-END
