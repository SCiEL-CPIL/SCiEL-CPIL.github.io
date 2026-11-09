"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 130  # Safety margin before 150s kill


def _simulate_bin_packing(items, capacity, score_fn, deadline=None):
    """
    Simulate online bin packing with a given scoring function.
    items: array of item sizes
    capacity: bin capacity
    score_fn: callable(item, remaining_array) -> score_array
    Returns: number of bins used
    """
    bins = []  # list of remaining capacities
    for item in items:
        if deadline is not None and time.time() > deadline:
            return None  # timeout sentinel
        if not bins:
            bins.append(capacity - item)
            continue

        remaining = np.array(bins)
        valid = remaining >= item

        if not np.any(valid):
            bins.append(capacity - item)
            continue

        # Compute scores for valid bins
        scores = score_fn(item, remaining)
        # Among valid bins, pick the one with highest score
        valid_indices = np.where(valid)[0]
        valid_scores = scores[valid_indices]
        best_idx = valid_indices[np.argmax(valid_scores)]
        bins[best_idx] -= item

    return len(bins)


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


def _make_score_fn(family, *args):
    """Build a scoring function closure for a given family/params."""
    if family == 'alpha':
        alpha = args[0]
        def score_fn(item, remaining):
            valid = remaining >= item
            scores = np.full(len(remaining), -np.inf, dtype=np.float64)
            if np.any(valid):
                scores[valid] = -(remaining[valid] - item) ** alpha
            return scores
        return score_fn
    if family == 'target':
        t = args[0]
        def score_fn(item, remaining):
            valid = remaining >= item
            scores = np.full(len(remaining), -np.inf, dtype=np.float64)
            if np.any(valid):
                scores[valid] = -((remaining[valid] - item) - t) ** 2
            return scores
        return score_fn
    if family == 'linear':
        a = args[0]
        def score_fn(item, remaining):
            valid = remaining >= item
            scores = np.full(len(remaining), -np.inf, dtype=np.float64)
            if np.any(valid):
                scores[valid] = a * remaining[valid]
            return scores
        return score_fn
    if family == 'quadratic':
        a, b = args
        def score_fn(item, remaining):
            valid = remaining >= item
            scores = np.full(len(remaining), -np.inf, dtype=np.float64)
            if np.any(valid):
                diff = remaining[valid] - item
                scores[valid] = -a * diff - b * diff ** 2
            return scores
        return score_fn
    if family == 'worst_fit':
        # Explicit worst fit: prefer max remaining
        def score_fn(item, remaining):
            valid = remaining >= item
            scores = np.full(len(remaining), -np.inf, dtype=np.float64)
            if np.any(valid):
                scores[valid] = remaining[valid]
            return scores
        return score_fn
    raise ValueError(f"unknown family {family}")


def _eval_params(sequences, family, args, deadline):
    """Evaluate a (family, args) pair across all sequences. Returns mean excess or inf on timeout."""
    total_excess = 0.0
    count = 0
    for seq in sequences:
        if time.time() > deadline:
            return float('inf')
        score_fn = _make_score_fn(family, *args)
        bins_used = _simulate_bin_packing(seq, 100, score_fn, deadline)
        if bins_used is None:
            return float('inf')
        lb = _l1_lower_bound(seq, 100)
        total_excess += (bins_used - lb) / lb * 100.0
        count += 1
    if count == 0:
        return float('inf')
    return total_excess / count


def _fit_parameters():
    """
    Coarse grid search over scoring-function families, then a local refinement.
    """
    sequences = _generate_sequences()
    
    # Coarse grid
    coarse_grid = []
    
    # Family: alpha (Best Fit variants)
    for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        coarse_grid.append(('alpha', (alpha,)))
        
    # Family: target (Ideal remaining space)
    # Expand range to include Worst-Fit territory (high t)
    for t in [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0]:
        coarse_grid.append(('target', (t,)))
        
    # Family: linear (a * remaining)
    # a < 0 is Best Fit, a > 0 is Worst Fit
    for a in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        coarse_grid.append(('linear', (a,)))
        
    # Family: worst_fit (explicit)
    coarse_grid.append(('worst_fit', ()))

    # Stage 1: coarse search
    best_score = float('inf')
    best_params = None
    
    # Time management
    total_time = 120.0
    coarse_deadline = time.time() + total_time * 0.5
    
    for family, args in coarse_grid:
        if time.time() > coarse_deadline:
            break
        s = _eval_params(sequences, family, args, coarse_deadline)
        if s < best_score:
            best_score = s
            best_params = (family,) + args

    # Stage 2: local refinement
    if best_params is not None and time.time() < _deadline - 10:
        fam = best_params[0]
        refine_deadline = _deadline - 5
        
        # Coordinate descent or fine grid around best
        refine_grid = []
        
        if fam == 'alpha':
            center = best_params[1]
            for d in np.linspace(-0.2, 0.2, 9):
                a = center + d
                if a > 0.01:
                    refine_grid.append(('alpha', (a,)))
        elif fam == 'target':
            center = best_params[1]
            for d in np.linspace(-10, 10, 11):
                t = center + d
                if 0 <= t <= 100:
                    refine_grid.append(('target', (t,)))
        elif fam == 'linear':
            center = best_params[1]
            for d in np.linspace(-0.5, 0.5, 11):
                a = center + d
                refine_grid.append(('linear', (a,)))
        elif fam == 'worst_fit':
            # Worst fit is a point, try nearby linear
            for a in [0.5, 0.8, 1.0, 1.2, 1.5]:
                refine_grid.append(('linear', (a,)))
        
        for family, args in refine_grid:
            if time.time() > refine_deadline:
                break
            s = _eval_params(sequences, family, args, refine_deadline)
            if s < best_score:
                best_score = s
                best_params = (family,) + args

    if best_params is None:
        best_params = ('alpha', 1.0)

    return best_params, best_score


_start_time = time.time()

try:
    _BEST_PARAMS, _BEST_SCORE = _fit_parameters()
except Exception:
    _BEST_PARAMS = ('alpha', 1.0)
    _BEST_SCORE = float('inf')

# Build the scoring function based on best params
_param_type = _BEST_PARAMS[0]
if _param_type == 'alpha':
    _ALPHA = _BEST_PARAMS[1]
    _USE_ALPHA = True
    _USE_TARGET = False
    _USE_LINEAR = False
    _USE_WORST = False
    _USE_QUAD = False
elif _param_type == 'target':
    _TARGET = _BEST_PARAMS[1]
    _USE_ALPHA = False
    _USE_TARGET = True
    _USE_LINEAR = False
    _USE_WORST = False
    _USE_QUAD = False
elif _param_type == 'linear':
    _LIN_A = _BEST_PARAMS[1]
    _USE_ALPHA = False
    _USE_TARGET = False
    _USE_LINEAR = True
    _USE_WORST = False
    _USE_QUAD = False
elif _param_type == 'worst_fit':
    _USE_ALPHA = False
    _USE_TARGET = False
    _USE_LINEAR = False
    _USE_WORST = True
    _USE_QUAD = False
elif _param_type == 'quadratic':
    _QUAD_A = _BEST_PARAMS[1]
    _QUAD_B = _BEST_PARAMS[2]
    _USE_ALPHA = False
    _USE_TARGET = False
    _USE_LINEAR = False
    _USE_WORST = False
    _USE_QUAD = True
else:
    _ALPHA = 1.0
    _USE_ALPHA = True
    _USE_TARGET = False
    _USE_LINEAR = False
    _USE_WORST = False
    _USE_QUAD = False


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    remaining = bins if isinstance(bins, np.ndarray) and bins.dtype == np.float64 else np.asarray(bins, dtype=np.float64)
    valid_mask = remaining >= item

    scores = np.full(remaining.shape, -np.inf, dtype=np.float64)

    if not valid_mask.any():
        return scores

    if _USE_ALPHA:
        scores[valid_mask] = -(remaining[valid_mask] - item) ** _ALPHA
    elif _USE_TARGET:
        scores[valid_mask] = -((remaining[valid_mask] - item) - _TARGET) ** 2
    elif _USE_LINEAR:
        scores[valid_mask] = _LIN_A * remaining[valid_mask]
    elif _USE_WORST:
        scores[valid_mask] = remaining[valid_mask]
    elif _USE_QUAD:
        diff = remaining[valid_mask] - item
        scores[valid_mask] = -_QUAD_A * diff - _QUAD_B * diff ** 2

    return scores
# EVOLVE-BLOCK-END
