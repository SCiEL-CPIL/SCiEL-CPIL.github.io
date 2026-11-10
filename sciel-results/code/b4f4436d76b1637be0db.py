"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item (larger = more attractive)
    """
    t_start = time.time()
    deadline = t_start + 280.0

    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Ensure correct shapes
    if prizes.ndim == 2 and prizes.shape[0] == 5 and prizes.shape[1] == 100:
        p = prizes
    elif prizes.ndim == 1 and prizes.shape[0] == 100:
        p = prizes.reshape(1, 100)
    else:
        flat = prizes.reshape(-1)
        if flat.shape[0] >= 100:
            p = flat[:100].reshape(1, 100)
        else:
            p = np.zeros((1, 100))
            p[0, :flat.shape[0]] = flat

    if weights.ndim == 2 and weights.shape[0] == 5 and weights.shape[1] == 100:
        w = weights
    elif weights.ndim == 1 and weights.shape[0] == 100:
        w = weights.reshape(1, 100)
    else:
        flat = weights.reshape(-1)
        if flat.shape[0] >= 100:
            w = flat[:100].reshape(1, 100)
        else:
            w = np.zeros((1, 100))
            w[0, :flat.shape[0]] = flat

    n_items = 100
    n_constraints = p.shape[0]
    capacity = 25.0

    # Total prize and weight per item
    total_prize = p.sum(axis=0)       # (100,)
    total_weight = w.sum(axis=0)      # (100,)
    max_w = w.max(axis=0)             # (100,)

    # Core ratio: total prize / total weight
    ratio = total_prize / (total_weight + 1e-9)

    # Per-constraint ratios
    if n_constraints > 1:
        cr = p / (w + 1e-9)           # (n_constraints, 100)
        # Soft-min of per-constraint ratios via log-sum-exp with temperature 0.05
        temp = 0.05
        neg_cr = -cr / temp
        max_neg = neg_cr.max(axis=0, keepdims=True)
        lse = max_neg + np.log(np.exp(neg_cr - max_neg).sum(axis=0, keepdims=True))
        soft_min_cr = -temp * lse.squeeze(axis=0)  # (100,)
    else:
        soft_min_cr = ratio

    # Bottleneck penalty: items heavy in any single constraint are less desirable
    bottleneck = max_w

    # Combined score: high ratio, high soft-min per-constraint ratio, low bottleneck
    base_score = ratio * soft_min_cr / ((1.0 + bottleneck) ** 2)

    # Normalize to [0, 1]
    bmin = base_score.min()
    bmax = base_score.max()
    if bmax > bmin:
        normalized = (base_score - bmin) / (bmax - bmin)
    else:
        normalized = np.ones(n_items) / n_items

    # Sharpen: use a higher power to strongly favour top items
    power = 3.0
    scores = normalized ** power

    # Small floor to keep all entries positive
    scores = scores + 1e-6

    # Guard against non-positive or non-finite values
    scores = np.where(scores > 0, scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)

    # Deadline guard
    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
