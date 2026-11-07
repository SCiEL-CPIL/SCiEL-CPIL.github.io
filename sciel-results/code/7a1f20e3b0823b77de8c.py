"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    pair_sum = d[:, None] + d[None, :]
    waste = capacity - pair_sum

    # Base: reward tight fits with inverse waste, sharpened by high power
    base = np.where(waste >= 0, 1.0 / (waste + 1.0), 1e-9)
    power = 10.0
    base = np.power(base, power)

    # Soft-min bonus: smooth surrogate for "tightest fit"
    # For pairs that fit, bonus peaks when waste is near 0.
    # Use a Gaussian in waste: exp(-waste^2 / (2*sigma^2))
    sigma = 5.0
    fit_mask = waste >= 0
    soft_bonus = np.zeros((n, n), dtype=np.float64)
    soft_bonus[fit_mask] = np.exp(-0.5 * (waste[fit_mask] / sigma) ** 2)

    # Complementarity structure: sort items and identify ideal partner pairs
    # via two-pointer sweep. Assign higher attractiveness to pairs that are
    # near-optimal complements in sorted order.
    sorted_idx = np.argsort(d)
    sorted_d = d[sorted_idx]

    # Two-pointer sweep to find ideal complement pairs
    # For each item i (in sorted order), find j such that sorted_d[i] + sorted_d[j] <= capacity
    # and the sum is as large as possible (tightest fit)
    ideal_partner = np.zeros(n, dtype=np.int64)
    j = n - 1
    for i in range(n):
        while j >= i and sorted_d[i] + sorted_d[j] > capacity:
            j -= 1
        if j >= i:
            ideal_partner[i] = j
        else:
            ideal_partner[i] = i  # fallback: pair with self

    # Map back to original indices
    ideal_partner_orig = np.zeros(n, dtype=np.int64)
    for k in range(n):
        orig_idx = sorted_idx[k]
        partner_sorted_idx = ideal_partner[k]
        partner_orig_idx = sorted_idx[partner_sorted_idx]
        ideal_partner_orig[orig_idx] = partner_orig_idx

    # Build complementarity score vectorized:
    # pos_of_orig[i] = position of original item i in sorted order
    pos_of_orig = np.zeros(n, dtype=np.int64)
    for k in range(n):
        pos_of_orig[sorted_idx[k]] = k

    # pos_ideal[i] = sorted position of ideal_partner_orig[i]
    pos_ideal = pos_of_orig[ideal_partner_orig]

    # dist[i, j] = |pos_of_orig[j] - pos_ideal[i]|
    # pos_of_orig[j] for all j is just pos_of_orig
    # pos_ideal[i] for all i is pos_ideal
    dist = np.abs(pos_of_orig[None, :] - pos_ideal[:, None])
    complementarity = np.exp(-0.5 * dist)

    # Mask out non-fitting pairs in complementarity
    complementarity = np.where(fit_mask, complementarity, 0.0)

    # Combine: base * (1 + alpha*complementarity + beta*soft_bonus)
    result = base * (1.0 + 0.5 * complementarity + 0.3 * soft_bonus)

    # Ensure non-finite values are replaced
    result = np.where(np.isfinite(result), result, 1e-9)

    # Ensure all non-positive become 1e-9
    result = np.where(result > 0, result, 1e-9)

    return result
# EVOLVE-BLOCK-END
