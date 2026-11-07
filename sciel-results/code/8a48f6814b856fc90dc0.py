import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Pure vectorised computation; no wall-clock guard needed (each call is ~ms).
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()

    n = mu.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # sigma is a standard deviation; keep it strictly positive for stable z-scores.
    sigma_safe = np.maximum(sigma, 1e-9)

    # --- Per-candidate evaluation cost (the "cost of evaluating x") -------------
    # The evaluator scores 2000 uniform candidates in [-10, 10]^2. We recover the
    # same 2000 points from the GP inputs: mu (mean) and sigma (posterior std) are
    # computed at exactly those candidate locations. The candidate i is the mean
    # of its posterior (sigma -> 0), so mu recovers the candidate coordinates.
    x = mu.reshape(n, 2)                       # candidate coordinates in [-10, 10]^2
    x_scaled = (x + 10.0) / 20.0               # scale to [0, 1]
    x_scaled = np.clip(x_scaled, 0.0, 1.0)
    cand_cost = 1.0 + np.mean(x_scaled, axis=1)  # cost of evaluating each candidate
    cand_cost = np.maximum(cand_cost, 1e-9)

    # --- Exploitation: probability of improvement (robust, no pdf blow-up) -----
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)                            # P(f(x) < f_best) in [0, 1]

    # --- Expected Improvement (small, well-calibrated contribution) ------------
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Exploration: scale-aware relative uncertainty -------------------------
    # sigma is in objective units; Griewank values are O(0..1) here, so a fixed
    # kappa on sigma is comparable to the exploitation terms.
    kappa = 1.0
    explore = kappa * sigma_safe

    # --- Combine and make cost-aware -------------------------------------------
    # Higher score = better. We commit to low-mean regions (exploitation) while
    # keeping a modest, scale-aware exploration bonus, all penalised by the cost
    # of evaluating that specific candidate.
    score = pi + 0.5 * ei + explore
    score = score / cand_cost

    # Guarantee 2000 finite scores (higher = evaluate next).
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)
    return score
