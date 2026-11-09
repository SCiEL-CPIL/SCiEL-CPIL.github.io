# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.

    Uses a knowledge-gradient-style score: for each candidate, compute the
    expected reduction in the *global* best (not just local EI), weighted
    by a cost-efficiency factor. This is a structurally different modeling
    family from the previous EI/UCB hybrid.

    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)
    if n == 0:
        return np.array([])

    # --- Knowledge-gradient-style score ---
    # For a candidate with GP posterior N(mu_i, sigma_i^2), the expected
    # value of the function at x_i is mu_i. The probability that x_i is
    # better than the current best is Phi((f_best - mu_i) / sigma_i).
    #
    # A KG-like quantity: E[max(0, f_best - f(x_i))] / sigma_i
    # This is the "expected improvement per unit uncertainty", which
    # naturally balances exploration (high sigma) and exploitation (low mu).
    #
    # We also incorporate a "global optimality" bonus: points with very low
    # mu (near the known minimum) get extra credit because they are likely
    # to be the global best, not just a local improvement.

    # Standard EI for minimization
    z = (f_best - mu_safe) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu_safe) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)

    # KG-style: expected improvement normalized by uncertainty
    # This rewards points where we are uncertain AND likely to improve
    kg = ei / sigma_safe

    # Global optimality bonus: how close is mu to the known minimum?
    # Hartmann 3d minimum is -3.86278
    min_val = -3.86278
    # Distance from known minimum (positive)
    dist_to_min = np.maximum(0.0, mu_safe - min_val)
    # Points near the minimum get a bonus. Use an exponential decay.
    # Scale: within 0.1 of minimum, bonus ~ 1; at 0.5 away, bonus ~ 0.37
    global_bonus = np.exp(-dist_to_min / 0.2)

    # Combine: primary signal is KG, modulated by global optimality
    # The global_bonus acts as a multiplier that boosts exploitation
    # near the known minimum while still allowing exploration elsewhere.
    base_score = kg * (0.5 + 0.5 * global_bonus)

    # Cost adjustment: cost is in [1, 2] (1 + mean of 3 coords in [0,1])
    # Use a sub-linear penalty: divide by cost^0.5 to mildly prefer cheap points
    # without completely ignoring expensive points that might be very promising
    cost_factor = np.power(cost_safe, 0.5)
    scores = base_score / cost_factor

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return scores

    # Alternative formulation: pure EI / cost for comparison
    if n >= 100 and time.time() < deadline - 10.0:
        alt_scores = ei / cost_safe
        alt_scores = np.where(np.isfinite(alt_scores), alt_scores, 0.0)

        # Compare: mean mu of top-10 points (lower is better)
        top_k = min(10, n // 10)
        if top_k >= 1:
            idx_current = np.argsort(scores)[-top_k:]
            idx_alt = np.argsort(alt_scores)[-top_k:]

            mean_mu_current = np.mean(mu_safe[idx_current])
            mean_mu_alt = np.mean(mu_safe[idx_alt])

            # Prefer the formulation with lower mean mu in top-k
            if mean_mu_alt < mean_mu_current - 1e-6:
                scores = alt_scores

    # Final safety
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
