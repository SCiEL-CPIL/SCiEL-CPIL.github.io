# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition for minimizing a multimodal objective.
    # mu, sigma, cost: shape (2000,)
    # Higher returned score => evaluate next.

    n = mu.shape[0]
    # Guard: ensure finite inputs and positive sigma.
    sigma = np.maximum(sigma, 1e-9)

    # --- Cost model ---
    # cost = 1 + mean(x_scaled) in [1, 2]. Cheaper near the center of the box.
    # We want to *encourage* cheap points (they let us do more evaluations),
    # but not so much that we ignore the objective.
    cost = np.maximum(cost, 1e-9)
    # cheapness in [0, 1]: 1 when cost=1 (center), 0 when cost=2 (corner).
    cheapness = np.clip(1.0 - (cost - 1.0), 0.0, 1.0)

    # --- Expected Improvement (for minimization) ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu)/sigma
    # This is zero when mu >= f_best (no improvement expected).
    z = (f_best - mu) / sigma
    # Numerically stable: for very negative z, cdf ~ 0, pdf ~ 0.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Lower-bound regret term (encourages exploration) ---
    # The minimum possible value in a region is roughly mu - sigma (one std below mean).
    # Regret if we *don't* explore = f_best - (mu - sigma) = (f_best - mu) + sigma.
    # This is positive whenever there's uncertainty, even when EI == 0.
    # We use a softened version: the "potential improvement" = (f_best - mu) + sigma.
    potential = (f_best - mu) + sigma
    # Normalize potential to [0, 1] using f_best as a scale reference.
    # f_best is the best observed; typical improvement magnitudes are O(1) for this problem.
    # Use a soft normalization to avoid division by zero.
    scale = max(abs(f_best), 1e-6)
    potential_norm = np.clip(potential / (scale + 1e-9), 0.0, 10.0)

    # --- Combine ---
    # 1) Cost-aware EI: divide by cost to get "improvement per unit cost".
    #    But don't over-penalize: use sqrt(cost) instead of cost to be gentler.
    ei_cost_aware = ei / np.sqrt(cost)

    # 2) Exploration bonus: potential improvement, also cost-aware.
    #    Weight it so that when EI is small (mu near f_best), exploration still drives.
    exploration = potential_norm / np.sqrt(cost)

    # 3) Cheapness bonus: small reward for cheap points to maximize budget usage.
    #    Keep this small so it doesn't dominate the objective-driven terms.
    cheap_bonus = 0.05 * cheapness

    # Weighted combination:
    # - EI dominates when there's clear improvement.
    # - Exploration kicks in when EI is small (uncertainty remains).
    # - Cheapness is a minor tiebreaker.
    # Use a dynamic blend: as we get closer to the optimum (f_best small),
    # trust EI more; early on, trust exploration more.
    # f_best starts high (uniform random on 3HC2 can be ~10-30), drops toward 0.
    # Use f_best as a proxy for "how much is left to find".
    f_best_abs = abs(f_best)
    # When f_best_abs is large, we're far from optimum -> explore more.
    # When f_best_abs is small, we're close -> exploit more.
    explore_weight = np.clip(1.0 - f_best_abs / 5.0, 0.0, 1.0)
    # explore_weight ~ 1 when f_best < 0 (impossible, but safe), ~ 0 when f_best > 5.
    # Actually f_best is the *minimum* observed, so it decreases over time.
    # Let's use: weight_exploration = clip(1 - f_best/5, 0, 1)
    # Early: f_best ~ 20 -> weight ~ 0 (exploit EI)
    # Late: f_best ~ 0 -> weight ~ 1 (explore)
    # Hmm, that's backwards. Let me reconsider.
    #
    # Actually: EI is good for exploitation. When f_best is high (early), EI values
    # are large and well-separated, so EI-based selection works well.
    # When f_best is low (late), EI values are small and noisy, so we need
    # exploration to find the global min.
    # So: exploration weight should *increase* as f_best decreases.
    # weight_explore = clip(1.0 - f_best_abs / 10.0, 0.0, 1.0)
    # Early (f_best~20): weight ~ 0.0 -> EI dominates
    # Late (f_best~0): weight ~ 1.0 -> exploration dominates
    weight_explore = np.clip(1.0 - f_best_abs / 10.0, 0.0, 1.0)

    # Final score: blend of EI-based and exploration-based, plus cheapness.
    # score = (1 - w) * ei_cost_aware + w * exploration + cheap_bonus
    # But we need the magnitudes to be comparable. Let's normalize each term.
    # ei_cost_aware: typically O(1) early, O(0.1) late.
    # exploration: potential_norm / sqrt(cost), potential_norm ~ O(1) early, O(0.1) late.
    # They're roughly comparable. Let's just blend them.

    score = (1.0 - weight_explore) * ei_cost_aware + weight_explore * exploration + cheap_bonus

    # Ensure finite output.
    score = np.where(np.isfinite(score), score, 0.0)
    # Ensure non-negative (higher = better).
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
