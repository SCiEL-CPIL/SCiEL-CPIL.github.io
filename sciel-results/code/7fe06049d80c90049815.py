"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for safety
_start_time = time.time()
_DEADLINE = 330.0  # Seconds

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Look-Ahead Expected Improvement (LAEI).
    
    For minimization of f, we want to find low f.
    The GP predicts mean mu and std sigma for the objective f.
    
    LAEI strategy:
    1. For each candidate point x_i with posterior (mu_i, sigma_i), consider the expected
       improvement if we observe a value from N(mu_i, sigma_i^2).
    2. The look-ahead considers how observing this value updates the posterior at the
       current best point (where mu is minimized).
    3. We approximate the value by computing the expected minimum after a hypothetical
       observation at x_i.
    
    Parameters:
    mu: array of shape (2000,) - GP posterior mean
    sigma: array of shape (2000,) - GP posterior std dev
    f_best: float - best observed value so far (lowest)
    
    Returns:
    scores: array of shape (2000,) - higher is better
    """
    # Check time budget
    current_time = time.time()
    if current_time > _DEADLINE - 5.0:
        # Fallback to a simple exploitation strategy if running out of time
        mu_arr = np.asarray(mu, dtype=np.float64).ravel()
        return -mu_arr
    
    # Ensure inputs are numpy arrays and 1D
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Safety: sigma should be non-negative. Clip small negative values due to numerical noise.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Find the index with the lowest mean (current best prediction)
    idx_best = np.argmin(mu)
    mu_best = mu[idx_best]
    sigma_best = sigma_safe[idx_best]
    
    # The value f_best is the lowest observed value. We use max(mu_best, f_best) as the
    # current reference for improvement, but typically mu_best <= f_best if the GP is well-tuned.
    # For EI calculation, we use f_best as the threshold.
    reference = f_best
    
    # For each candidate i, we want to compute the expected improvement of the minimum
    # after observing y_i ~ N(mu_i, sigma_i^2).
    # 
    # The key insight: observing y_i at x_i updates our belief. If y_i < f_best, the new
    # best becomes y_i. If y_i > f_best, the best remains f_best, but the posterior at
    # the best point might improve.
    #
    # A practical approximation:
    # 1. Compute standard EI for each point: EI_i = E[max(f_best - y_i, 0)]
    # 2. Add a look-ahead bonus that estimates the probability that observing y_i leads
    #    to a better posterior mean at the current best point.
    #
    # The look-ahead bonus: if sigma_i is large and mu_i is low, there's a good chance
    # y_i is much better than f_best. But we also want to consider that even if y_i is
    # not better, the correlation structure might improve our estimate at x_best.
    #
    # Without the full covariance matrix, we approximate the correlation by assuming
    # that nearby points (in input space) are correlated. Since we don't have input
    # coordinates, we use a simpler model:
    #
    # Approximate the expected reduction in the best value after observing y_i:
    # E[best_after - best_before] where best_before = f_best
    #
    # best_after = min(f_best, y_i, updated_mu_best)
    # where updated_mu_best is the posterior mean at x_best given y_i.
    #
    # A common approximation: updated_mu_best ≈ mu_best + rho * (y_i - mu_i)
    # where rho is the correlation between x_i and x_best. Without coordinates, we
    # approximate rho using the relative sigma values or a heuristic.
    #
    # Simpler approach: use a weighted combination of:
    # a) Standard EI (immediate improvement)
    # b) A bonus for high uncertainty at low-mean points (exploration)
    # c) A bonus for points that are likely to improve the posterior at the best point
    #
    # Let's compute:
    # 1. Standard EI
    z = (reference - mu) / sigma_safe
    from scipy.stats import norm
    ei = sigma_safe * (z * norm.cdf(z) + norm.pdf(z))
    ei = np.where(sigma_safe > 1e-8, ei, np.maximum(reference - mu, 0))
    
    # 2. Look-ahead bonus: probability that y_i < f_best times the expected gap
    # P(y_i < f_best) = Phi((f_best - mu_i) / sigma_i)
    prob_improve = norm.cdf(z)
    
    # Expected gap if y_i < f_best: E[f_best - y_i | y_i < f_best]
    # = sigma_i * [phi(z) / Phi(z) - z] for z > 0, but this is already captured in EI.
    
    # The look-ahead component: we estimate the expected improvement in the posterior
    # mean at x_best after observing y_i. This is approximately:
    # E[max(0, mu_best - (mu_best + rho * (y_i - mu_i)))]
    # = E[max(0, -rho * (y_i - mu_i))]
    #
    # For rho > 0 (positive correlation), this is positive when y_i < mu_i.
    # E[max(0, -rho * (y_i - mu_i))] = rho * E[max(0, mu_i - y_i)]
    # = rho * sigma_i * [phi(z) + z * (1 - Phi(z))] for the lower tail
    #
    # Actually, E[max(0, mu_i - y_i)] where y_i ~ N(mu_i, sigma_i^2):
    # Let d = y_i - mu_i ~ N(0, sigma_i^2)
    # E[max(0, -d)] = sigma_i * phi(0) = sigma_i / sqrt(2*pi)
    # Wait, that's not right. Let me recalculate.
    # E[max(0, mu_i - y_i)] = E[max(0, -d)] where d ~ N(0, sigma_i^2)
    # = sigma_i * E[max(0, -Z)] where Z ~ N(0,1)
    # = sigma_i * phi(0) = sigma_i / sqrt(2*pi)
    #
    # So the look-ahead bonus is proportional to rho * sigma_i / sqrt(2*pi).
    #
    # We need to estimate rho. A common heuristic: assume correlation decays with
    # the ratio of sigmas. Points with similar sigma might be closer in input space.
    # Alternatively, use a fixed moderate correlation estimate.
    #
    # Let's use a simpler approach: the look-ahead bonus is the expected reduction
    # in the best value, which we approximate as:
    # LAEI_i = ei_i + beta * prob_improve_i * sigma_i
    # where beta controls the weight of the exploration/look-ahead component.
    
    # Actually, let's use a cleaner formulation. The true LAEI for a single point is:
    # LAEI(x) = E_x[ min(f_best, y, mu_best_updated) - f_best ]
    # where the expectation is over y ~ N(mu(x), sigma(x)^2).
    #
    # Since min is concave and we're taking expectation, we can bound or approximate.
    # A practical approximation used in practice:
    # LAEI(x) ≈ EI(x) + gamma * P(y < f_best) * (f_best - mu(x)) * (sigma(x) / f_scale)
    #
    # Let's use a weighted combination that balances exploitation and exploration:
    # score = ei + alpha * prob_improve * sigma_safe
    
    # Tune alpha based on the spread of sigma to adapt exploration strength
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe) + 1e-8
    
    # Adaptive alpha: more exploration when uncertainty is high
    alpha = 1.0 + sigma_std / (sigma_mean + 1e-8)
    alpha = min(alpha, 5.0)  # Cap to prevent runaway
    
    scores = ei + alpha * prob_improve * sigma_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Convert to "higher is better" for the evaluator (which maximizes the score)
    # The evaluator picks the candidate with the highest score.
    # Since we want to minimize f, and higher scores mean "evaluate next",
    # we return the scores as-is (higher EI/LAEI means more promising).
    
    return scores

# EVOLVE-BLOCK-END
