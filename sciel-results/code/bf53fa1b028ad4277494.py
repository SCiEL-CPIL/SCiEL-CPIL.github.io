"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t as t_dist

# Global state to track evaluation count for adaptive exploration
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 300s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Increment evaluation count
    _eval_count += 1
    
    # Strategy: Student's t-based Expected Improvement
    # Standard EI assumes a Gaussian posterior. With only ~30 samples,
    # the uncertainty is often underestimated (heavy tails are ignored).
    # Using a t-distribution with low degrees of freedom (df) accounts for
    # this extra uncertainty, making the acquisition function more robust
    # and encouraging exploration.
    
    # Degrees of freedom: start low (more exploration) and increase slightly
    # as we gather more data, but keep it low enough to maintain heavy tails.
    # With 5 initial points + (t-5) evaluations, we have t-4 total points after the first 5.
    # However, GP uncertainty is not just about sample size but also about kernel fit.
    # A fixed low df like 5-10 is often a good heuristic for small sample sizes.
    # Let's use a schedule: df = 5 + 0.5 * _eval_count, capped at 20.
    # Actually, simpler: just use a constant low df like 8 or 10.
    # Let's try df = 8.
    df = 8.0
    
    # Compute standardized improvement: z = (f_best - mu) / sigma
    # For minimization, improvement is f_best - f(x).
    # We want E[max(0, f_best - f(x))].
    # If f(x) ~ t_df(mu, sigma), then let y = (f(x) - mu) / sigma ~ t_df(0, 1).
    # f(x) = mu + sigma * y.
    # Improvement I = max(0, f_best - mu - sigma * y).
    # Let alpha = (f_best - mu) / sigma.
    # I = max(0, sigma * (alpha - y)).
    # E[I] = sigma * E[max(0, alpha - y)].
    
    # For y ~ t_df(0,1), the PDF is f(y) and CDF is F(y).
    # E[max(0, alpha - y)] = integral_{-inf}^{alpha} (alpha - y) f(y) dy
    # = alpha * F(alpha) - integral_{-inf}^{alpha} y f(y) dy
    
    # Note: For t-distribution, the mean is 0 for df > 1.
    # The term integral_{-inf}^{alpha} y f(y) dy is related to the truncated mean.
    # Alternatively, we can use the property:
    # E[max(0, alpha - y)] = alpha * (1 - F(-alpha)) + f(-alpha) * (df / (df - 1)) * (1 - F(-alpha)) ? No, that's for Gaussian.
    
    # Let's derive it properly or use a numerical integration? No, too slow.
    # Let's use the formula for Expected Improvement with t-distribution.
    # Reference: "Expected Improvement for Bayesian Optimization with Student's t"
    # E[Im] = (f_best - mu) * P(f(x) < f_best) + sigma * E[(f_best - f(x)) | f(x) < f_best] * P(f(x) < f_best)
    # Wait, E[max(0, A)] = E[A * 1(A>0)].
    # Let Z = (f(x) - mu) / sigma ~ t_df.
    # A = f_best - mu - sigma * Z = sigma * (alpha - Z).
    # We want E[max(0, alpha - Z)] * sigma.
    
    # Let's compute this expectation analytically or semi-analytically.
    # For Gaussian: E[max(0, alpha - Z)] = alpha * Phi(alpha) + phi(alpha).
    # For t-distribution, there is no closed form as simple as that involving just CDF/PDF.
    # However, we can approximate it or use a different acquisition function that is easier.
    
    # Alternative: Use Upper Confidence Bound (UCB) with t-quantiles.
    # UCB = mu + kappa * sigma * t_quantile(1 - 1/(2t), df) ? No, for minimization we want LCB.
    # LCB = mu - kappa * sigma * t_quantile(1 - 1/(2t), df) ?
    # Actually, let's stick to the LCB family but use t-quantiles for the confidence bound.
    # This is "t-UCB" or "t-LCB".
    # The confidence bound width is scaled by the t-quantile instead of the z-quantile (which is 1 for 1-sigma).
    # For a 95% confidence interval, Gaussian uses 1.96, t-distribution uses t.ppf(0.975, df).
    # For df=8, t.ppf(0.975, 8) is approx 2.306.
    # For Gaussian, it's 1.96.
    # So t-LCB will be more conservative (wider bound) for the same probability, encouraging more exploration.
    
    # Let's compute the t-quantile for a fixed probability, say 0.95 or 0.975.
    # Or we can use the adaptive beta approach but scale beta by the ratio of t-quantile to z-quantile.
    # Let's just use a fixed beta that reflects the t-distribution's heavier tails.
    # A common choice is to use the t-quantile directly as the multiplier.
    # Let's use p = 0.95.
    # t_quantile = t_dist.ppf(0.95, df)
    # scores = mu - t_quantile * sigma
    # We want to maximize this (since lower is better, we pick the most negative? No).
    # Wait, LCB = mu - beta * sigma.
    # We want the point with the lowest possible value.
    # So we want to minimize LCB.
    # But the evaluator picks the HIGHEST score.
    # So we should return -LCB = -mu + beta * sigma.
    # Or simply return (beta * sigma - mu).
    
    # Let's calculate the t-quantile.
    # To avoid calling scipy.stats.ppf every time (it's fast but let's be safe),
    # we can precompute it or just call it. It's O(1).
    t_quantile = t_dist.ppf(0.95, df)
    
    # Adaptive beta: start high, decay.
    # The t-quantile already adds exploration pressure.
    # Let's combine: beta = t_quantile * decay_factor
    # Or just use the t-quantile as the base beta.
    # Let's use a schedule for the probability p.
    # Early on, use higher p (more exploration). Later, lower p.
    # p = 0.95 + 0.02 * np.exp(-0.1 * _eval_count)
    # p = min(p, 0.99)
    # t_quantile = t_dist.ppf(p, df)
    
    # Let's try a simpler approach:
    # beta = t_dist.ppf(0.95, df) * (1.0 + 0.5 * np.exp(-0.1 * _eval_count))
    
    beta = t_dist.ppf(0.95, df) * (1.0 + 0.5 * np.exp(-0.1 * _eval_count))
    
    scores = beta * sigma - mu
    
    # Check for non-finite values
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
    # Check deadline
    if time.time() > deadline:
        # Return best found so far? No, just return the computed scores.
        # The computation is O(N) and very fast.
        pass

    return scores
# EVOLVE-BLOCK-END
