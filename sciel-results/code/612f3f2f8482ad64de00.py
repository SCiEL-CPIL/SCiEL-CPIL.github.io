import numpy as np
import time

# Deadline guard: kill after 330s wall-clock, use safety margin
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0
_start_time = time.time()

def _deadline_exceeded():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
    
    Uses Thompson Sampling: draw multiple samples from the GP posterior
    (defined by mu and sigma), compute the minimum cost-normalized objective
    across samples for each candidate, and score by the expected value.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Check deadline
    if _deadline_exceeded():
        # Return a simple fallback: cost-normalized EI-like score
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        pi = norm.cdf(z)
        ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
        score = ei / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    n = len(mu)
    
    # Number of Monte Carlo samples for Thompson Sampling
    # Keep this moderate to stay within time budget
    n_samples = 20
    
    # Cost normalization: divide by cost to make it cost-aware
    cost_safe = np.maximum(cost, 1e-9)
    
    # Draw samples from the GP posterior: y ~ N(mu, sigma^2)
    # For each candidate, we want the expected minimum over samples
    # of the cost-normalized objective value.
    
    # Precompute: for each sample, the objective value for each candidate
    # sample_values[s, i] = mu[i] + sigma[i] * z_s_i
    # where z_s_i ~ N(0, 1)
    
    # Generate random normals: shape (n_samples, n)
    z = np.random.standard_normal((n_samples, n))
    
    # sample_values: shape (n_samples, n)
    sample_values = mu[np.newaxis, :] + sigma[np.newaxis, :] * z
    
    # Cost-normalized objective: divide by cost
    # cost_norm_values: shape (n_samples, n)
    cost_norm_values = sample_values / cost_safe[np.newaxis, :]
    
    # For Thompson Sampling in minimization, we want to select the candidate
    # that has the lowest expected cost-normalized value across samples.
    # The score should be higher for candidates that are likely to be good.
    # 
    # Approach: For each candidate, compute the mean of the cost-normalized
    # values across samples. Lower mean = better. So score = -mean.
    # But we also want to balance exploration: candidates with high variance
    # in their cost-normalized values across samples are more uncertain.
    # 
    # Alternative: Use the expected minimum. For each candidate, the "value"
    # under Thompson Sampling is the sample that gives the minimum.
    # The expected score is the mean of the minimum across samples.
    # 
    # Actually, a cleaner approach: 
    # - For each sample s, the best candidate is argmin_i cost_norm_values[s, i]
    # - The Thompson Sampling score for candidate i is the fraction of samples
    #   in which i is the best, weighted by how much better it is.
    # 
    # Simpler and effective: score_i = -mean_s(cost_norm_values[s, i])
    # This is the expected cost-normalized objective, and we maximize -expected value.
    # But this is just the mean, which doesn't capture uncertainty well.
    
    # Better Thompson Sampling approach:
    # For each candidate, compute the expected value of the minimum
    # across a set of "virtual" evaluations. But that's complex.
    
    # Practical approach: 
    # score_i = -mean_s(cost_norm_values[s, i]) + alpha * std_s(cost_norm_values[s, i])
    # This is like UCB on the Thompson samples.
    
    # Even simpler and more principled:
    # The posterior mean of the objective for candidate i is mu[i]/cost[i]
    # The posterior std is sigma[i]/cost[i]
    # Thompson Sampling: draw one sample per candidate, pick the best.
    # But we're scoring all candidates, so we use the expected rank or
    # the probability of being best.
    
    # Let's use: score = -expected_cost_normalized_value + beta * std_cost_normalized_value
    # This is a UCB-like score on the cost-normalized posterior.
    
    mean_cn = np.mean(cost_norm_values, axis=0)  # shape (n,)
    std_cn = np.std(cost_norm_values, axis=0)    # shape (n,)
    
    # Adaptive beta: higher when f_best is large (early), lower when f_best is small (late)
    # f_best is a scalar (the best observed value)
    if np.isscalar(f_best) or np.ndim(f_best) == 0:
        f_best_val = float(f_best)
    else:
        f_best_val = float(np.min(f_best))
    
    # For Rastrigin, the minimum is 0. f_best starts high (~20+) and decreases.
    # beta should be high early, low late.
    C = 10.0
    beta = 1.0 * f_best_val / (f_best_val + C)
    
    # Score: we want higher score = better candidate
    # mean_cn is the expected cost-normalized value (lower is better)
    # std_cn is the uncertainty
    # score = -mean_cn + beta * std_cn
    score = -mean_cn + beta * std_cn
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
