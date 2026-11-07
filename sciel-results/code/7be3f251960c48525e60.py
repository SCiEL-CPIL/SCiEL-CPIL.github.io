"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Approximate Knowledge Gradient acquisition function.
    
    Args:
        mu: array of shape (n,), posterior mean
        sigma: array of shape (n,), posterior std
        f_best: float, best observed value so far
    
    Returns:
        scores: array of shape (n,), higher is better
    """
    n = len(mu)
    
    # Safety check for time
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple EI if running out of time
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Number of MC samples for KG approximation
    # 50 samples is a good balance between accuracy and speed for 2000 points
    n_samples = 50
    
    # Generate standard normal samples
    # Shape: (n_samples, n)
    Z = np.random.standard_normal((n_samples, n))
    
    # Simulate true values y ~ N(mu, sigma^2)
    # y[s, i] = mu[i] + sigma[i] * Z[s, i]
    y = mu[None, :] + sigma_safe[None, :] * Z
    
    # For each candidate i, the Knowledge Gradient is the expected improvement
    # in the best value found after evaluating i.
    # KG_i = E[ max(0, f_best - min(f_best, y_i)) ]
    #      = E[ max(0, f_best - y_i) ]  (since y_i is the new value, if it's better than f_best, improvement is f_best - y_i)
    # Wait, standard KG for minimization:
    # If we evaluate point i, the new best is min(f_best, y_i).
    # The value of the information is the expected reduction in the best value.
    # Reduction = f_best - min(f_best, y_i) = max(0, f_best - y_i).
    # So KG_i is simply the Expected Improvement (EI) of point i?
    # No, that's not right. EI is max(0, f_best - y_i).
    # KG accounts for the fact that evaluating i might not be the global minimum, but it updates our belief.
    # However, in a "one-step look-ahead" KG where we only care about the immediate best value, 
    # and assuming we only get one sample, the expected reduction in the *current* best is indeed EI.
    
    # BUT, the true KG considers that evaluating point i gives information about the *global* minimum.
    # A more robust approximation often used is:
    # KG_i ≈ E[ max_{j} ( -y_j ) ] - max_{j} ( -mu_j ) ... this is complex.
    
    # Let's stick to a more practical and effective alternative: **Upper Confidence Bound (UCB)**
    # or **Expected Improvement with a tuned beta**.
    # Actually, let's look at why EI might be suboptimal here.
    # EI is greedy. UCB is more exploratory.
    # Let's try a hybrid: **Expected Improvement with a penalty for being close to already evaluated regions?**
    # We don't have coordinates.
    
    # Let's try **Probability of Improvement (PI)**? No, too noisy.
    
    # Let's try **Knowledge Gradient** properly.
    # The KG value for candidate i is:
    # V_i = E[ max_{j in candidates} ( -y_j ) ] - max_{j in candidates} ( -mu_j )
    # This is hard to compute directly for 2000 points.
    
    # A common practical approximation for KG in BO is to use the **expected maximum** of the posterior.
    # However, a simpler and often more effective strategy for 1D problems with limited evaluations is
    # to use **UCB (Upper Confidence Bound)** with a time-varying or fixed exploration parameter.
    
    # Let's switch strategy to **UCB** (Upper Confidence Bound for Minimization, so we look for low values).
    # For minimization, we want to select points with low mu and high uncertainty.
    # Score = mu - beta * sigma
    # We want to MINIMIZE this score? No, the function returns scores where HIGHER is better.
    # So we want high scores for points that are likely to be low.
    # So Score = -mu + beta * sigma.
    
    # Choosing beta is critical.
    # A common choice is beta = 2 * sqrt(log(t)) or similar, but we don't have t directly, 
    # although we can infer it from f_best or just use a heuristic.
    # Since we have 30 evaluations total, and start with 5, we have 25 steps.
    # Early on, we want high exploration. Later, exploitation.
    # Without explicit t, we can estimate progress based on f_best improvement? No, that's risky.
    # Let's use a fixed beta that balances exploration well for 1D.
    # beta = 2.0 is a common default.
    
    # However, there is another powerful acquisition function: **Expected Improvement (EI)**
    # is actually quite strong. The current program uses EI and gets 1.0.
    # Inspiration 1 (score 1.0) used a "SEARCH-style portfolio".
    # Inspiration 2 (score 0.02) used mean minimization (bad).
    
    # Let's try **PI (Probability of Improvement)**? No.
    # Let's try **KG approximation via MC** as initially planned, but correctly.
    
    # Correct 1-step KG for minimization:
    # We want to maximize the expected reduction in the global minimum value.
    # Let Y be the vector of true values for all 2000 candidates.
    # Y ~ N(mu, diag(sigma^2)).
    # Current best is f_best.
    # If we evaluate candidate i, we observe y_i.
    # The new global minimum (among evaluated points) will be min(f_best, y_i).
    # BUT, the KG also considers that this observation updates the posterior for other points.
    # For a single step, the most common approximation is:
    # KG_i = E[ max(0, f_best - y_i) ]  <-- This is just EI.
    
    # Why is EI not beating 1.0? Maybe it's not bad, maybe 1.0 is the baseline.
    # To beat 1.0, we need a better balance.
    
    # Let's try **Thompson Sampling**.
    # Sample y_i ~ N(mu_i, sigma_i^2) for all i.
    # Return the index of the minimum y_i?
    # No, we must return scores for ALL 2000 points.
    # The evaluator picks the argmax of the scores.
    # So if we use Thompson Sampling, we should return the sampled values y_i?
    # If we return y_i, the evaluator picks the smallest y_i (since higher score is better, we should return -y_i).
    # So: Sample y ~ N(mu, sigma^2). Return -y.
    # The evaluator will pick the point with the lowest sampled y.
    # This is a valid acquisition strategy (Thompson Sampling).
    
    # Let's implement Thompson Sampling.
    # It naturally balances exploration and exploitation.
    # High uncertainty (sigma) leads to wide distribution of y, increasing chance of a very low y.
    # Low mean (mu) leads to low y.
    
    # Implementation:
    # 1. Generate random normal Z of shape (2000,)
    # 2. y = mu + sigma * Z
    # 3. return -y
    
    # This is extremely fast and robust.
    
    # Let's verify the logic.
    # If point A has mu=0, sigma=1.
    # If point B has mu=10, sigma=1.
    # A is more likely to have low y.
    # If point C has mu=5, sigma=100.
    # C has a high chance of having a very low y (e.g., -95), but also high chance of high y.
    # Thompson Sampling will pick C often enough to explore, but A will be picked when C's sample is high.
    # This is ideal for global optimization.
    
    # Let's add a small safety check.
    Z = np.random.standard_normal(n)
    y = mu + sigma_safe * Z
    scores = -y
    
    # Ensure finite
    scores[~np.isfinite(scores)] = -1e9
    
    return scores
# EVOLVE-BLOCK-END
