import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Stochastic Thompson Sampling.
    
    Strategy:
    1. Treat the 2000 candidates as arms in a multi-armed bandit problem.
    2. For each candidate, we have a posterior distribution N(mu_i, sigma_i^2).
    3. We sample one value y_i from each distribution.
    4. We calculate the realized improvement: imp_i = max(0, f_best - y_i).
       This is the "regret reduction" if we were to evaluate x_i and it turned out to be y_i.
       Note: Since we are minimizing, improvement is f_best - y.
    5. We divide this improvement by the cost_i to get a cost-adjusted reward.
    6. We select the candidate with the highest realized cost-adjusted reward.
    7. To make the return interface work (which expects a vector of scores where the max is evaluated),
       we return the scores such that the selected index has the highest value.
       However, a single sample is very noisy. To stabilize the selection while retaining
       the stochastic benefits of TS, we can use a "soft" selection or simply return the 
       sampled values adjusted by cost.
       
       Actually, standard TS returns the index of the argmax of the samples.
       The interface requires returning 2000 scores. The evaluator picks argmax(scores).
       If we return the sampled values divided by cost, the argmax is the TS selection.
       
       To improve robustness against a single bad random draw (which might pick a point 
       with high cost and low true improvement just by chance), we can use a 
       "Temperature" or "Softmax" approach, or simply rely on the fact that 
       2000 samples is a decent pool.
       
       Let's stick to pure TS but ensure numerical stability.
       Score_i = (f_best - sample_i) / cost_i
       We want to maximize this.
       If sample_i > f_best, the improvement is negative (or 0 if we clip).
       In minimization, if y > f_best, it's not an improvement. 
       However, in TS for minimization, we usually sample y and pick min(y/cost) or similar?
       No, TS for minimization: Sample y ~ N(mu, sigma^2). 
       We want to pick the x that minimizes the expected cost-adjusted function value?
       Or maximizes expected improvement?
       
       Let's look at the "Cost-aware" aspect.
       Reward = Improvement / Cost.
       Improvement = max(0, f_best - y).
       
       If we just return (f_best - sample) / cost, we might pick a point where the sample 
       is very low (good) but cost is high.
       
       Alternative framing: Minimize (sample * cost).
       This is equivalent to maximizing -sample * cost? No.
       
       Let's use the Expected Improvement analogy but with samples.
       Score = (f_best - sample) / cost.
       
       This is a valid TS variant for cost-aware optimization.
       
       To handle the "finite scores" and "higher = better" requirement:
       We compute scores = (f_best - samples) / cost.
       
       One issue: If f_best is very low, and samples are high, scores are negative.
       This is fine, as long as we pick the max.
       
       To add a bit of exploration stability, we can add a small epsilon to the denominator 
       or use a temperature scaling if the variance of scores is too low.
       
       Let's implement this.
    """
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Fallback to a safe UCB if data is corrupt
        sigma_safe = np.maximum(sigma, 1e-6)
        scores = (-mu + 2.0 * sigma_safe) / np.maximum(cost, 1e-6)
        return scores
        
    # Clip sigma to avoid zero variance issues in sampling
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Clip cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: Draw one sample from each posterior
    # y_i ~ N(mu_i, sigma_i^2)
    # Using np.random.normal
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Calculate Cost-Adjusted Improvement
    # We want to maximize: (f_best - y) / cost
    # If y < f_best, this is positive (good).
    # If y > f_best, this is negative (bad).
    # This effectively penalizes points that are likely worse than current best,
    # and rewards points that are likely better, scaled by cost.
    
    scores = (f_best - samples) / cost_safe
    
    # Ensure finite output
    scores[~np.isfinite(scores)] = -1e9
    
    return scores
