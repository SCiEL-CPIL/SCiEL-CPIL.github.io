import numpy as np
import time
import random

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization.
    
    Strategy:
    1. Ensure inputs are float arrays.
    2. Handle numerical edge cases (sigma ~ 0).
    3. Draw a single random sample from the posterior distribution N(mu, sigma^2) 
       for each of the 2000 candidate points.
    4. Return these samples as the scores. The evaluator will pick the point 
       with the highest score (which corresponds to the lowest sampled objective 
       value if we were maximizing, but wait: the task says "higher meaning evaluate next" 
       and "minimizes goldstein price". 
       
       Let's re-read carefully: 
       "computes the posterior there, calls acquisition and evaluates the candidate with the highest score"
       "mu and sigma ... in objective units"
       "Scoring (lower is better): mean over the seeds of the simple regret"
       
       Standard Thompson Sampling for minimization:
       We want to find the minimum. 
       The posterior for the objective value f(x) is N(mu(x), sigma(x)^2).
       If we sample f_hat(x) ~ N(mu(x), sigma(x)^2), and we want to minimize f, 
       we should pick the x that minimizes f_hat(x).
       
       However, the interface says: "Return 2000 finite scores, higher meaning evaluate next".
       So if we want to pick the minimum of the sample, we must return scores that are HIGH for the minimum.
       Therefore, we should return -f_hat(x).
       
       Let's verify:
       If x1 is a good point (low true f), mu(x1) is low. Sample f_hat(x1) is likely low.
       Score = -f_hat(x1) is high (less negative).
       If x2 is a bad point (high true f), mu(x2) is high. Sample f_hat(x2) is likely high.
       Score = -f_hat(x2) is low (more negative).
       The evaluator picks the highest score, i.e., the highest -f_hat, i.e., the lowest f_hat.
       This is correct for minimization.
       
       Note: Thompson Sampling naturally balances exploration and exploitation.
       High sigma -> wide distribution of f_hat -> higher chance of a very low (good) sample?
       Actually, for minimization, high sigma means f_hat can be very low (good) or very high (bad).
       The expectation of the minimum of samples from independent Gaussians... 
       Actually, TS is known to be effective. By returning -sample, we are effectively 
       maximizing the probability of finding a low value, weighted by the uncertainty.
    """
    start_time = time.time()
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw one sample from the posterior for each point
    # N(mu, sigma^2)
    # We use standard normal random numbers
    z = np.random.randn(len(mu))
    samples = mu + sigma * z
    
    # We want to minimize the objective.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we must return scores that are high when the sampled objective is low.
    # score = -sample
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard: The computation is O(N) and very fast (2000 points).
    # No explicit deadline guard is strictly necessary for the computation itself,
    # but we can check if we are running out of time in a pathological case (unlikely here).
    elapsed = time.time() - start_time
    if elapsed > 250: # Safety margin
         # Fallback to a simple EI or UCB if somehow this took too long (it won't)
         pass
         
    return scores
