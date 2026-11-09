import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function for cost-aware Bayesian Optimization.
    
    Instead of maximizing Expected Improvement (EI) per unit cost, we sample from
    the posterior distribution of the objective function at each candidate point.
    We then select the candidate that minimizes the sampled value, adjusted by a
    penalty for evaluation cost.
    
    Score = - (sampled_value + lambda * cost)
    Higher score means more likely to be selected.
    
    Why this is different:
    - EI is deterministic given (mu, sigma). It exploits the mean improvement.
    - Thompson Sampling introduces stochasticity (via sampling) which encourages
      exploration of regions with high uncertainty (high sigma) more effectively
      in some regimes, or at least diversifies the search path compared to the
      greedy EI/UCB selection which can get stuck in local optima or over-exploit
      a single promising region.
    - The cost is incorporated directly into the sampled value, modeling the
      "effective value" of the point including the cost of finding it.
    """
    
    # Handle edge cases
    if sigma is None or len(sigma) == 0:
        return np.zeros(len(mu))
        
    # Ensure sigma is positive to avoid log issues or invalid sampling
    # In GP, sigma should be > 0, but numerical errors might make it 0 or negative.
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Sample from the posterior: N(mu, sigma^2)
    # We use a fixed random seed? No, the function should be stateless or use global RNG.
    # Using np.random.rand is standard. However, for reproducibility in tests, 
    # sometimes fixed seeds are preferred, but in BO, the stochasticity is the feature.
    # The prompt says "five seeded instances", implying the evaluator runs the whole loop 5 times.
    # The acquisition function itself is called many times.
    # If we use a global RNG, the results are stochastic.
    # To make it more robust and less dependent on the specific random draw (which can be noisy),
    # we can use a deterministic approximation of Thompson Sampling:
    # Sample multiple times and take the mean? Or just one sample?
    # One sample is the standard TS.
    
    # However, a pure single sample can be very noisy. 
    # A common deterministic alternative that captures the TS spirit is 
    # maximizing the probability of being the minimum, or using a 
    # "Soft" Thompson Sampling where we average over a few samples.
    
    # Let's try a single sample first, but with a cost penalty.
    # Cost is in [1, 2]. 
    # f_best is the current best value.
    # The sampled value `s` is an estimate of f(x).
    # We want to minimize `f(x) + cost`.
    # So we maximize `-(f(x) + cost)`.
    
    # To reduce variance of the TS score, let's average over a small number of samples (e.g., 10).
    # This is a "Monte Carlo TS" approximation.
    # N=10 samples is fast for 2000 points.
    
    n_samples = 10
    # Generate random normal numbers: (n_samples, 2000)
    # Note: np.random.normal with size (n_samples, len(mu))
    # This is memory efficient enough (10 * 2000 * 8 bytes = 160KB)
    
    # Use a local random state to avoid interfering with global state, 
    # or just use np.random. Given the context, np.random is fine.
    
    # Actually, to be safe against any potential state issues and to ensure 
    # the "stochasticity" is consistent across the 5 instances if they share 
    # the same RNG state (which they likely don't, as they are separate runs),
    # we just use np.random.
    
    samples = np.random.normal(loc=mu, scale=sigma_safe, size=(n_samples, len(mu)))
    
    # The cost penalty:
    # We want to penalize high cost.
    # The cost is in [1, 2].
    # The objective values for Rosenbrock can be large (e.g., 10, 100, 1000).
    # A linear penalty `lambda * cost` needs lambda to be scaled appropriately.
    # If f_best is ~10, a cost difference of 1 (from 1 to 2) is small.
    # If f_best is ~1, it's significant.
    # A common heuristic is to normalize cost relative to the current scale of f.
    # Let's use lambda = 0.1 * f_best? Or a fixed small lambda?
    # If lambda is too large, we always pick the cheapest point (x near 0,0 in scaled coords).
    # If lambda is too small, it's like standard TS.
    
    # Let's look at the cost definition: `1 + mean(x_scaled)`.
    # x_scaled is in [0, 1].
    # The minimizer of Rosenbrock is at (1,1) in original [-2,2] coords?
    # Wait, Rosenbrock min is usually at (1,1) for f(x,y)=(1-x)^2+100(y-x^2)^2.
    # The domain is [-2, 2]^2.
    # Scaled to [0, 1]: x=1 -> (1-(-2))/4 = 0.75. y=1 -> 0.75.
    # So the optimal point has cost `1 + (0.75+0.75)/2 = 1.75`.
    # The cheapest point is x=0,y=0 -> scaled 0,0 -> cost 1.0.
    # The most expensive is x=2,y=2 -> scaled 1,1 -> cost 2.0.
    
    # The difference in cost is at most 1.0.
    # The difference in f values can be huge.
    # Therefore, the cost penalty should be small relative to f.
    # Let's set lambda = 0.05. This means a cost difference of 1.0 is equivalent to an f difference of 0.05.
    # This is a mild preference for cheaper points.
    
    lambda_cost = 0.05
    
    # Compute the score for each sample: - (sample + lambda * cost)
    # We want to maximize this.
    # scores shape: (n_samples, 2000)
    scores = -(samples + lambda_cost * cost[None, :])
    
    # For each candidate, take the mean of the scores over the samples?
    # Or take the max?
    # Standard TS picks the argmax of a *single* sample.
    # If we return the mean of scores, we are doing "Expected TS" or "Soft TS".
    # E[-(s + c)] = -mu - lambda*cost. This is just Linear UCB with a weird sign!
    # -mu - lambda*cost. Maximizing this is minimizing mu + lambda*cost.
    # This is very exploitative. It ignores sigma entirely in the mean.
    
    # To capture the exploration aspect of TS, we should NOT average the samples.
    # We should pick the candidate that is best in a *specific* sample.
    # But we must return a score for ALL 2000 candidates to let the evaluator pick the argmax.
    # If we return a score that varies randomly each call, the evaluator picks the argmax of that random vector.
    # This is exactly how TS works in a loop:
    # 1. Sample f(x) for all x.
    # 2. Pick x* = argmin f_sample(x).
    # 3. Evaluate x*.
    # 
    # How do we implement this with a function that returns 2000 scores?
    # We generate one set of samples.
    # We compute the score for each candidate as:
    # Score_i = - sample_i - lambda * cost_i
    # The evaluator will pick argmax(Score_i), which is argmin(sample_i + lambda*cost_i).
    # This is exactly one step of Thompson Sampling.
    
    # So, we just need to return the scores from a SINGLE sample.
    # The stochasticity is handled by the fact that `samples` is random each call.
    
    # Let's use a single sample to be true to TS.
    # However, a single sample can be very noisy. 
    # Is there a way to stabilize it?
    # We could use the median of a few samples? No, that changes the distribution.
    # Let's stick to 1 sample for pure TS, but maybe add a tiny bit of regularization?
    # No, pure TS is the structural change.
    
    sample = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Score to maximize: -(sample + lambda * cost)
    final_scores = -(sample + lambda_cost * cost)
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
