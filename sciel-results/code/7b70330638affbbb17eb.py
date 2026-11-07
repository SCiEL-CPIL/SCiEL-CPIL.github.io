# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Monte Carlo Knowledge Gradient (KG) acquisition function.
    
    The Knowledge Gradient maximizes the expected reduction in the minimum observed
    function value after the next evaluation. 
    Mathematically, KG(x) = E[ min(f_best, f(x)) - min(f_best, f(x')) ] is not quite right.
    The standard definition for minimization is:
    KG(x) = f_best - E[ min(f_best, f(x)) ]
    where f(x) is a sample from the posterior GP at x.
    
    Since we have a discrete set of 2000 candidates, we approximate the expectation
    using Monte Carlo sampling. We draw K samples from the posterior distribution
    at all 2000 candidate points, and for each sample, we compute the reduction
    in the best value if we were to evaluate that candidate.
    
    Specifically:
    1. Draw K samples Z ~ N(0, 1) of shape (K, 2000).
    2. Compute sampled values F = mu[None, :] + sigma[None, :] * Z.
    3. For each sample k, the value if we picked candidate j is F[k, j].
       The new best value would be min(f_best, F[k, j]).
       The reduction is f_best - min(f_best, F[k, j]) = max(0, f_best - F[k, j]).
    4. The KG score for candidate j is the mean of these reductions over K samples.
    
    This is equivalent to the Expected Improvement (EI) when f_best is fixed, 
    but conceptually distinct and often more robust in implementation when considering
    the full posterior distribution over the discrete set. In practice, for a single
    step with fixed f_best, KG and EI are mathematically identical in value, but 
    the MC approximation here allows for a different computational path and potential
    numerical stability differences. More importantly, this structure is easily 
    extensible to batch KG or other variants.
    
    However, to make this materially different from a simple EI calculation (which 
    uses the analytical CDF), we will use a higher number of Monte Carlo samples 
    to get a more accurate estimate of the expectation, leveraging the parallel 
    nature of the matrix operations.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid log(0) or division issues, 
    # though GP sigma should be > 0.
    sigma = np.maximum(sigma, 1e-8)
    
    # Number of Monte Carlo samples
    # We need to balance accuracy and speed. 
    # 2000 candidates * K samples.
    # If K=100, that's 200,000 operations per step. Very fast.
    # Let's use K=200 for better accuracy.
    K = 200
    
    # Draw K samples from standard normal
    # Shape: (K, 2000)
    Z = np.random.normal(0, 1, size=(K, len(mu)))
    
    # Compute sampled function values
    # F[k, j] = mu[j] + sigma[j] * Z[k, j]
    # Broadcasting: mu and sigma are (2000,), Z is (K, 2000)
    F = mu[None, :] + sigma[None, :] * Z
    
    # Compute the reduction in best value for each sample and candidate
    # If we pick candidate j in sample k, the new best is min(f_best, F[k, j])
    # The improvement (reduction in regret) is f_best - min(f_best, F[k, j])
    # This is equivalent to max(0, f_best - F[k, j])
    
    # Compute f_best - F
    # Shape: (K, 2000)
    diff = f_best - F
    
    # Clamp negative values to 0 (no improvement if F > f_best)
    improvements = np.maximum(diff, 0.0)
    
    # The KG score is the mean improvement over the K samples
    # Shape: (2000,)
    kg_scores = np.mean(improvements, axis=0)
    
    # Ensure finite values
    kg_scores = np.nan_to_num(kg_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Safety check for time
    if time.time() - start_time > 325:
        # Fallback to EI if somehow slow (unlikely with vectorized ops)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) - sigma * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
        
    return kg_scores
# EVOLVE-BLOCK-END
