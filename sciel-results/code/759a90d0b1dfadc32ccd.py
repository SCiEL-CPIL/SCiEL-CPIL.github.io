"""
Cost-aware acquisition function for Bayesian optimization (Rastrigin 2D).
Strategy: Probabilistic Best-Point Selection with Cost-Adjusted Posterior.
Instead of a fixed-parameter LCB or EI, this approach computes the probability
that each candidate x is the global minimizer among all 2000 candidates, 
adjusted by the evaluation cost. 
Specifically, we model the joint distribution of the objective values at the 
2000 candidates given the GP posterior. Since exact joint sampling is expensive, 
we use a deterministic approximation based on the "Probability of Being the Best" 
(PBB) combined with a cost penalty.

For each candidate i, we estimate P(f(x_i) < min_{j != i} f(x_j) and f(x_i) < f_best) 
or more simply, we approximate the value of information by looking at the 
distribution of the minimum of the samples.

A more robust and computationally feasible "Different" strategy is **Cost-Adjusted 
Expected Improvement (CEI)** but computed via a **Gaussian Mixture Approximation** 
of the posterior minimum, or simply **Probability of Improvement (POI)** scaled 
by cost. However, POI is often too exploitative.

Let's use **Cost-Adjusted Probability of Being the Global Minimum**.
We approximate the probability that candidate i is the best among all candidates 
using the fact that if we sample f_i ~ N(mu_i, sigma_i^2), the probability that 
f_i is the minimum of all N samples is:
P_i = E[ 1_{f_i < min(f_j, j!=i)} ]

This is hard to compute exactly for N=2000.
However, a strong surrogate for "finding the global minimum" in multimodal 
functions like Rastrigin is to identify regions where the posterior mean is low 
AND the posterior variance is high enough to cover the true minimum, but not 
so high that it's pure noise.

Let's try a **Hybrid Score**: 
Score = ( -mu / cost ) * ( 1 + log(1 + sigma) )
This is a form of **Cost-Effective LCB** where the exploration term is logarithmic 
in sigma, making it less sensitive to extreme sigma values but still encouraging 
exploration. The division by cost makes it cost-aware.

Wait, the previous program used a linear sigma term with a dynamic beta. 
Let's try a **Non-linear** transformation of sigma that is more robust.
Also, let's incorporate a **Normalization** step to ensure the scores are 
comparable across different scales of mu and sigma, which can happen if the 
GP hasn't converged or if the function scale varies.

New Strategy: **Normalized Cost-Effective UCB with Log-Exploration**.
1. Normalize mu and sigma by their respective standard deviations (or range) to 
   create unitless "z-scores" of the posterior.
2. Compute score = ( -z_mu + kappa * z_sigma ) / cost.
3. Use a fixed kappa that is slightly higher than 1 to encourage exploration 
   in the early stages, but since we don't know the stage, we can use a 
   **Self-Adaptive** kappa based on the correlation between mu and sigma.
   Actually, simpler: Use a **fixed** kappa but with a **log-sigma** term.
   
   Score = ( -mu + kappa * log(1 + sigma) ) / cost
   
   This prevents candidates with very high sigma (pure noise) from dominating 
   the acquisition, which is a common failure mode of linear LCB in multimodal 
   functions. It encourages exploring areas with "moderate" uncertainty and 
   low mean.

Let's refine:
Score = ( -mu + kappa * sigma / (1 + sigma) ) / cost
This is a bounded exploration term.

Actually, the most effective "different" strategy for Rastrigin (multimodal) 
is often **Expected Improvement (EI)** because it naturally focuses on areas 
where the probability of improvement is high, which corresponds to the 
"valleys" of the posterior. However, EI is not cost-aware. 
**Cost-Effective EI (CEI)** is:
CEI(x) = EI(x) / cost(x)

Let's implement **Cost-Effective Expected Improvement (CEI)**.
EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
where Phi is the CDF and phi is the PDF of the standard normal distribution.

This is a fundamentally different acquisition function from LCB. 
EI is known to be more robust for multimodal functions in some contexts 
because it directly measures the expected gain.

We will compute EI for all 2000 candidates and divide by cost.
"""

import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Computes acquisition scores for 2000 candidates.
    Uses Cost-Effective Expected Improvement (CEI).
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure arrays are numpy
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Basic sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    # Ensure cost is strictly positive
    cost = np.maximum(cost, 1e-8)

    # f_best is a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # Check time
    if time.time() > deadline - 5.0:
        # Fallback: simple LCB
        return (1.5 * sigma - mu) / cost

    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Optimization: If f_best is very large compared to mu, EI is large.
    # If mu is very low, EI is small.
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    # norm.cdf(-10) is ~0, norm.cdf(10) is ~1
    # norm.pdf(10) is ~0
    # We can safely clip to [-10, 10]
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Compute Phi and phi
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    # EI formula
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Effective EI
    cei = ei / cost
    
    # Ensure finite
    cei = np.nan_to_num(cei, nan=0.0, posinf=1e6, neginf=0.0)
    
    return cei
