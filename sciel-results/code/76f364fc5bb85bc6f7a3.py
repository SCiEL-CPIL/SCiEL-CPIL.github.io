"""
Cost-aware acquisition function for Bayesian optimization (Rastrigin 2D).
Strategy: Cost-Aware Thompson Sampling.

Unlike deterministic acquisition functions (EI, UCB) which use point estimates
(mean and variance) to rank candidates, Thompson Sampling samples from the
posterior predictive distribution and selects the candidate with the best
sampled value. This naturally balances exploration and exploitation without
requiring tuning of hyperparameters like kappa or beta.

We adapt this for cost-awareness by normalizing the sampled values by the
evaluation cost. Specifically, we compute the expected minimum of the
posterior samples divided by the cost. To approximate the expected minimum
of a Gaussian distribution N(mu, sigma^2), we use the formula:
E[min] ≈ mu - sigma * E[min(Z)] where Z ~ N(0,1).
Since we are minimizing, a lower sampled value is better.
The score for Thompson Sampling is typically the negative of the sampled value
(or the sampled value itself if we want the minimum).
However, for cost-awareness, we want to maximize the "improvement per cost".
A robust proxy is to use the posterior median (or a quantile) adjusted for cost.

A simpler and very effective variant for cost-aware BO is to use the
Posterior Expected Improvement (PEI) or simply rank by:
score = - (mu - quantile * sigma) / cost
where quantile is chosen to reflect exploration.
Actually, standard Thompson Sampling for minimization picks the argmin of
samples y_i ~ N(mu_i, sigma_i^2).
We can approximate the expected value of the minimum of a single sample
(which is just the sample itself in expectation) but we need a deterministic
score for the 2000 candidates.
The expected value of a sample from N(mu, sigma^2) is mu. That's just the mean.
To get the exploration benefit, we can look at the distribution of the
"best" sample among many, or simply use a lower quantile.
Using the 10th percentile (or similar) of the posterior predictive
distribution as the "sampled" value is a common deterministic approximation
of Thompson Sampling (sometimes called "Quantile Sampling" or "LCB with
probabilistic calibration").

Let's use the posterior mean minus a dynamic number of standard deviations,
but derived from the probability of being the minimum.
Actually, the most direct structural change from EI/UCB is to use
**Expected Feasible Improvement (EFI)** or **Posterior Expected Improvement**
is still EI.

Let's try **Cost-Effective Thompson Sampling** via the **Posterior Probability of Being the Minimum** (PPBM) or similar.
But a simpler, distinct family is **Knowledge Gradient (KG)**.
KG estimates the expected improvement in the *knowledge* of the global minimum
after evaluating a point.
KG(x) = E[ max(0, f_best - min(f_best, f(x))) ]
This is structurally different from EI because it considers the *minimum*
of the current best and the new sample, not just the improvement of the new
sample over the current best.
For a Gaussian f(x) ~ N(mu, sigma^2):
KG(x) = E[ max(0, f_best - min(f_best, f(x))) ]
      = E[ max(0, f_best - f(x)) * I(f(x) < f_best) ]  -- Wait, min(f_best, f(x)) is f(x) if f(x)<f_best, else f_best.
      If f(x) < f_best, min is f(x), so f_best - min = f_best - f(x).
      If f(x) >= f_best, min is f_best, so f_best - min = 0.
      So KG(x) = E[ (f_best - f(x)) * I(f(x) < f_best) ]
      This is exactly the definition of Expected Improvement (EI)!
      Wait, EI is E[ max(0, f_best - f(x)) ].
      Yes, for a single new point, KG is equivalent to EI.
      
So KG is not a structural change in 1D step.

Let's look at **Information Gain** or **Entropy Reduction**.
But cost-awareness is key.

Let's try **Probability of Improvement (PoI)** divided by cost.
PoI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
Score = PoI(x) / cost(x).
This is a very different family from EI (which weights by magnitude of improvement)
and UCB (which weights by uncertainty magnitude). PoI focuses purely on the
*chance* of finding a better point, regardless of how much better.
This can be more robust in noisy or complex landscapes where large improvements
are rare but small improvements are frequent.

Let's implement **Cost-Adjusted Probability of Improvement (CA-PoI)**.
"""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI).
    
    Strategy:
    1. Compute the probability that each candidate x has a value lower than f_best.
       P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    2. Normalize this probability by the evaluation cost to prefer cheap points
       that have a high chance of improvement.
    3. Add a small exploration bonus based on uncertainty to prevent getting
       stuck in low-uncertainty, low-probability regions early on.
       However, pure PoI/cost can be too greedy. 
       
    To make it structurally distinct from the current EI/UCB blend, we will
    use a **Hybrid of PoI and Uncertainty** but with a different weighting
    mechanism: **Softmax Selection Probability**.
    
    Actually, let's stick to a clean **Cost-Effective PoI** with a dynamic
    temperature or scaling to ensure exploration.
    
    Score = (PoI(x) + epsilon * sigma(x)) / cost(x)
    where epsilon controls the exploration-exploitation trade-off.
    Since we don't have a feedback loop to tune epsilon, we can use a
    heuristic: epsilon = 1 / (1 + f_best) or similar?
    
    Let's try a different structural approach: **Bayesian Optimization with
    Cost-Effective Global Search (CEGS) using PoI**.
    
    The current program uses EI and UCB. PoI is a fundamentally different
    acquisition function family.
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin before 330s kill

    # Ensure arrays are numpy
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Basic sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # f_best is a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # 1. Probability of Improvement (PoI)
    # z = (f_best - mu) / sigma
    # PoI = Phi(z)
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # 2. Cost-Adjusted PoI
    # We want to maximize the probability of improvement per unit cost.
    score_poi_cost = poi / cost
    
    # 3. To ensure exploration, we add a term proportional to uncertainty.
    # However, pure PoI can be very greedy. 
    # Let's blend PoI/cost with a normalized uncertainty term.
    # Uncertainty term: sigma / cost
    # We need to scale them to comparable ranges.
    # PoI is in [0, 1]. sigma can be large.
    
    # Let's use a dynamic weight for exploration based on the mean PoI.
    # If mean PoI is low, we need more exploration.
    mean_poi = np.mean(poi)
    
    # If mean_poi is very low, increase exploration weight.
    # exploration_weight = 1 / (1 + mean_poi * 10)
    # This is a heuristic.
    
    # Alternative: Use **Expected Improvement** but with a different
    # cost model? No, that's not a structural change.
    
    # Let's try **Posterior Expected Improvement** is EI.
    
    # What about **Information Gain**?
    # IG(x) = H(posterior) - E[H(posterior | x)]
    # This is hard to compute analytically for GP.
    
    # Let's stick to **PoI + Exploration** but implement it robustly.
    # We will normalize the PoI and Sigma terms separately.
    
    # Normalize PoI to [0, 1] (it already is)
    # Normalize Sigma: min-max scaling to [0, 1]
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    if sigma_max > sigma_min:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
        
    # Cost normalization? No, we divide by cost.
    
    # Blend: alpha * (PoI / cost) + (1 - alpha) * (Sigma / cost)
    # But PoI/cost and Sigma/cost have different scales.
    # Let's scale them.
    max_poi_cost = np.max(poi / cost)
    max_sigma_cost = np.max(sigma / cost)
    
    if max_poi_cost > 0:
        poi_scaled = (poi / cost) / max_poi_cost
    else:
        poi_scaled = np.zeros_like(poi)
        
    if max_sigma_cost > 0:
        sigma_scaled = (sigma / cost) / max_sigma_cost
    else:
        sigma_scaled = np.zeros_like(sigma)
        
    # Dynamic alpha: 
    # Early on (high sigma, low f_best improvement), alpha should be lower (more exploration).
    # Later on (low sigma, high f_best improvement), alpha should be higher (more exploitation).
    
    # Use mean sigma as a proxy for "early" vs "late".
    # High mean sigma -> early -> explore more (lower alpha).
    # Low mean sigma -> late -> exploit more (higher alpha).
    
    mean_sigma = np.mean(sigma)
    # Normalize mean_sigma to [0, 1] roughly. Rastrigin values are ~0-60.
    # Sigma in objective units.
    # Let's use a sigmoid function to map mean_sigma to alpha.
    # If mean_sigma is large, alpha -> 0.2. If small, alpha -> 0.8.
    # alpha = 0.5 + 0.3 * np.tanh(-mean_sigma / 5.0)
    
    alpha = 0.5
    # If f_best is very high (bad), we are early. 
    # If f_best is close to 0, we are late.
    # Let's use f_best to adjust alpha.
    # Rastrigin min is 0. Max is ~60.
    # If f_best > 30, alpha = 0.3 (explore).
    # If f_best < 1, alpha = 0.8 (exploit).
    
    if f_best > 30:
        alpha = 0.3
    elif f_best > 10:
        alpha = 0.5
    elif f_best > 1:
        alpha = 0.7
    else:
        alpha = 0.9
        
    score = alpha * poi_scaled + (1 - alpha) * sigma_scaled
    
    # The final score should be higher for better candidates.
    # We returned scaled scores in [0, 1].
    # We can return them directly, or scale them back to a larger range
    # if the evaluator expects large values? 
    # The task says "Return 2000 finite scores (higher = evaluate next)".
    # It doesn't specify the range. 
    # However, if the scores are all very small, it might not matter.
    # But to be safe, let's multiply by a constant or return the raw 
    # (PoI + Sigma) / cost.
    
    # Let's return the raw blended score, not normalized, to preserve
    # the relative importance of cost.
    raw_score = alpha * (poi / cost) + (1 - alpha) * (sigma / cost)
    
    if np.all(np.isfinite(raw_score)):
        return raw_score
    else:
        return score_poi_cost

# EVOLVE-BLOCK-END
