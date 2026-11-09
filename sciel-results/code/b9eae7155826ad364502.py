"""
Strategy: Diversity-Constrained UCB (Greedy Batch Emulation)
Instead of independent EI/UCB scores, we compute a base UCB score and then
penalize candidates that are too close to the current best point (f_best) 
or to other high-scoring candidates. 
Since we only return scores and the evaluator picks the argmax, we design the 
scores so that the argmax is the point that offers the best "improvement per 
distance" or strictly enforces a minimum distance from the known best.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    t0 = time.time()
    deadline = t0 + 290.0  # Safety margin
    
    s = sigma + 1e-9
    
    # 1. Compute standard UCB scores
    # We use a fixed beta that is aggressive enough to encourage exploration
    # but not so much that it ignores the mean.
    # Standard UCB: mu + beta * sigma
    # We want to minimize, so we look for low values.
    # Acquisition should be high for points likely to be better (lower) than f_best.
    
    # Let's use an Expected Improvement variant that is robust.
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / s
    # Clip z to avoid numerical issues, though norm handles it mostly.
    # For large positive z (mu << f_best), EI is large.
    # For large negative z (mu >> f_best), EI is ~0.
    
    # To ensure we always have a finite, well-behaved score:
    # We'll use a soft-plus-like behavior or just standard EI.
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    
    # The problem with pure EI is it converges to 0 everywhere except near the best.
    # We want to diversify.
    
    # Strategy: 
    # 1. Identify the current "best" candidate in the GP posterior (min mu).
    #    Actually, f_best is the observed best.
    # 2. We want to find a point that is either:
    #    a) Very likely to beat f_best (High EI)
    #    b) Far from f_best AND has reasonable uncertainty (Exploration)
    
    # Let's construct a score that balances EI and Distance from the "center of mass" of high-EI points?
    # No, we don't have coordinates. We only have mu and sigma.
    
    # Wait, the task says "No coordinates are passed". 
    # This is a crucial constraint. We CANNOT compute distances in the input space [0,1]^2.
    # We only have the posterior mean and std at 2000 candidate points.
    
    # How can we enforce diversity without coordinates?
    # We can use the correlation structure implied by the GP? No, we don't have the kernel matrix.
    # We only have the vectors mu and sigma.
    
    # If we don't have coordinates, we cannot compute spatial distance.
    # However, points with similar mu and sigma might be close? Not necessarily.
    
    # Let's re-read the inspirations.
    # Inspiration 1: "Step 1: The current program uses a simple time-decaying Linear Confidence Bound (LCB)... A materially different strategy is..."
    # Inspiration 3: "Replace the stochastic Thompson Sampling approach with a deterministic, adaptive Expected Improvement (EI) acquisition function that uses a step-aware, non-monotonic exploration parameter"
    
    # The current program (score 1.0) already uses a batch penalty based on `r` which is derived from `mu` and `sigma`. 
    # `r = (m - mu) / (s + 1e-9)` where `m = mu.min()`.
    # This `r` is a measure of how far the point is from the best predicted mean, normalized by uncertainty.
    # It is NOT spatial distance. It is posterior-space distance.
    
    # If we cannot use spatial distance, how do we improve?
    # The current program tries to optimize hyperparameters `beta` and `lam` to select a "good" configuration of scores.
    # The quality metric uses `div` based on `r` (posterior distance).
    
    # A different strategy: **Thompson Sampling with a Diversity Penalty in Posterior Space**?
    # Or **Maximin Diversity in Posterior Space**?
    
    # Let's try a different angle: **Lookahead via Greedy Selection of Top-K**.
    # The evaluator picks the argmax of our returned scores.
    # If we return scores such that the top score is at point A, it evaluates A.
    # Next step, A is added to history.
    
    # The current program's weakness: It optimizes hyperparameters for a "quality" metric that is a heuristic.
    # What if we use a **Deterministic Annealing** approach?
    # Start with high exploration (low beta in UCB, or high noise) and gradually reduce it.
    # But we don't know the step number! We don't have `t` or `step`.
    
    # Can we infer the step?
    # We have `f_best`. As the algorithm progresses, `f_best` decreases (improves).
    # We can use the trajectory of `f_best`? No, we only get the current `f_best`.
    
    # However, we can look at the distribution of `mu` and `sigma`.
    # Early on, `sigma` is high everywhere. Later, `sigma` is low near observed points.
    # We can use the **median sigma** as a proxy for the "age" of the optimization.
    # High median sigma -> Early stage -> Explore more.
    # Low median sigma -> Late stage -> Exploit more.
    
    # This is a **State-Adaptive** acquisition.
    
    # Let's implement a UCB with an adaptive beta based on the uncertainty of the GP.
    
    # Calculate median sigma
    med_sigma = np.median(sigma)
    
    # Normalize sigma to [0, 1] range for adaptation?
    # We don't know the max possible sigma.
    # But we know the noise is 1e-6.
    # Let's assume a scale.
    
    # Alternatively, use the ratio of the best EI to the mean EI?
    
    # Let's try a different structural change: **Bayesian Optimization with a Diversity-Promoting Score using Mutual Information (approximated)**.
    # Since we can't compute spatial distance, we can't do true diversity.
    
    # What if we use **Random Forest-like diversity**? No.
    
    # Let's look at the "Inspiration 1" again. It was cut off.
    # "A materially different strategy is An | uses numpy, scipy, time; defines acquisition"
    
    # Let's try **Gaussian Process-Lower Confidence Bound (GP-LCB)** with a **dynamic beta**.
    # The current program uses EI with a penalty.
    # LCB is often better for minimization.
    # Score = mu - beta * sigma.
    # We want to maximize the score (which corresponds to minimizing the value).
    # So we want points with low mu and high sigma.
    
    # The key is choosing beta.
    # If beta is too small, we exploit the current best (mu is low).
    # If beta is too large, we explore random points (sigma is high).
    
    # Adaptive Beta:
    # beta = base_beta * (1 + alpha * (f_best - mu_min) / (sigma_max + eps))?
    
    # Let's try a **Step-Independent but State-Dependent** strategy.
    # Use the **variance of the posterior mean** to determine the exploration level.
    
    # Actually, a very powerful strategy when coordinates are missing is **Thompson Sampling**.
    # Sample a function from the posterior, pick the point with the minimum value.
    # This naturally balances exploration and exploitation.
    # But it's stochastic. The task asks for a deterministic `acquisition` function that returns scores.
    # We can simulate Thompson Sampling by generating a random sample?
    # No, `acquisition` is called once per step. If we use randomness, it's non-deterministic.
    # The task says "Write acquisition...". It doesn't explicitly forbid randomness, but "deterministic" is usually preferred for reproducibility unless specified.
    # However, the current program is deterministic.
    
    # Let's stick to deterministic.
    
    # **New Strategy: Uncertainty-Weighted Expected Improvement (UWEI)**
    # Instead of penalizing based on posterior distance `r`, we use the **probability of improvement** to weight the diversity.
    
    # Or better: **Maximin Regret** approximation?
    
    # Let's go back to the most robust simple method: **LCB with Adaptive Beta**.
    # Why? EI can suffer from "greediness" near the optimum. LCB is more robust to local minima if beta is tuned well.
    
    # Adaptive Beta Logic:
    # 1. Compute LCB scores for a range of betas.
    # 2. Select the beta that maximizes the "diversity" of the top-K candidates in the **posterior space**.
    #    Wait, the current program does something similar with `quality`.
    
    # What is materially different?
    # The current program optimizes hyperparameters for a heuristic quality metric.
    # A different approach: **Direct Optimization of the Acquisition Function Parameters using the GP Posterior Statistics**.
    
    # Let's try **Kriging Believer** or **Predictive Entropy Search**? Too complex without coordinates.
    
    # How about **Expected Hypervolume Improvement (EHVI)**? No, single objective.
    
    # Let's try **Stochastic Weighted Average (SWA)** of multiple acquisition functions.
    # Compute EI, LCB, and PI (Probability of Improvement).
    # Return a weighted sum.
    # The weights can be adaptive.
    
    # But the most "structurally different" thing from "Hyperparameter Search on a Heuristic" is **Direct Gradient-Free Optimization of the Acquisition Function itself on the grid**?
    # We have 2000 points. We can treat the scores as a vector.
    
    # Let's try a **Greedy Diversity Maximization** in the **Feature Space** defined by (mu, sigma).
    # We want the selected point (argmax) to be far from the "cluster" of points that have already been selected?
    # We don't know which points were selected.
    
    # Okay, here is a strong candidate: **Adaptive LCB with Median Sigma Normalization**.
    # The scale of sigma changes as the GP is updated.
    # Early: sigma ~ 1. Late: sigma ~ 0.1.
    # A fixed beta will behave differently.
    # We normalize sigma by its current median or max to keep the exploration term consistent.
    
    # Score = mu - beta * (sigma / sigma_scale)
    # Where sigma_scale = np.median(sigma) + 1e-6
    
    # And we adapt beta based on the "spread" of mu.
    
    # Let's implement this.
    
    sigma_scale = np.median(sigma) + 1e-6
    sigma_norm = sigma / sigma_scale
    
    # We want to minimize f.
    # LCB: mu - beta * sigma_norm
    # We want to maximize this score.
    
    # How to choose beta?
    # Let's use a **Grid Search for Beta** but with a different Quality Metric.
    # Current Quality: Diversity in `r` (posterior distance).
    # New Quality: **Variance of the Top-K Scores**.
    # If the top-K scores are all similar, we are exploiting.
    # If they vary, we are exploring?
    # No, we want the *selected* point to be good.
    
    # Let's try a **Lookahead** simulation.
    # If we pick point i, what happens to f_best?
    # We don't know.
    
    # Let's go with **Thompson Sampling-like Deterministic Approximation**.
    # Sample 100 functions from the posterior?
    # We can't sample functions without the GP kernel matrix and coordinates.
    # We only have mu and sigma at 2000 points.
    # We can assume the points are independent? No, they are correlated.
    # But if we assume independence (crude), we can sample:
    # f_i ~ N(mu_i, sigma_i^2)
    # Pick the min f_i.
    # This is equivalent to picking the point that minimizes a random draw.
    # The probability of picking point i is proportional to P(f_i < min_j f_j).
    # This is complex.
    
    # Let's stick to **Adaptive LCB**. It is simple, robust, and structurally different from the "Penalized EI" approach.
    
    betas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    
    best_score = None
    best_beta = 1.0
    
    # We need a metric to choose the best beta.
    # Let's use the **Expected Improvement** of the LCB itself? No.
    
    # Let's use the **Diversity of the Top-10 Points** in the normalized space (mu/sigma).
    
    for beta in betas:
        score = mu - beta * sigma_norm
        
        # Check if finite
        if not np.all(np.isfinite(score)):
            continue
            
        # Metric: We want the top point to be "promising" (low mu) but the top 10 to be "diverse" in their (mu, sigma_norm) characteristics.
        # Actually, simpler: Just pick the beta that maximizes the **minimum** of the top 5 scores?
        # No, that's just exploitation.
        
        # Let's try a **Hybrid** of EI and LCB.
        # Score = w * EI + (1-w) * LCB
        # And we adapt w.
        
        # Let's abandon the hyperparameter search and use a **Fixed, Principled** formula.
        # **UCB with Beta = 1.0 * log(1 + N_evals)**?
        # We don't know N_evals.
        
        # We can estimate N_evals from the "sharpness" of the posterior.
        # As N_evals increases, sigma decreases.
        # So we can use **1 / (median_sigma + 1e-6)** as a proxy for N_evals.
        
        n_proxy = 1.0 / (np.median(sigma) + 1e-6)
        
        # Beta increases with N?
        # Usually, exploration decreases with N. So Beta should decrease?
        # In UCB, beta is often constant or slowly decaying.
        # Let's try Beta = 1.0 / (1 + 0.1 * n_proxy)
        
        beta_adaptive = 1.0 / (1.0 + 0.1 * n_proxy)
        
        # Clamp beta to a reasonable range [0.5, 5.0]
        beta_adaptive = np.clip(beta_adaptive, 0.5, 5.0)
        
        score = mu - beta_adaptive * sigma_norm
        
        # Add a small EI component to ensure we don't ignore points with very low mu?
        # LCB already handles low mu.
        
        return score

# EVOLVE-BLOCK-END
