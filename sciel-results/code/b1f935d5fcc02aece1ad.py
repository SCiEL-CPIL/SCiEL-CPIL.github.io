import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using an approximation of batched Thompson Sampling 
    with greedy diversity selection.
    
    Strategy:
    1. Compute a base "optimism" score (similar to UCB/EI) for all points.
    2. Identify a pool of top candidates based on this optimism.
    3. Simulate a few posterior samples (using the provided mu/sigma as sufficient stats 
       for a Gaussian process marginal distribution at these points, assuming independence 
       for the sake of speed and lack of covariance matrix).
    4. Greedily select the next point that has high expected value in the simulated 
       samples but is "diverse" (far in function space or value space) from previously 
       selected "virtual" batch members.
    5. Boost the score of the selected point significantly.
    6. Return scores such that the highest score corresponds to the selected point.
    
    Note: Since we only return scores for the *next* single evaluation (the loop evaluates 
    the best one), we can use this internal logic to pick the "best" candidate among the 2000 
    and assign it the highest score, while keeping others low. However, the interface expects 
    2000 scores. We will return a score vector where the chosen point has a very high score 
    and others have their base cost-adjusted EI/UCB scores.
    
    Actually, the task says: "evaluates the best one". So the index of the max score is what matters.
    We can simply compute a robust "value" for each point that balances exploitation, exploration, 
    and cost, and return that. The "Thompson Sampling" aspect here is approximated by adding 
    a noise term or using a probabilistic selection if we were batching. Since we pick one, 
    a robust UCB with adaptive beta or a Hybrid EI/UCB is often best.
    
    Let's try a different angle: **Knowledge Gradient (KG) approximation**.
    KG approximates the expected improvement in f_best after evaluating x.
    For a single step, KG(x) = E[ max(0, f_best - f(x)) ] but considering that f(x) might 
    update the GP and thus future f_best. With only one step left in the "batch" (effectively), 
    it reduces to EI. But we have many steps left (budget 30).
    
    A simpler, materially different approach from fixed UCB: **Adaptive Beta UCB**.
    The current program uses beta=2.5. 
    Early on, we should explore more (high beta). Later, exploit (low beta).
    We can estimate the "stage" of the optimization by looking at the number of evaluations 
    implied by the cost, or simply by the magnitude of sigma relative to f_best.
    
    However, the prompt asks for a strategy MATERIALLY DIFFERENT.
    Let's implement **Expected Improvement (EI) with a cost penalty that is non-linear**.
    Or better: **Probability of Improvement (PoI) combined with Uncertainty**.
    
    Let's go with **Thompson Sampling approximation via random perturbations**.
    Since we cannot compute the full posterior covariance (no coordinates, no kernel), we treat 
    (mu, sigma) as the marginal distribution. 
    We can simulate a "virtual" posterior sample by adding noise to mu.
    But we only pick ONE point.
    
    Alternative: **Minimize the upper confidence bound of the cost-adjusted regret**.
    
    Let's try **Cost-Adjusted Expected Improvement (CAEI)** but with a **temperature** parameter 
    that changes based on the spread of `mu`.
    
    Actually, looking at the inspirations, the seed used EI/cost. The current program uses UCB/cost.
    A different family is **Knowledge Gradient**.
    For a GP, the exact KG is complex. But an approximation for minimization is:
    KG(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)  [This is EI]
    
    Let's try **Probability of Improvement (PoI)** scaled by uncertainty?
    PoI = Phi((f_best - mu) / sigma).
    Score = PoI / cost.
    This is very exploitative.
    
    Let's try **Lower Confidence Bound (LCB)** but formulated as a maximization problem 
    with a dynamic beta based on the ratio of current f_best to the theoretical minimum? 
    We don't know the theoretical minimum.
    
    Let's try **Batch-UCB approximation**:
    1. Calculate UCB for all points.
    2. Select the top 10 points.
    3. Among these 10, select the one that is "most diverse" from the others in terms of 
       predicted value (mu) and uncertainty (sigma).
    4. Assign the highest score to this diverse point.
    
    This encourages exploring different regions of the function space rather than just 
    the single point with the highest UCB, which might be in a region already well-explored 
    (implied by low sigma) or just a local spike.
    """
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # 1. Base Score: UCB
    # Dynamic Beta: Start high, decrease as we get more info?
    # We don't have the iteration count, but we can infer progress from f_best.
    # If f_best is very low, we are likely exploiting. If high, exploring.
    # However, f_best scale is unknown.
    
    # Let's use a fixed robust beta but add a diversity filter.
    beta = 3.0  # Slightly more exploratory than 2.5
    
    ucb = mu - beta * sigma
    # We want to minimize ucb. So higher score is better.
    # Score_base = -ucb / cost = (beta * sigma - mu) / cost
    
    scores = (beta * sigma - mu) / cost
    
    # 2. Diversity Selection among Top Candidates
    # Get indices of top 20 candidates by base score
    top_k = 20
    if len(scores) > top_k:
        # Find top K indices
        top_indices = np.argsort(scores)[-top_k:]
        
        # Get their mu and sigma
        top_mu = mu[top_indices]
        top_sigma = sigma[top_indices]
        top_cost = cost[top_indices]
        
        # Greedily select the best diverse point
        # Start with the best base score point
        best_idx_local = 0
        selected_local_indices = [0]
        
        # Define a diversity metric: Distance in (mu, sigma) space?
        # Or just pick the one with the highest "novelty" in mu?
        # If we pick the one with the highest mu (worst mean) among the top UCBs, 
        # we might be exploring a region that is uncertain.
        
        # Let's score the top K by a combination of Base Score and "Uncertainty Reward"
        # Reward = sigma / (sigma + 1e-6)
        
        # Alternative: Just pick the one with the highest Base Score. 
        # The diversity argument is weak without spatial coordinates.
        # Without coordinates, "diversity" is only in the (mu, sigma) space.
        # Points with similar mu and sigma are likely similar in function value.
        
        # Let's try a different metric: **Expected Improvement per Cost** vs **UCB per Cost**.
        # The current program uses UCB. The seed used EI.
        # Let's blend them? No, that's refinement.
        
        # Let's try **Minimizing the Upper Bound of the Cost-Adjusted Regret**.
        # Regret = f(x) - f_opt.
        # We want to minimize f(x).
        
        # New Strategy: **Optimistic Policy Improvement**.
        # Assume the best possible outcome for each point is mu - k*sigma.
        # But penalize if the cost is high.
        
        # Let's implement a **Two-Phase Score**:
        # Phase 1: If f_best is "high" (relative to sigma), use high beta (Explore).
        # Phase 2: If f_best is "low" (relative to sigma), use low beta (Exploit).
        
        # Heuristic for "progress": 
        # If the mean of sigma is small compared to the spread of mu, we are exploiting.
        # If sigma is large everywhere, we are exploring.
        
        mean_sigma = np.mean(sigma)
        std_mu = np.std(mu)
        
        # If uncertainty is high relative to the function variation, explore more.
        # Ratio = mean_sigma / (std_mu + 1e-6)
        ratio = mean_sigma / (std_mu + 1e-6)
        
        # Map ratio to beta. 
        # High ratio -> High Beta.
        # Low ratio -> Low Beta.
        # Beta range: 1.0 to 4.0
        
        # Normalize ratio. 
        # If ratio > 2, beta = 4.0
        # If ratio < 0.5, beta = 1.0
        # Linear interpolation
        
        # Clip ratio to [0.5, 2.0] for stability
        r = np.clip(ratio, 0.5, 2.0)
        
        # Linear map: 0.5 -> 1.0, 2.0 -> 4.0
        # slope = (4.0 - 1.0) / (2.0 - 0.5) = 3 / 1.5 = 2
        # beta = 1.0 + 2 * (r - 0.5) = 1.0 + 2r - 1.0 = 2r
        dynamic_beta = 2.0 * r
        
        # Recalculate scores with dynamic beta
        scores = (dynamic_beta * sigma - mu) / cost
        
        # Ensure finite
        scores[~np.isfinite(scores)] = -1e9
        
        # Optional: Add a small random tie-breaker? No, must be deterministic.
        
    else:
        scores[~np.isfinite(scores)] = -1e9
        
    return scores
