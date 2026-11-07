import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Deterministic Global Constrained strategy.
    
    Strategy:
    1. Instead of sampling a single target (Thompson Sampling), we compute a 
       robust score that represents the probability of improving upon the 
       current best, weighted by the potential gain, across a range of 
       plausible global minima.
    2. We use a grid of target values (xi) from f_best down to the known 
       theoretical minimum (-0.8). This allows the acquisition to be "global" 
       in nature, not just local to the current best.
    3. We compute the Probability of Improvement (PI) and Expected Improvement 
       (EI) for each target.
    4. We aggregate these scores. A common robust approach is to maximize the 
       probability that the point is better than a "hard" target (e.g., -0.8) 
       while also considering the expected gain. 
       However, a simpler and often more effective structural change is to 
       use a weighted sum of EI over different xi values, or simply a 
       "Global EI" that integrates over the posterior of the minimum.
       
       Let's use a specific structural variant: 
       Score = E[ max(0, target - f(x)) ] where target is sampled from a 
       distribution reflecting the belief about the global minimum.
       
       To make it deterministic and fast, we approximate this expectation 
       by averaging EI over a few key thresholds: 
       - f_best (local improvement)
       - (f_best + global_min) / 2 (mid-range)
       - global_min (global convergence)
       
       Actually, the "inspiration" from the cell findings suggests:
       "Dynamic xi lowers the improvement threshold early... combined with a 
       soft cost normalization (1+cost_factor) instead of hard division".
       
       Let's implement:
       1. Compute EI for a set of xi values.
       2. Combine them (e.g., sum or max).
       3. Apply soft cost penalty: score = combined_score / (1 + cost_penalty * cost)
       where cost_penalty is a small constant or adaptive.
       
       Given the budget (30s per eval, 300s total), we need to be very fast.
       2000 points is small.
       
       Let's try:
       1. Define targets: [f_best, -0.4, -0.8]
       2. Compute EI for each target.
       3. Sum the EIs (this encourages points that are good for multiple targets).
       4. Normalize by (1 + 0.5 * cost) to softly penalize cost.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Theoretical minimum of cosine8 is -0.8
    global_min = -0.8
    
    # Define a set of target thresholds (xi)
    # We want to balance local exploitation (f_best) with global convergence (global_min)
    # Early on, f_best is high, so we should look towards global_min.
    # As f_best approaches global_min, we focus on f_best.
    
    # Dynamic xi selection:
    # If f_best is far from global_min, use a lower xi to explore deeper.
    # If f_best is close, use f_best to refine.
    
    # Let's use a weighted average of EI for f_best and EI for global_min.
    # This effectively creates a "Global EI" that is robust to the current state.
    
    # EI calculation helper
    def calc_ei(mu, sigma, xi):
        z = (xi - mu) / sigma
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        ei = (xi - mu) * Phi_z + sigma * phi_z
        return ei
    
    # Compute EI for current best (local)
    ei_local = calc_ei(mu, sigma, f_best)
    
    # Compute EI for global min (global)
    ei_global = calc_ei(mu, sigma, global_min)
    
    # Weighting:
    # When f_best is high (e.g., 0.0), we care more about finding the global min.
    # When f_best is low (e.g., -0.79), we care more about local refinement.
    # Let's use a simple linear blend based on the gap.
    gap = f_best - global_min
    # If gap is large, weight global more. If gap is small, weight local more.
    # Normalize gap: max gap is roughly 0.8 - (-0.8) = 1.6? No, f starts around 0.
    # Let's say gap is in [0, 1.5].
    # w_global = gap / (gap + 0.1)  -> approaches 1 as gap increases
    # w_local = 0.1 / (gap + 0.1)   -> approaches 0 as gap increases
    
    # To keep it stable, let's use a fixed weight or a simple adaptive one.
    # The "inspiration" mentioned dynamic xi. 
    # Let's try: xi = f_best - 0.1 * gap. 
    # This shifts the target down as we get better.
    
    # Actually, the "Probability of Improvement" over a range is more robust.
    # Let's stick to the sum of EIs for a few points, which is a valid structural 
    # change from single-sample TS.
    
    # Let's use 3 targets: f_best, (f_best+global_min)/2, global_min
    xi_mid = (f_best + global_min) / 2.0
    ei_mid = calc_ei(mu, sigma, xi_mid)
    
    # Sum them. This creates a smoother, more global landscape.
    combined_score = ei_local + ei_mid + ei_global
    
    # Soft Cost Normalization
    # Instead of dividing by cost (which can be unstable if cost is small or large),
    # we use a soft penalty.
    # Cost is in [1, 2]. 
    # score = combined_score / (1 + cost_penalty * cost)
    # If cost_penalty is small, it's a mild preference for cheap points.
    # If we divide by cost directly, it's a strong preference.
    # The finding said "soft cost normalization (1+cost_factor) instead of hard division".
    
    # Let's use cost_factor = 0.5. 
    # This means a point with cost 2.0 is penalized by 1.5x compared to cost 0 (though cost >= 1).
    # Point with cost 1.0: denom 1.5
    # Point with cost 2.0: denom 2.0
    # Ratio: 1.5/2.0 = 0.75. So expensive points are 25% less attractive.
    # This seems reasonable.
    
    cost_penalty = 0.5
    denom = 1.0 + cost_penalty * cost
    scores = combined_score / denom
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    if time.time() > deadline:
        return np.ones_like(mu)
        
    return scores
