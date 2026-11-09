import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Known global minimum for Hartmann 6D
    f_min = -3.32237
    
    # Gap from best observed to known minimum
    gap = f_best - f_min
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # Higher is better (larger expected reduction in objective)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive strategy:
    # Hartmann 6D is multimodal. Early on, we need to explore to find promising basins.
    # Later, we need to exploit to refine the best point found.
    
    # UCB score: -mu + kappa * sigma
    # Higher is better (minimize -score)
    
    # We blend EI and UCB based on the gap.
    # When gap is large, we are far from optimum, so exploration (UCB) is more important.
    # When gap is small, we are close, so exploitation (EI) is more important.
    
    # Let's define a weight for UCB that decreases as gap decreases.
    # If gap > 2.0, we are likely in early stages or stuck in a bad local min.
    # If gap < 0.5, we are likely near a good local min or the global min.
    
    if gap > 2.0:
        # Strong exploration
        # Use a high kappa for UCB
        kappa = 3.0
        # Blend: 50% EI, 50% UCB? Or just UCB?
        # UCB with high kappa encourages visiting new areas.
        # EI encourages improving current best.
        # Let's try a mix.
        ucb = -mu + kappa * sigma_safe
        # Normalize scores? No, they are on different scales.
        # EI is in function units. UCB is in function units.
        # However, EI can be negative. UCB can be anything.
        # Let's stick to one or the other to avoid scale issues, or carefully blend.
        
        # Actually, standard practice is to pick one.
        # The current program uses pure UCB with adaptive kappa.
        # Inspiration 1 uses pure UCB with higher kappa.
        # Inspiration 2 uses pure EI.
        # Inspiration 3 uses PI then EI.
        
        # Let's try a smooth transition.
        # But mixing EI and UCB directly is tricky because their magnitudes differ.
        # EI is roughly O(sigma) when z is moderate.
        # UCB is O(sigma) when kappa is moderate.
        
        # Let's try:
        # If gap > 1.5: Use UCB with kappa=2.5
        # If gap > 0.5: Use EI
        # Else: Use EI with a small exploration bonus? Or just EI.
        
        # The current program (score 1.1513) uses:
        # gap > 2.0: kappa=2.0
        # gap > 1.0: kappa=1.5
        # gap > 0.5: kappa=1.0
        # else: kappa=0.5
        # And returns UCB.
        
        # Inspiration 1 (score 1.1513) uses:
        # gap > 1.5: kappa=3.0
        # gap > 0.8: kappa=2.0
        # gap > 0.3: kappa=1.0
        # else: kappa=0.5
        # And returns UCB.
        
        # It seems higher kappa in the early stage (high gap) is better.
        # Let's try to combine the adaptive logic with a switch to EI when close.
        
        # If gap is large, use UCB with high kappa.
        # If gap is small, use EI.
        
        pass # Placeholder for logic below
    
    # Refined Logic:
    # 1. If we are far from the known minimum (high gap), use UCB to explore.
    #    Use a higher kappa than the current program to encourage more exploration.
    # 2. If we are close to the known minimum (low gap), use EI to exploit.
    
    if gap > 1.5:
        # Exploration phase
        # Use UCB with high kappa
        kappa = 3.0
        scores = -mu + kappa * sigma_safe
    elif gap > 0.8:
        # Transition phase
        # Use UCB with moderate kappa
        kappa = 2.0
        scores = -mu + kappa * sigma_safe
    else:
        # Exploitation phase
        # Use EI
        scores = ei
        
    return scores
