import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Deterministic Upper Confidence Bound (UCB) for minimization
    # UCB(x) = mu(x) + kappa * sigma(x)
    # We want to minimize, so we maximize -UCB(x)
    # For multimodal functions like Weierstrass, high kappa encourages exploration
    
    # Adaptive kappa: start high for exploration, decrease as f_best improves
    # f_best is the lowest observed value; for Weierstrass on [-0.5, 0.5], min is 0
    # As f_best approaches 0, we can reduce exploration
    # Normalize f_best to [0, 1] range approximately
    f_best_norm = max(0.0, min(1.0, f_best / 1.0))  # f_best is typically in [0, 1]
    
    # Start with high kappa for exploration, decrease as we get closer to optimum
    kappa = 2.0 + 3.0 * (1.0 - f_best_norm)
    
    # Compute UCB scores: higher is better for selection
    # For minimization, we want to explore regions with high uncertainty AND low mean
    # UCB for minimization: score = -(mu + kappa * sigma)
    scores = -(mu + kappa * sigma)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
