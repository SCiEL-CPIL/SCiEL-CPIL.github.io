"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
# Set deadline to 320s to be safe against the 330s kill
_DEADLINE = _START_TIME + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining LCB (Lower Confidence Bound) 
    with Probability of Improvement (PI), adapted for short-budget Rosenbrock 2D.
    
    Parameters:
    -----------
    mu : numpy.ndarray
        GP mean predictions, shape (2000,).
    sigma : numpy.ndarray
        GP standard deviations, shape (2000,).
    f_best : float
        Best observed function value so far.
    cost : numpy.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : numpy.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return zeros to indicate no preference, or random? 
        # Zeros are finite and valid.
        return np.zeros_like(mu)

    # Ensure sigma is positive to avoid division by zero
    # Using a small epsilon
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate z-scores for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, and PI is high.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is the CDF of the standard normal distribution at z.
    pi_scores = norm.cdf(z)
    
    # Cost-aware adjustment
    # We divide by cost to penalize expensive evaluations.
    # Cost is in [1, 2], so this is a mild penalty.
    
    # LCB (Lower Confidence Bound): -mu + beta * sigma
    # For minimization, we want low mu and high sigma (exploration).
    # The sign is flipped so higher score = better (lower predicted value + more uncertainty).
    # We use an adaptive beta that decreases as f_best improves (exploitation later).
    
    # Adaptive beta:
    # Early on (f_best large), we want more exploration (higher beta).
    # Later (f_best small), we want more exploitation (lower beta).
    # Using a sigmoid-like transition based on f_best.
    # Rosenbrock 2D min is 0. f_best starts high (~100+) and drops to 0.
    # We want beta to be high when f_best is high, low when f_best is low.
    # beta = beta_min + (beta_max - beta_min) * sigmoid(k * (f_best - center))
    
    beta_min = 0.5
    beta_max = 2.0
    center = 10.0  # Transition point
    k = 0.5        # Sharpness
    
    # Sigmoid: 1 / (1 + exp(-k * (f_best - center)))
    # If f_best >> center, sigmoid ~ 1, beta ~ beta_max
    # If f_best << center, sigmoid ~ 0, beta ~ beta_min
    # To avoid overflow in exp, we clip the argument
    arg = k * (f_best - center)
    arg = np.clip(arg, -50, 50)
    sigmoid_val = 1.0 / (1.0 + np.exp(-arg))
    beta = beta_min + (beta_max - beta_min) * sigmoid_val
    
    # LCB score: -mu + beta * sigma
    # Higher score means lower mu or higher sigma (uncertainty).
    lcb_scores = -mu + beta * sigma_safe
    
    # Normalize scores to comparable scales
    # LCB scores can be large negative or positive.
    # PI scores are in [0, 1].
    # We blend them.
    
    # Cost-aware LCB: divide by cost
    lcb_cost = lcb_scores / cost_safe
    
    # Cost-aware PI: divide by cost
    pi_cost = pi_scores / cost_safe
    
    # Blend LCB and PI
    # LCB is good for exploration (high uncertainty, low mean).
    # PI is good for exploitation (high probability of improvement).
    # We use a fixed blend or adaptive blend.
    # Given the short budget, exploration is critical early.
    # Let's use 0.5 * LCB + 0.5 * PI.
    # However, LCB scores are not bounded like PI.
    # Let's normalize LCB scores to [0, 1] range roughly?
    # Or just scale them.
    
    # Alternative: Use Expected Improvement (EI) which is a better blend.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # EI is always positive and scales with the potential improvement.
    # Let's compute EI and compare.
    
    # Compute EI
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * pi_scores + sigma_safe * phi_z
    
    # Cost-aware EI
    ei_cost = ei / cost_safe
    
    # Now we have three candidates:
    # 1. pi_cost
    # 2. lcb_cost
    # 3. ei_cost
    
    # The finding says: "Cost-scaled LCB (beta*sigma/cost - mu) shifts exploration toward low-cost, high-uncertainty regions, which in the short 30-cost budget avoids wasting evaluations on expensive far corners of the Rosenbrock domain and reaches the valley sooner than pure EI/cost"
    
    # So LCB/cost is better than EI/cost.
    # Let's stick with LCB/cost but maybe blend with a bit of PI/cost for stability.
    # Or just use LCB/cost.
    
    # Let's try: score = lcb_cost
    # But LCB can be very negative if mu is very high.
    # Let's shift LCB so the minimum is 0?
    # min_lcb = np.min(lcb_cost)
    # lcb_shifted = lcb_cost - min_lcb
    # This ensures non-negative scores, which is safer.
    
    min_lcb = np.min(lcb_cost)
    lcb_shifted = lcb_cost - min_lcb
    
    # Blend with PI?
    # PI is bounded [0, 1]. LCB shifted is unbounded.
    # Let's normalize PI to similar scale?
    # Or just use LCB shifted.
    
    # Let's try a mix: 0.7 * lcb_shifted + 0.3 * pi_cost * (some_scale)
    # The scale of lcb_shifted depends on the data.
    # Let's just use lcb_shifted for now, as the finding supports it.
    
    scores = lcb_shifted
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
