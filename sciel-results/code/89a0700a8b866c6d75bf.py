"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division errors
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # Higher EI means better expected reduction in f_best
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB score (higher is better for minimization)
    # UCB(x) = -mu(x) + beta * sigma(x)
    # We use a small beta to slightly favor exploration in high uncertainty regions,
    # but primarily rely on the mean (exploitation).
    beta = 0.1
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to have zero mean and unit variance
    # This allows for a fair linear combination regardless of scale
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_mean) / ei_std
        
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std < 1e-9:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_mean) / ucb_std
        
    # Determine the weight for EI vs UCB
    # Early on (f_best large), we want to explore (UCB).
    # Later (f_best small), we want to exploit (EI).
    # However, EI naturally shifts from exploration to exploitation.
    # UCB is more aggressive in exploration.
    # Let's use a fixed blend or a simple adaptive one.
    # Since the directive asks for a structural change towards PMM (which is just -mu),
    # and the current best is EI-based, let's try a hybrid that heavily favors
    # the predictive mean but uses EI to break ties or guide local search.
    
    # Actually, the directive suggests: "Use a Posterior Mean Minimization (PMM) strategy 
    # that directly returns -mu as the score, ignoring uncertainty (sigma) entirely".
    # But the task is to COMBINE the current program (EI) with inspirations.
    # Inspiration 2 (PMM) scored 1.0832. Inspiration 1 (UCB) scored 14.9486.
    # Current program (EI hybrid) scored 0.1977.
    
    # The UCB score is very high (14.9). The PMM score is decent (1.08).
    # The EI score is low (0.19).
    
    # Let's look at the UCB formula: -mu + beta * sigma.
    # If we set beta=0, it becomes PMM (-mu).
    # If we set beta large, it becomes pure exploration.
    # The UCB with beta=0.5 scored very well.
    
    # Let's try a pure UCB with a tuned beta, or a blend of UCB and EI.
    # Given the stagnation, let's pivot to the structure of Inspiration 1 (UCB)
    # but refine it. The high score of UCB suggests that for this problem,
    # aggressive exploitation of the mean with a small uncertainty bonus works well.
    
    # Let's return a UCB score with a beta that decays as f_best improves.
    # Early: high beta (explore). Late: low beta (exploit).
    # f_best starts high (sphere 1d max is 25 at x=5, but min is 0).
    # If f_best is close to 0, we want beta close to 0.
    
    # Use a simple decay: beta = beta_max * exp(-decay * f_best)
    # If f_best is 25, beta = beta_max.
    # If f_best is 0, beta = beta_max * exp(0) = beta_max? No, we want it to go down.
    # Let's use beta = beta_max * (f_best / f_max_estimate) or similar.
    # Or just a fixed beta that worked well: beta=0.5.
    
    # Let's try a blend of the normalized EI and normalized UCB.
    # Since UCB scored much higher, let's weight UCB more heavily.
    w_ei = 0.2
    w_ucb = 0.8
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
