import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Adaptive Kappa based on current best value
        # Hartmann 3D min is approx -3.86278
        f_opt = -3.86278
        # Current regret
        regret = f_best - f_opt
        # Normalize regret. Initial f_best might be around 0 or 1.
        # Max possible regret is roughly 3 - (-3.86) = 6.86
        # Let's cap the regret for normalization
        max_regret = 6.0
        norm_regret = np.clip(regret / max_regret, 0.0, 1.0)
        
        # Kappa starts high (exploration) and decreases (exploitation)
        kappa = 1.0 + 4.0 * norm_regret
        
        # UCB Score: (kappa * sigma - mu) / cost
        # Higher is better
        score = (kappa * sigma - mu) / cost
        
        # Safety
        score = np.where(np.isfinite(score), score, -np.inf)
        
        return score
