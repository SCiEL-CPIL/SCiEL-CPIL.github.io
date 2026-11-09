"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Standard EI for minimization
    z_ei = (f_best_safe - mu_safe) / sigma_safe
    ei_scores = (f_best_safe - mu_safe) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Multiple exploration strategies with different kappa values
    # For minimization, LCB score = -mu + kappa * sigma
    kappa_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    lcb_scores_list = []
    for kappa in kappa_values:
        lcb_scores_list.append(-mu_safe + kappa * sigma_safe)
    
    # PI (Probability of Improvement) for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi_scores = norm.cdf((f_best_safe - mu_safe) / sigma_safe)
    
    # EI with noise (xi parameter) - adds robustness
    xi_values = [0.0, 0.01, 0.02, 0.05]
    ei_xi_scores_list = []
    for xi in xi_values:
        z_xi = (f_best_safe - xi - mu_safe) / sigma_safe
        ei_xi = (f_best_safe - xi - mu_safe) * norm.cdf(z_xi) + sigma_safe * norm.pdf(z_xi)
        ei_xi_scores_list.append(ei_xi)
    
    # Normalize each score component to [0, 1] range for fair combination
    def normalize(scores):
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min < 1e-10:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
    
    # Create weighted portfolio of acquisition functions
    # Weights are chosen to balance exploitation and exploration
    components = []
    weights = []
    
    # EI variants
    components.append(normalize(ei_scores))
    weights.append(0.25)
    
    for i, ei_xi in enumerate(ei_xi_scores_list):
        components.append(normalize(ei_xi))
        weights.append(0.05)
    
    # LCB variants
    for lcb in lcb_scores_list:
        components.append(normalize(lcb))
        weights.append(0.10)
    
    # PI
    components.append(normalize(pi_scores))
    weights.append(0.10)
    
    # Combine with weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize weights
    
    final_scores = np.zeros_like(mu_safe)
    for w, comp in zip(weights, components):
        final_scores += w * comp
    
    # Safety check
    if time.time() - start_time > 280.0:
        return _sanitize(ei_scores)
    
    final_scores = _sanitize(final_scores)
    
    return final_scores
# EVOLVE-BLOCK-END
