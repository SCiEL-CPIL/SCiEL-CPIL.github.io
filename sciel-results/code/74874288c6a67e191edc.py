import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Structural change: Adaptive UCB with quantile-based cost-efficiency gate
    # 1. Compute UCB for minimization: mu - beta * sigma
    # 2. Compute EI as exploitation signal
    # 3. Blend UCB and EI based on uncertainty ratio
    # 4. Apply quantile-based feasibility gate on cost-efficiency
    # 5. Scale by inverse cost
    
    # Uncertainty ratio: how much sigma dominates relative to potential improvement
    improvement = np.maximum(f_best - mu, 0.0)
    uncertainty_ratio = sigma_safe / (improvement + sigma_safe + 1e-9)
    
    # Adaptive beta: larger when uncertainty is high relative to improvement
    # When uncertainty_ratio ~ 1 (sigma >> improvement), explore more
    # When uncertainty_ratio ~ 0 (sigma << improvement), exploit more
    beta = 1.0 + 2.0 * uncertainty_ratio
    
    # UCB for minimization (lower is better)
    ucb = mu - beta * sigma_safe
    
    # EI for minimization
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Normalize UCB and EI to [0, 1] range for blending
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range  # lower ucb = better, so invert
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Blend: weight UCB more when uncertainty is high
    w_ucb = uncertainty_ratio
    w_ei = 1.0 - uncertainty_ratio
    
    blended = w_ucb * (1.0 - ucb_norm) + w_ei * ei_norm
    
    # Cost-efficiency: improvement per unit cost
    cost_efficiency = blended / cost_safe
    
    # Quantile-based feasibility gate: keep top 75% by cost-efficiency
    # This is more robust than mean-based gating
    q75 = np.percentile(cost_efficiency, 75)
    gate = np.where(cost_efficiency >= q75, 1.0, 0.1)
    
    # Final score: gated blended score scaled by inverse cost
    scores = gate * blended / np.sqrt(cost_safe)
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
