import numpy as np
from scipy.stats import norm
import time

# Track call time for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # 330s limit minus safety margin

# Track f_best history to detect stagnation
_f_best_history = []
_last_f_best = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration parameter based on progress.
    
    Higher score means evaluate next.
    """
    global _f_best_history, _last_f_best
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: prioritize low mu (exploitation)
        return -mu.copy()
    
    # Track f_best for stagnation detection
    if _last_f_best is not None and f_best >= _last_f_best - 1e-10:
        _f_best_history.append(0)
    else:
        _f_best_history.append(1)
    _last_f_best = f_best
    
    # Keep only last 10 history entries
    if len(_f_best_history) > 10:
        _f_best_history = _f_best_history[-10:]
    
    # Determine exploration parameter based on progress
    # Early on (few evaluations), explore more; later, exploit more
    n_steps = len(_f_best_history)
    
    # Count how many recent steps had improvement
    recent_improvements = sum(_f_best_history[-5:]) if len(_f_best_history) >= 5 else sum(_f_best_history)
    recent_count = min(5, len(_f_best_history))
    improvement_ratio = recent_improvements / max(1, recent_count)
    
    # Adaptive kappa: higher when stagnating, lower when making progress
    # Base kappa depends on how far along we are
    if n_steps <= 5:
        base_kappa = 2.0  # Early: high exploration
    elif n_steps <= 15:
        base_kappa = 1.0  # Mid: moderate exploration
    else:
        base_kappa = 0.5  # Late: more exploitation
    
    # If stagnating, increase exploration
    if improvement_ratio < 0.2:
        kappa = base_kappa * 2.0
    elif improvement_ratio < 0.5:
        kappa = base_kappa * 1.5
    else:
        kappa = base_kappa
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Compute UCB = -mu + kappa * sigma (for minimization, we want low mu + high uncertainty)
    UCB = -mu + kappa * sigma_safe
    
    # Normalize EI and UCB to similar scales for combination
    # EI is in objective units, UCB is also in objective units
    # Weight them: 50% EI, 50% UCB, but adjust based on progress
    if n_steps <= 10:
        ei_weight = 0.3
        ucb_weight = 0.7
    elif n_steps <= 20:
        ei_weight = 0.5
        ucb_weight = 0.5
    else:
        ei_weight = 0.7
        ucb_weight = 0.3
    
    # Scale EI to be comparable with UCB
    # EI is typically smaller than |mu| scale, so normalize
    ei_max = np.max(EI) if np.max(EI) > 1e-10 else 1.0
    EI_normalized = EI / ei_max
    
    ucb_min = np.min(UCB)
    ucb_max = np.max(UCB)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-10:
        ucb_range = 1.0
    UCB_normalized = (UCB - ucb_min) / ucb_range
    
    # Combine
    score = ei_weight * EI_normalized + ucb_weight * UCB_normalized
    
    # Handle numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=0.0)
    
    return score
