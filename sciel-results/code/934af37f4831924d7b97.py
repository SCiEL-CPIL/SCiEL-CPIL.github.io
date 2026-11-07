"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive UCB with a deadline guard.
    Higher score = more likely to be evaluated next.
    """
    # Deadline guard: if we're running out of time, fall back to a cheap exploitation score
    if time.time() > _DEADLINE:
        scores = -mu / cost
        if not np.all(np.isfinite(scores)):
            scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Adaptive beta: more exploration early, more exploitation later.
    # f_best starts high (rosenbrock values can be large) and decreases toward 0.
    # Map f_best from [large, 0] to beta in [2.5, 0.6].
    # Use a smooth transition: beta = 0.6 + 1.9 * sigmoid(-f_best / scale)
    # scale ~ 50 is reasonable for rosenbrock 2d on [-2,2]^2
    scale = 50.0
    # sigmoid(-f_best / scale): when f_best is large positive, sigmoid ~ 0, beta ~ 0.6
    # when f_best is near 0, sigmoid ~ 0.5, beta ~ 1.55
    # when f_best is negative (unlikely but possible), sigmoid > 0.5, beta > 1.55
    # Actually we want: early (f_best large) -> high beta (explore), late (f_best small) -> low beta (exploit)
    # So: beta = 0.6 + 1.9 * sigmoid(f_best / scale)
    # When f_best is large: sigmoid ~ 1, beta ~ 2.5 (explore)
    # When f_best is near 0: sigmoid ~ 0.5, beta ~ 1.55
    # When f_best is negative: sigmoid < 0.5, beta < 1.55 (exploit)
    # Hmm, that doesn't give enough exploitation late. Let me use a different mapping.
    
    # Better: use f_best directly. Early f_best might be ~100-1000, late f_best ~0.
    # beta = 0.5 + 2.0 * (1 / (1 + exp((f_best - 10) / 5)))
    # When f_best >> 10: exp large, 1/(1+large) ~ 0, beta ~ 0.5
    # When f_best << 10: exp small, 1/(1+small) ~ 1, beta ~ 2.5
    # When f_best ~ 10: beta ~ 1.5
    
    # Actually, let's think about it differently. We want:
    # - When f_best is still far from optimal (large), explore more (high beta)
    # - When f_best is close to optimal (small), exploit more (low beta)
    
    # Use: beta = 0.5 + 2.0 * sigmoid((f_best - 5) / 3)
    # f_best = 20: sigmoid(5) ~ 1, beta ~ 2.5
    # f_best = 5: sigmoid(0) = 0.5, beta ~ 1.5
    # f_best = 0: sigmoid(-5/3) ~ 0.06, beta ~ 0.62
    # f_best = -1: sigmoid(-2) ~ 0.12, beta ~ 0.74
    
    # This seems reasonable. Let's use it.
    beta = 0.5 + 2.0 * (1.0 / (1.0 + np.exp(-(f_best - 5.0) / 3.0)))
    
    # Clip beta to a reasonable range
    beta = np.clip(beta, 0.3, 3.0)
    
    # Cost-aware UCB for minimization:
    # We want low predicted value (mu) and high uncertainty (sigma), but penalize high cost.
    # Score = (-mu + beta * sigma) / cost
    # This gives higher score to points that are cheap and either have low mean or high uncertainty.
    
    scores = (-mu + beta * sigma) / cost
    
    # Ensure finite values
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
# EVOLVE-BLOCK-END
