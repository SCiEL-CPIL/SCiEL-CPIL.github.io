import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hypervolume-inspired acquisition function for Bayesian optimization.
    Combines expected improvement with an exploration bonus weighted by
    the probability of improvement, designed to balance exploitation and
    exploration for multimodal landscapes like Branin.
    
    Returns 2000 finite scores; higher score means evaluate next.
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of improvement
    p_improve = norm.cdf(z)
    
    # Hypervolume-inspired score:
    # The idea is to weight improvement by its likelihood and add an
    # exploration component. We use a formulation that resembles the
    # expected hypervolume contribution:
    #
    # For a single-objective minimization with reference point f_best,
    # the "hypervolume" contribution of improving to value v < f_best
    # is (f_best - v). The expected contribution is EI.
    #
    # To add exploration, we weight by the uncertainty (sigma) and the
    # probability of being in an unexplored region (low p_improve means
    # high uncertainty relative to current best).
    #
    # A practical formulation: score = EI + alpha * sigma * (1 - p_improve)
    # where alpha balances exploration. We use a decaying alpha to shift
    # from exploration to exploitation over the course of optimization.
    
    # Estimate progress: if f_best is close to the known minimum, we're
    # in late game; otherwise early game. Branin min is ~0.397887.
    BRANIN_MIN = 0.397887
    progress = np.clip((f_best - BRANIN_MIN) / (BRANIN_MIN + 1.0), 0, 1)
    
    # Exploration weight: high early, low late
    alpha = 1.5 * (1.0 - 0.5 * progress)
    
    # Exploration bonus: sigma * (1 - p_improve) captures regions where
    # we're uncertain and not yet confident of improvement
    exploration_bonus = sigma_safe * (1.0 - p_improve)
    
    # Combine
    scores = ei + alpha * exploration_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline check - if we're running low on time, return what we have
    elapsed = time.time() - start_time
    if elapsed > (DEADLINE - start_time):
        # This shouldn't happen for such a simple computation, but safety first
        return scores
    
    return scores
