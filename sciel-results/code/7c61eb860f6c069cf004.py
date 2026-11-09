"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track evaluation count for adaptive exploration
_eval_count = 0
_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining Thompson Sampling with
    an exploration bonus that decays over the evaluation budget.
    
    The strategy uses a hybrid approach:
    - Early evaluations: more exploration (higher weight on sigma)
    - Later evaluations: more exploitation (lower weight on sigma)
    
    We use a deterministic UCB-like score with a time-decaying exploration
    coefficient, combined with a small Thompson Sampling component for
    stochastic exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means evaluate next.
    """
    global _eval_count, _start_time
    
    _eval_count += 1
    
    # Total budget is 30 evaluations (including initial 5)
    # We've had 5 initial + (eval_count - 1) more = eval_count evaluations total
    # So remaining fraction: (30 - eval_count) / 30
    total_budget = 30
    remaining_frac = max(0.0, (total_budget - _eval_count) / total_budget)
    
    # Time-based safety: if we're running low on time, be more exploitative
    elapsed = time.time() - _start_time
    time_remaining_frac = max(0.0, 1.0 - elapsed / 320.0)  # 320s margin before 330s kill
    
    # Exploration coefficient: starts high, decays
    # Use a combination of evaluation-based and time-based decay
    beta = 2.0 * (0.5 + 0.5 * remaining_frac) * (0.5 + 0.5 * time_remaining_frac)
    
    # Ensure beta is within reasonable bounds
    beta = np.clip(beta, 0.5, 3.0)
    
    # UCB component: -mu + beta * sigma (for minimization, higher score = better)
    ucb_score = -mu + beta * sigma
    
    # Add a small Thompson Sampling component for stochasticity
    # This helps escape local optima
    z = np.random.standard_normal(len(mu))
    ts_component = 0.3 * sigma * z
    
    # Combine: primarily UCB, with small TS noise
    scores = ucb_score + ts_component
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
