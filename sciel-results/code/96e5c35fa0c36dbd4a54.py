"""Rewritten program with a structurally different acquisition strategy:
Bayesian optimization via Thompson sampling (posterior sampling) with cost-aware
scaling, using a Gaussian process posterior to sample from and select the
candidate with the best expected value per unit cost."""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds to spare before 300s limit

def _get_elapsed_time():
    return time.time() - _start_time

def _check_deadline():
    return _get_elapsed_time() > (300.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson sampling with posterior sampling.
    
    Strategy:
    1. Sample from the GP posterior: y_sample = mu + sigma * z, where z ~ N(0,1).
    2. The candidate with the lowest sampled value is most promising for minimization.
    3. To incorporate cost awareness, we scale the sampled value by a cost factor.
       Specifically, we want to maximize the "value per cost", i.e., minimize
       y_sample / cost, which is equivalent to maximizing -y_sample / cost.
    
    However, Thompson sampling typically picks the argmax of the sampled function.
    For minimization, we pick the argmin of the sampled function.
    
    Cost awareness: divide by cost so that cheaper evaluations are preferred.
    score = -y_sample / cost  (higher = better)
    
    To stabilize, we can also blend with a small exploration term.
    """
    if _check_deadline():
        # Fallback: simple cost-aware EI
        z = np.where(sigma > 1e-9, (f_best - mu) / sigma, 0.0)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return np.nan_to_num(ei / cost_safe, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Thompson sampling: draw a sample from the posterior for each candidate
    # y_sample = mu + sigma * z, z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    y_sample = mu + sigma * z
    
    # For minimization, we want low y_sample.
    # Cost-aware: we prefer candidates with low y_sample / cost.
    # Score = -y_sample / cost (higher is better)
    cost_safe = np.maximum(cost, 1e-9)
    score = -y_sample / cost_safe
    
    # Add a small deterministic exploration bonus based on sigma to avoid
    # purely random behavior in highly uncertain regions
    # This encourages exploring areas with high uncertainty, weighted by cost
    exploration_bonus = 0.1 * sigma / cost_safe
    
    final_score = score + exploration_bonus
    
    # Ensure finite values
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_score
# EVOLVE-BLOCK-END
