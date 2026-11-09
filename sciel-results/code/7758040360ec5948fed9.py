"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for timing
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to leave before the 330s limit

def _check_deadline():
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a dynamic UCB-like strategy.
    
    Instead of simple EI / cost, we use a formulation that explicitly separates
    exploitation (low mu) and exploration (high sigma), weighted by cost.
    
    Score = ( -mu + beta * sigma ) / cost
    
    The parameter beta is adjusted dynamically based on the current budget usage.
    However, since we don't have direct access to the number of iterations or total budget
    within the function signature (only cost array which is per-candidate, not total spent),
    we must infer the stage from the cost distribution or use a fixed, well-tuned beta.
    
    Wait, the 'cost' parameter is the cost of evaluating *each* of the 2000 candidates.
    It is NOT the total cost spent. The total cost is managed by the external loop.
    The external loop stops when total cost > 30.
    
    The key insight from the "different strategy" requirement:
    Standard EI tends to be too greedy. UCB (Upper Confidence Bound) is more exploratory.
    For minimization, we want to maximize: ( -mu + beta * sigma ) / cost.
    
    Why UCB over EI here?
    1. EI requires computing normal CDF/PDF, which is slower.
    2. EI can get stuck in local minima if sigma is underestimated.
    3. UCB with a well-chosen beta allows for better exploration of the domain,
       which is crucial for functions like Rastrigin and Levy which have many local minima.
    
    We will use a hybrid: A cost-weighted UCB.
    To make it "structurally different" and potentially better, we will also add a
    small penalty for points that are very far from the current best in terms of
    predicted value, to avoid wasting budget on obviously bad regions, but primarily
    rely on the UCB structure.
    
    Actually, a more robust structural change is to use **Expected Hypervolume Improvement (EHVI)**
    or **Knowledge Gradient (KG)**, but those are complex.
    
    Let's stick to a refined **Cost-Weighted UCB** with an adaptive beta.
    Since we don't know the iteration count, we can't easily adapt beta based on time left
    unless we track it globally. Let's track the number of calls to `acquisition` to estimate
    progress.
    """
    if _check_deadline():
        # Fallback: simple -mu / cost
        cost_safe = np.maximum(cost, 1e-6)
        return -mu / cost_safe

    # Track number of calls to estimate budget usage
    # We assume the loop runs until cost > 30.
    # Initial 5 points cost ~ 1 + 0.5 = 1.5 each? No, cost is 1 + mean(x_scaled).
    # Average cost per eval is roughly 1.5.
    # Total budget 30. So roughly 20 evaluations.
    # We can track the call index.
    
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    acquisition._call_count += 1
    
    # Estimate fraction of budget used.
    # This is a heuristic. If we've made many calls, we are later in the process.
    # Let's assume ~20-25 calls total.
    # Early on: high exploration (high beta).
    # Late on: low exploration (low beta).
    
    # A simple decay: beta starts at 2.0 and decays to 0.5.
    # Let's say we expect 20 calls.
    expected_calls = 20
    progress = min(1.0, acquisition._call_count / expected_calls)
    
    # Beta decays from 3.0 to 1.0
    beta = 3.0 - 2.0 * progress
    
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-6)
    
    # UCB for minimization: we want to maximize ( -mu + beta * sigma ) / cost
    # -mu is the exploitation term (higher is better, i.e., lower mu)
    # beta * sigma is the exploration term
    # Dividing by cost makes it cost-aware
    
    scores = (-mu + beta * sigma_safe) / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
