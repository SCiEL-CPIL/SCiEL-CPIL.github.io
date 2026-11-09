"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: track elapsed time to avoid exceeding the 330s limit
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds to spare before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Strategy:
    - Uses UCB instead of EI to encourage global exploration.
    - UCB = mu + kappa * sigma - (f_best - mu) is not standard UCB for minimization.
    - For minimization, we want low mu and high sigma.
    - Standard UCB for minimization: Score = -mu + kappa * sigma.
    - However, we must account for cost.
    - Cost-aware UCB: Score = (-mu + kappa * sigma) / cost.
    - We also add a small term to favor points that are likely to improve upon f_best significantly.
    
    The Levy function is multi-modal. EI can get stuck in local minima.
    UCB with a sufficiently large kappa explores the space more broadly.
    
    kappa is adapted based on the remaining budget.
    Total budget is 30. Initial cost is roughly 5 * (1 + 0.5) = 7.5 (approx).
    Let's estimate remaining budget dynamically if possible, but we only have current cost.
    Wait, 'cost' in the signature is the cost of the *candidate* points, not the total spent.
    The prompt says: "While total cost is below 30...". The function receives 'cost' for the 2000 candidates.
    It does NOT receive the total spent so far.
    However, we can infer the stage of the search if we track state, but we must be stateless or use a global variable.
    Let's use a simple heuristic for kappa.
    
    In the early stages, we want high exploration (high kappa).
    In the late stages, we want high exploitation (low kappa).
    Since we don't know the total spent, we can't perfectly adapt.
    However, the GP's sigma will naturally decrease as we add points.
    A fixed kappa might work, but an adaptive one is better.
    
    Let's look at the inputs: mu, sigma, f_best, cost.
    f_best is the current best.
    
    Let's try a hybrid:
    Score = ( (f_best - mu) + kappa * sigma ) / cost
    This is similar to EI but without the non-linear saturation of the normal CDF/PDF.
    It's linear in mu and sigma.
    
    Let's refine the UCB for minimization:
    We want to minimize f(x).
    Lower bound on f(x) is roughly mu - kappa*sigma.
    Upper bound on improvement is f_best - (mu - kappa*sigma) = f_best - mu + kappa*sigma.
    So, Expected Improvement proxy = f_best - mu + kappa*sigma.
    
    Cost-aware score = (f_best - mu + kappa*sigma) / cost.
    
    We need to choose kappa.
    If kappa is too high, we explore random points.
    If kappa is too low, we exploit the GP mean (greedy).
    
    Given the budget is small (30 units), and we start with 5 points, we have ~25 units left.
    Average cost per eval is ~1.5 (if uniform in [0,1] scaled, mean cost 1.5).
    So ~16-17 evaluations.
    This is a very small budget.
    
    With such a small budget, broad exploration is crucial to find the global minimum of Levy 2D.
    The initial 5 points are uniform.
    
    Let's use a time-based or call-count based adaptation if we can track calls.
    We can use a global counter.
    
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330 - _DEADLINE_MARGIN:
        # Fallback: pure exploration weighted by inverse cost
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        scores = sigma_safe / cost_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure safety
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Simple stateful adaptation: track number of calls
    # This helps distinguish early vs late phase
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    
    acquisition._call_count += 1
    calls = acquisition._call_count
    
    # Total budget 30.
    # Initial 5 points cost approx 5 * 1.5 = 7.5.
    # Remaining approx 22.5.
    # Each acquisition call represents one evaluation.
    # So we have about 15 evaluations.
    
    # Decay kappa from high to low.
    # Start with kappa = 2.0 (strong exploration)
    # End with kappa = 0.5 (exploitation)
    
    # Map calls 1..15 to kappa 2.0..0.5
    # If calls > 15, keep it low.
    max_expected_calls = 15.0
    progress = min(calls / max_expected_calls, 1.0)
    
    # Linear decay
    kappa = 2.0 - 1.5 * progress
    
    # Calculate Score
    # Improvement term: f_best - mu
    # Exploration term: kappa * sigma
    # Combined: f_best - mu + kappa * sigma
    
    # Note: f_best is a scalar. mu is array.
    improvement_potential = f_best - mu
    exploration_term = kappa * sigma
    
    combined = improvement_potential + exploration_term
    
    # Cost awareness
    scores = combined / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
