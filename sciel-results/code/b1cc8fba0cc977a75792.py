"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Start time for deadline guard
    start_time = time.time()
    
    # Calculate total cost to determine budget consumption
    # The loop runs while total cost < 30
    # Initial cost is 5 points * avg cost ~ 1.5 = 7.5
    # So we have ~22.5 cost units left
    # We can estimate the "progress" by looking at the mean cost of evaluated points?
    # No, we don't have access to the history of evaluated points directly, only mu, sigma, f_best, cost of candidates.
    # However, we can infer the number of evaluations or the total cost if we assume a standard trajectory?
    # Actually, the function signature does NOT provide the total cost.
    # It provides `cost` which is the cost of the *candidates* being scored.
    # The directive says: "kappa is dynamically increased as the total budget is consumed".
    # Since we don't have `total_cost` as an input, we must approximate it or use a fixed schedule.
    # Wait, looking at the task description: "Each function runs on five seeded instances... Runs start with 5 uniform random points... While total cost is below 30..."
    # The function `acquisition` is called inside the loop. It does not receive the current total cost.
    # This implies we cannot dynamically adjust kappa based on the *actual* total cost unless we track it externally or infer it.
    # However, we can approximate the number of iterations.
    # Max cost 30. Min cost 1.0. Max evaluations ~30.
    # Average cost ~1.5. Max evaluations ~20.
    # Since we don't have the iteration count, let's look at the distribution of `cost`.
    # The cost of candidates is fixed by their location.
    # Maybe we can use a fixed kappa that works well on average?
    # Or, we can assume that the optimization process moves towards lower cost regions?
    # Let's try a fixed kappa first, as dynamic adjustment without state is impossible.
    # The previous PI approach was equivalent to a specific type of exploitation.
    # LCB: score = (mu - kappa * sigma) / cost
    # Let's try kappa = 1.0 or 2.0.
    # In standard BO, kappa is often around 1-3.
    # Since we divide by cost, we are penalizing expensive points.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Try a fixed kappa
    # If we want to be more exploratory early and exploitative later, we can't do it without state.
    # But we can use the magnitude of f_best? No, f_best is just a value.
    # Let's stick to a fixed kappa.
    # Let's try kappa = 2.0.
    kappa = 2.0
    
    # LCB score: we want to minimize mu + kappa*sigma, so maximize -(mu + kappa*sigma)
    # Or maximize (f_best - (mu + kappa*sigma))? No, LCB is usually mu - kappa*sigma for minimization?
    # Standard LCB for minimization: minimize mu + kappa * sigma.
    # So score should be - (mu + kappa * sigma).
    # But we also want to account for cost.
    # Score = - (mu + kappa * sigma) / cost?
    # Or Score = (f_best - mu - kappa * sigma) / cost?
    # The prompt says: "maximizes the worst-case performance... scoring candidates with (mu - kappa * sigma) / cost"
    # Wait, for minimization, lower mu is better.
    # If we use (mu - kappa * sigma), lower is better for the numerator part?
    # No, if mu is low, (mu - kappa*sigma) is low. We want HIGH scores.
    # So we should use -mu + kappa * sigma?
    # Let's re-read: "maximizes the worst-case performance... (mu - kappa * sigma) / cost"
    # If the objective is to MINIMIZE f(x), then the "worst case" is high f(x).
    # LCB for minimization is usually defined as minimizing mu + kappa*sigma.
    # If we are maximizing a score, the score should be inversely related to the LCB.
    # Score = - (mu + kappa * sigma) / cost.
    # Let's check the prompt's formula: "(mu - kappa * sigma) / cost".
    # If this is for MAXIMIZATION of f, then LCB is mu - kappa*sigma.
    # But the task is MINIMIZATION.
    # So for minimization, the acquisition function is often EI (Expected Improvement) or LCB (Lower Confidence Bound, where we minimize the bound).
    # If we convert to a maximization problem: Maximize -f(x).
    # Then mu' = -mu, sigma' = sigma.
    # LCB for max: mu' - kappa*sigma' = -mu - kappa*sigma.
    # So we want to maximize (-mu - kappa*sigma) / cost.
    # Let's try this.
    
    scores = (-mu - kappa * sigma) / (cost + eps)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
