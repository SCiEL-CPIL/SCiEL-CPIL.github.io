import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate EI as a baseline for exploitation potential
    # Note: In standard UCB, we use mu + beta*sigma. 
    # However, since we want to minimize, and f_best is the best so far,
    # a common cost-aware approach is to combine an improvement metric with uncertainty.
    # The prompt suggests CA-UCB with adaptive beta.
    # UCB for minimization: Score = -mu + beta * sigma (maximize score to find low mu)
    # Or: Score = (f_best - mu) + beta * sigma ? No, that's like EI without the CDF/PDF weighting.
    
    # Let's stick to the directive: "Cost-Aware Upper Confidence Bound (CA-UCB) 
    # that uses an adaptive beta parameter, where beta is inversely proportional 
    # to the current budget consumption"
    
    # Total budget is 30.
    # current_cost is passed as an array, but the prompt says "cost has shape (2000,)" 
    # and "Evaluating x costs 1 + mean(x scaled to [0, 1])". 
    # Wait, the input `cost` in the function signature is likely the cost of evaluating 
    # each of the 2000 candidates? Or the current total cost?
    # "mu, sigma and cost have shape (2000,)."
    # "Each function runs on five seeded instances... Evaluating x costs 1 + mean(x...)"
    # The loop runs while total cost < 30.
    # The function `acquisition` is called with `cost` of shape (2000,).
    # This implies `cost[i]` is the cost of evaluating candidate i.
    # But we need the *total* cost to determine the adaptive beta.
    # The prompt says: "While total cost is below 30...".
    # It does not pass `total_cost` to the function.
    # However, we can infer the stage of the optimization.
    # The initial 5 points cost roughly 1 + 0.5 = 1.5 each? Total ~7.5.
    # Remaining budget ~22.5.
    # Each subsequent evaluation costs ~1.5.
    # So we have roughly 15-20 evaluations left.
    
    # Since we don't have explicit total_cost, we can approximate the "budget consumption" 
    # or simply use a fixed strategy that mimics the desired behavior.
    # The directive says: "beta is inversely proportional to the current budget consumption".
    # If we assume the function is called repeatedly, and we don't have state, 
    # we cannot track total cost across calls unless we use a global variable or infer it.
    # But wait, `f_best` is the lowest observed value. 
    # As we find better values, `f_best` decreases.
    # Also, the number of points grows, but we don't see N.
    
    # Let's look at the inputs again.
    # mu, sigma, f_best, cost.
    # If we can't track total cost, we can use a heuristic based on the spread of `f_best` 
    # or simply a fixed beta that works well for Hartmann 3D with this budget.
    
    # However, the prompt explicitly asks for "adaptive beta parameter... inversely proportional 
    # to the current budget consumption".
    # Is it possible `cost` array contains the *cumulative* cost up to that point? No, 
    # "Evaluating x costs...". It's the cost of the point x.
    
    # If we cannot track time/state, we might use a static beta that is a compromise.
    # Or, we can assume the "budget consumption" is related to how many points have been 
    # evaluated. But we don't know N.
    
    # Alternative interpretation: Maybe the `cost` passed is not the per-point cost, 
    # but something else? "mu, sigma and cost have shape (2000,)."
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    # So `cost[i]` is the cost of point i.
    
    # Without global state, we cannot know the total cost.
    # However, we can use the fact that `f_best` is a scalar.
    # If `f_best` is very low, we are exploiting. If it's high, we are exploring.
    # But that's not budget consumption.
    
    # Let's try a different angle. The prompt says "Keep the function names and signatures".
    # It does not forbid global variables.
    # We can maintain a global counter or timestamp to estimate progress.
    
    global _start_time
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    # Total time limit is 300s for the whole run? 
    # "Each evaluation must finish within 300 s." 
    # "The evaluator KILLS the program after 330s wall-clock."
    # The loop runs while total cost < 30.
    # Each acquisition call takes some time.
    
    # Actually, a simpler proxy for "late in the game" is that `f_best` has converged 
    # or the variance `sigma` is low.
    # But the directive is specific about budget.
    
    # Let's assume we can't track budget perfectly and use a robust CA-UCB with a 
    # beta that is tuned for this specific problem size (budget 30, 5 initial points).
    # Total points ~ 5 + (30-7.5)/1.5 ~ 20 points.
    # Early on (N=5), we want high exploration. Late on (N=20), low exploration.
    # Since we don't know N, let's use a fixed beta that is slightly lower than 
    # aggressive exploration, as UCB with high beta can be unstable with few points.
    
    # However, the "Inspiration 1" score was 1.0848. The current is 1.0.
    # The current is EI/cost.
    # UCB is: -mu + beta * sigma.
    # Cost-aware UCB: (-mu + beta * sigma) / cost? Or (-mu + beta * sigma) - lambda * cost?
    # Standard cost-aware acquisition often divides by cost.
    
    # Let's try: Score = (f_best - mu + beta * sigma) / cost
    # This combines improvement potential (f_best - mu) with uncertainty (sigma), 
    # normalized by cost.
    # Beta should be adaptive. If we can't track budget, let's use a moderate beta.
    # For Hartmann 3D, beta ~ 1.0-2.0 is common.
    
    # To satisfy the "adaptive" requirement without state, we can use the 
    # standard deviation of sigma or mu as a proxy for "convergence"?
    # No, that's unreliable.
    
    # Let's use a global variable to track the number of calls, which correlates with cost.
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    acquisition._call_count += 1
    calls = acquisition._call_count
    
    # Budget is 30. Initial cost ~7.5. Remaining ~22.5.
    # Each call evaluates 1 point. So we have ~15 calls left.
    # If calls < 5, we are early. If calls > 10, we are late.
    
    # Adaptive beta: Start high, decrease.
    # beta = max(0.1, 2.0 - 0.1 * calls) ?
    # Or beta = 2.0 / (1 + calls/10)
    
    beta = 2.0 / (1 + calls / 10.0)
    
    # CA-UCB Score:
    # We want to maximize the score.
    # Potential improvement: f_best - mu (positive if mu < f_best)
    # Uncertainty: sigma
    # Score = (f_best - mu + beta * sigma) / cost
    
    # This is different from EI which is E[max(f_best - f(x), 0)].
    # UCB is more aggressive in exploration.
    
    delta = f_best - mu
    scores = (delta + beta * sigma_safe) / cost
    
    # Handle NaNs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
