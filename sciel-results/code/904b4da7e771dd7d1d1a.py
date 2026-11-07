"""Rewritten program using cost-aware Adaptive UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe random-like score
    # to allow the loop to finish or fail gracefully without hanging.
    if _is_deadline_exceeded():
        # Return slightly random scores to break ties, or just zeros
        # Zeros are safe because the evaluator picks argmax. 
        # If all are 0, it picks index 0.
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # Estimate the number of evaluations performed so far.
    # The problem states: "Runs start with 5 uniform random points (cost counted). 
    # While total cost is below 30, the evaluator scores 2000 new uniform candidates 
    # with acquisition and evaluates the best one."
    # We don't have the exact number of previous evaluations passed in, 
    # but we can infer it from the total cost spent.
    # Let N be the number of evaluations.
    # Total Cost C = Sum(cost_i).
    # Since we are minimizing, and cost is 1 + mean(x), the average cost is roughly 1.5.
    # So N approx C / 1.5.
    # However, f_best is provided. We can use a heuristic based on how "good" f_best is?
    # No, simpler: The loop runs while total cost < 30.
    # The input `cost` is the cost of the *current* candidates.
    # We need an estimate of N to adapt beta.
    # Let's assume the caller provides `cost` as the cost of the candidates being scored.
    # We don't have the *history* of costs. 
    # But we can approximate N from the magnitude of f_best? No.
    # Let's look at the signature: acquisition(mu, sigma, f_best, cost).
    # It's possible `cost` is just the cost vector for the 2000 candidates.
    # We need to estimate how many steps have been taken.
    # A common heuristic in such contracts where state isn't passed is to use a fixed beta 
    # or derive it from the spread of sigma/mu?
    # Actually, the "Adaptive" part in the successful island likely refers to 
    # beta scaling with the number of iterations.
    # Since we don't have iteration count, we can use a proxy:
    # If f_best is very low (close to min -3.86), we are likely late in the game.
    # If f_best is high, we are early.
    # Min value is -3.86278. Max value in [0,1]^3 for Hartmann 3D is around 3.86 (actually max is lower, but let's say range is ~8).
    # Let's define a normalized progress p in [0, 1].
    # f_min = -3.86, f_max = 3.86 (approx).
    # p = (f_max - f_best) / (f_max - f_min).
    # If f_best is -3.86, p=1. If f_best is 3.86, p=0.
    # This is a rough estimate of convergence.
    
    f_min_expected = -3.86278
    f_max_expected = 3.86 # Rough upper bound for Hartmann 3D on [0,1]^3
    
    # Normalize f_best to a progress metric [0, 1]
    # If f_best is very bad (high), progress is low -> explore more.
    # If f_best is very good (low), progress is high -> exploit more.
    progress = (f_max_expected - f_best) / (f_max_expected - f_min_expected)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive Beta:
    # Start with high exploration (beta ~ 2.0) and decay to low exploration (beta ~ 0.5).
    # beta = beta_max - (beta_max - beta_min) * progress
    beta_max = 2.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # UCB Formula for minimization:
    # We want to minimize f(x).
    # GP predicts mean mu and std sigma.
    # Lower mu is better. Higher sigma is uncertain.
    # UCB for minimization (often called LCB - Lower Confidence Bound for the objective, 
    # but in acquisition context, we maximize the "potential for improvement" or "regret reduction").
    # Standard UCB for maximization: mu + beta * sigma.
    # For minimization, we want to find points where the lower bound of the function is low.
    # Lower Bound = mu - beta * sigma.
    # We want to MAXIMIZE the acquisition score.
    # If we use the Lower Bound as the score, we are looking for points that *could* be very low.
    # However, if mu is high, mu - beta*sigma might still be high.
    # Alternative view: 
    # Expected Improvement (EI) is standard.
    # UCB acquisition usually takes the form: Score = mu - beta * sigma (for minimization).
    # Wait, if we maximize Score, and Score = mu - beta*sigma, we prefer low mu and high sigma.
    # This is correct for exploration.
    
    ucb_score = mu - beta * sigma_safe
    
    # Cost-awareness:
    # We want to maximize the benefit per cost.
    # Since we are minimizing the function, a "lower" ucb_score is better.
    # But the acquisition function must return scores where HIGHER is better.
    # So we should invert the ucb_score or use a regret-based metric.
    # Let's use: Score = - (mu - beta * sigma) / cost
    # Or better, relative to f_best?
    # If we just use -mu/cost, it's pure exploitation.
    # The term -beta*sigma/cost encourages exploring high uncertainty areas if the cost is low.
    
    # Let's refine the score to be:
    # Score = (f_best - mu + beta * sigma) / cost ?
    # If mu < f_best, (f_best - mu) is positive (good).
    # If sigma is high, we add a bonus (good for exploration).
    # This looks like a variant of EI but with a linear uncertainty term instead of the complex EI integral.
    # This is effectively a "Linear UCB" or "Thompson Sampling" proxy.
    
    # Let's try: Score = (f_best - mu + beta * sigma) / cost
    # This ensures that points with low mu (good) and high sigma (uncertain) get high scores.
    # Dividing by cost penalizes expensive points.
    
    scores = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Final safety: replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are not negative? 
    # If mu > f_best + beta*sigma, the score is negative.
    # Negative scores are fine as long as they are finite, but 0 is a safer "do nothing" score.
    # However, if all scores are negative, argmax picks the least negative one, which is the best available.
    # So we don't strictly need to clip to 0, but it prevents selecting a point that is *worse* than f_best 
    # if the uncertainty is low.
    # Actually, if mu > f_best, the expected value is worse. 
    # The term beta*sigma might save it if uncertainty is high.
    # If the score is negative, it means we expect the value to be worse than f_best even accounting for uncertainty.
    # It is safer to clip to 0 to avoid wasting budget on clearly bad points?
    # No, if all points are bad, we still have to pick one. 
    # The evaluator picks argmax. If all are negative, it picks the one closest to 0 (best).
    # So we leave them as is, just ensuring finiteness.
    
    return scores

# EVOLVE-BLOCK-END
