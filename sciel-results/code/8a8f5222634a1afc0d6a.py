"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: keep a safety margin before the 300 s kill.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20 s margin


def _time_guard():
    """Return True if we are within the safe time budget."""
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses GP-UCB with budget-scaled exploration coefficient.
    Returns 2000 finite scores; higher = evaluate next.
    """
    # Deadline guard: if running out of time, return a simple valid fallback.
    if not _time_guard():
        # Fallback: favor cheap, uncertain points
        c = np.maximum(cost, 1e-9)
        s = np.maximum(sigma, 1e-9)
        return s / c

    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Budget-scaled exploration coefficient for UCB.
    # Total budget is 30, starting cost is ~5 (5 uniform random points).
    # As cost approaches 30, exploration should decay to favor exploitation.
    # Use a linear decay from high to low.
    # Let's say at cost=5, kappa=1.0, at cost=30, kappa=0.1
    # This is a structural change from the fixed EI + small explore term.
    
    # Calculate progress: 0 at start (cost~5), 1 at end (cost~30)
    # But cost is the total cost so far.
    # Let's normalize cost.
    # The loop runs while total cost < 30.
    # So cost ranges from ~5 to 30.
    
    # We want kappa to be high early, low late.
    # kappa = kappa_max - (kappa_max - kappa_min) * (cost - cost_min) / (cost_max - cost_min)
    # Let's use kappa_max = 1.5, kappa_min = 0.2
    # cost_min = 5, cost_max = 30
    
    cost_min = 5.0
    cost_max = 30.0
    kappa_max = 1.5
    kappa_min = 0.2
    
    # cost is an array, but it's the total cost so far?
    # Wait, the contract says "cost has shape (2000,)".
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # "Runs start with 5 uniform random points (cost counted)."
    # "While total cost is below 30..."
    # So `cost` passed to acquisition is the cost of *each candidate*?
    # Or is it the *cumulative* cost?
    # "mu, sigma and cost have shape (2000,)."
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # This implies `cost` is the cost of evaluating each of the 2000 candidates.
    # So `cost` varies per candidate.
    
    # But the budget is on *total* cost.
    # The acquisition function doesn't know the total cost directly?
    # Wait, the signature is `acquisition(mu, sigma, f_best, cost)`.
    # It doesn't take `total_cost`.
    # However, the task says "Cost-aware acquisition function".
    # Usually, cost-awareness means dividing by the cost of that specific evaluation.
    # The directive says: "budget-scaled exploration coefficient that decays linearly from high to low as total cost approaches 30".
    # How do we know total cost?
    # Maybe we can infer it? No, it's not passed.
    # Maybe the `cost` array contains the cumulative cost? No, "Evaluating x costs...".
    # Let's re-read carefully.
    # "mu, sigma and cost have shape (2000,)."
    # "Return 2000 finite scores".
    # "cost" is likely the cost of evaluating each candidate point.
    # If we don't have total cost, we can't scale kappa based on total cost directly.
    # Unless... the "cost" passed is actually the *remaining* budget? No.
    # Or maybe the "cost" is the *cumulative* cost up to that point? No, shape (2000,).
    
    # Let's look at the current code: `score_ei = ei / c`.
    # This uses the per-candidate cost.
    # The directive suggests using total cost.
    # Is it possible that `f_best` or something else encodes total cost? No.
    # Perhaps I should just use the per-candidate cost for the division, and use a fixed kappa?
    # But the directive specifically says "decays linearly from high to low as total cost approaches 30".
    # If I don't have total cost, I can't do that.
    
    # Let's check if there's any other info.
    # "Runs start with 5 uniform random points".
    # Maybe the number of iterations can be estimated?
    # No, the function is stateless.
    
    # Alternative interpretation:
    # Maybe the "cost" array *is* the cumulative cost?
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # If `cost` is shape (2000,), it's likely the marginal cost of each candidate.
    
    # If I cannot access total cost, I cannot implement the specific "budget-scaled" decay based on total cost.
    # However, I can implement a UCB with a fixed kappa, or a kappa that depends on the *marginal* cost?
    # No, that doesn't make sense for "budget-scaled".
    
    # Let's assume the directive implies that I *should* have access to total cost, or that I can approximate it.
    # But I can't.
    
    # Let's re-read the directive: "Apply Gaussian Process UCB ... with a budget-scaled exploration coefficient that decays linearly from high to low as total cost approaches 30".
    # This is a specific instruction. If I can't do it, maybe I should use a fixed UCB?
    # Or maybe I can estimate total cost from the number of evaluations?
    # The function is called every iteration.
    # I could use a global counter?
    # "Keep the function names and signatures of the seed program."
    # I can add global variables.
    
    # Let's use a global counter to estimate the number of evaluations.
    # Each iteration evaluates 1 point.
    # Initial: 5 points.
    # So, eval_count starts at 5.
    # Each call to acquisition increments the count?
    # No, the loop calls acquisition, then evaluates 1 point.
    # So I can count the calls to acquisition.
    
    global _eval_count
    if not hasattr(acquisition, '_eval_count'):
        acquisition._eval_count = 5  # Initial 5 points
    
    # Increment count?
    # The function is called *before* evaluation.
    # So the count of *past* evaluations is acquisition._eval_count.
    # The total cost is not just eval_count, because cost varies.
    # But we can estimate it. Average cost is roughly 1.5 (1 + 0.5).
    # So total_cost_est = acquisition._eval_count * 1.5.
    
    # Let's try this.
    total_cost_est = acquisition._eval_count * 1.5
    
    # Now scale kappa.
    # cost_min = 5 * 1.5 = 7.5? No, initial 5 points cost is counted.
    # Let's just use the estimated total cost.
    
    # If total_cost_est < 7.5 (approx start), kappa = 1.5
    # If total_cost_est > 30, kappa = 0.2
    
    # Let's refine the estimation.
    # Actually, the cost is 1 + mean(x). x is in [-5, 5].
    # Scaled to [0, 1], mean is 0.5.
    # So cost is 1 + 0.5 = 1.5 on average.
    
    # Let's use the estimated total cost.
    
    # Calculate progress
    # Start cost ~ 5 * 1.5 = 7.5
    # End cost 30
    
    # Let's just use the raw count.
    # n_evals = acquisition._eval_count
    # If n_evals < 10, kappa = 1.5
    # If n_evals > 20, kappa = 0.2
    
    # Let's stick to the cost-based scaling if possible, but since we don't have exact cost,
    # let's use the count.
    
    # Actually, the directive says "as total cost approaches 30".
    # Let's assume the average cost is 1.5.
    # n_max = 30 / 1.5 = 20.
    # So we expect about 20 evaluations.
    
    # Let's use n_evals to scale.
    n_evals = acquisition._eval_count
    n_start = 5
    n_end = 20
    
    # Linear decay
    if n_evals <= n_start:
        kappa = 1.5
    elif n_evals >= n_end:
        kappa = 0.2
    else:
        frac = (n_evals - n_start) / (n_end - n_start)
        kappa = 1.5 - frac * (1.5 - 0.2)
    
    # UCB: mu + kappa * sigma
    # For minimization, we want to maximize the "upper bound" of improvement?
    # No, UCB for minimization: we want to find low values.
    # The "confidence bound" is mu - kappa * sigma?
    # No, that's LCB (Lower Confidence Bound).
    # UCB (Upper Confidence Bound) is typically for maximization.
    # For minimization, we use LCB: mu - kappa * sigma.
    # Higher LCB means more promising (lower expected value with high confidence).
    # Wait, if we minimize f, we want f to be small.
    # The lower bound of f is mu - kappa * sigma.
    # We want to maximize this lower bound?
    # No, if the lower bound is high, it means f is likely high.
    # We want f to be low.
    # So we want to minimize the upper bound?
    # Let's think.
    # We have a belief that f(x) is in [mu - k*sigma, mu + k*sigma].
    # We want to find a point where f is low.
    # The most "promising" point is one where the *lowest* possible value is as low as possible?
    # Or one where the *expected* value is low?
    # Standard UCB for maximization: maximize mu + k*sigma.
    # Standard LCB for minimization: minimize mu - k*sigma.
    # So we want to *minimize* mu - k*sigma.
    # Therefore, the score should be *negative* of (mu - k*sigma), i.e., -mu + k*sigma.
    # Or simply, we want to select the point with the *lowest* LCB.
    # So score = -(mu - kappa * sigma) = -mu + kappa * sigma.
    
    # Let's verify.
    # If mu is low, -mu is high. Good.
    # If sigma is high, kappa*sigma is high. Good (exploration).
    # So score = -mu + kappa * sigma.
    
    # Now, cost-awareness.
    # Divide by cost.
    # score = (-mu + kappa * sigma) / c
    
    # But wait, if mu is large (bad), -mu is very negative.
    # If mu is small (good), -mu is large positive.
    # This seems correct.
    
    # Let's implement this.
    
    lcb = mu - kappa * s
    score = -lcb / c
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Increment the counter
    acquisition._eval_count += 1
    
    return score
# EVOLVE-BLOCK-END
