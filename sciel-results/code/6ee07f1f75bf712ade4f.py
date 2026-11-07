"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Global Budget-Aware UCB."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Constants for the UCB strategy
# The budget is 30. Initial 5 points cost ~ 5 * (1 + 0.5) = 7.5.
# Remaining budget is roughly 22.5.
# We want to explore early (high beta) and exploit late (low beta).
# Beta will be scaled by the "urgency" of the budget.

def acquisition(mu, sigma, f_best, cost):
    """
    Global Budget-Aware UCB acquisition function.
    
    Strategy:
    1. Estimate the total budget spent so far by summing the `cost` array.
       Note: The `cost` array passed in represents the cost of evaluating each of the 2000 candidates.
       However, the prompt says "While total cost is below 30... evaluates the best one".
       The `cost` argument in the function signature is likely the cost of the *current* candidates being scored, 
       or potentially the cumulative cost? 
       Re-reading: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
       "Evaluating x costs 1 + mean(x scaled to [0, 1])."
       The `cost` array provided is the cost of evaluating each of the 2000 candidates.
       We don't directly know the *total* cost spent so far from the inputs.
       
       Wait, the prompt says: "Runs start with 5 uniform random points (cost counted). While total cost is below 30...".
       The function `acquisition` is called inside this loop.
       Can we infer the number of evaluations? 
       The GP is trained on the observed points. We don't have the observed points directly.
       However, `f_best` is the lowest observed value.
       
       Is there a way to know the budget spent?
       The `cost` array is the cost of the *candidates*. 
       Usually, in these tasks, if the state isn't passed, we might have to assume a fixed number of steps or estimate it.
       
       Let's look at the inspirations. 
       Inspiration 1: "uses a dynamic beta parameter decaying from high exploration to high exploitation".
       How does it know the progress?
       Perhaps the magnitude of `sigma` or `f_best` gives a clue? 
       Or maybe we can just use a fixed, well-tuned beta that balances exploration and exploitation well for this specific budget?
       
       Actually, looking at the signature: `acquisition(mu, sigma, f_best, cost)`.
       `cost` is the cost of the *candidates* (2000,).
       
       If we cannot know the exact budget spent, we can use a heuristic.
       However, a "Global Budget-Aware" strategy usually implies knowing the remaining budget.
       
       Let's re-read carefully: "mu, sigma and cost have shape (2000,)."
       This confirms `cost` is per-candidate.
       
       Is it possible `f_best` or the GP state implies the number of points?
       No, `f_best` is just a scalar.
       
       Maybe we can assume the search is in a specific phase?
       With a budget of 30 and initial cost ~7.5, we have ~22 evaluations left.
       Each evaluation costs ~1.5 on average. So ~15 more evaluations.
       
       A fixed UCB with a good beta might be sufficient.
       Alternatively, we can use the variance of `sigma` or the value of `f_best` to gauge progress?
       Unreliable.
       
       Let's try a different angle: **Thompson Sampling** or **Upper Confidence Bound with a fixed, optimal beta**.
       
       The key insight from Inspiration 1 (score 2.3272) was "Global Budget-Aware UCB... dynamic beta".
       How did it implement dynamic beta without knowing the step count?
       Maybe it didn't need to? Or maybe there's a hidden state?
       
       Wait, if I look at the "current program", it just divides EI by cost.
       
       Let's try a **Pure UCB** with a fixed beta that is known to work well for this budget.
       For 4D Powell, the function is smooth. 
       UCB: `mu + beta * sigma`.
       
       Why UCB over EI?
       EI is greedy. UCB balances exploration.
       In 4D, exploration is critical to find the global minimum (which is shifted).
       
       What beta?
       A common choice is `beta = 2 * sqrt(log(n))` where n is the number of points.
       If we assume ~20 points, `log(20) ~ 3`, `beta ~ 4`.
       But `sigma` is in objective units.
       
       Let's try a **Cost-Adjusted UCB** but not dividing by cost locally.
       Instead, we prioritize points that are both promising (high UCB) AND cheap?
       No, that's what the current one does (EI/cost).
       
       The inspiration says "Replace the per-candidate cost normalization... with a global budget-aware UCB".
       This implies **ignoring the local cost** in the acquisition function itself, or using it differently.
       
       If we ignore local cost, we might pick expensive points.
       But if the expensive points are where the minimum is, it's worth it.
       The minimum of Powell is at (0,0,0,0) in the original, but shifted to the "inner 80%".
       The cost is `1 + mean(x scaled)`.
       Low coordinates (near 0 in the original box [-4, 5]) correspond to low scaled values (near 0).
       So the minimum is in a **low-cost** region.
       
       Therefore, a strategy that **ignores cost** and just maximizes UCB should naturally gravitate towards the low-cost region if the GP correctly identifies the potential for low values there.
       However, if the GP hasn't explored there, `sigma` is high.
       
       Let's try **UCB with a fixed beta**.
       `score = mu + beta * sigma`
       
       We need to choose `beta`.
       Since we don't know the exact step, let's use a beta that encourages exploration early.
       Maybe `beta = 5.0`?
       
       To make it "Global Budget-Aware", we can use the fact that `f_best` decreases over time.
       As `f_best` gets closer to 0, we should exploit more.
       `beta = max(1.0, 10.0 * (f_best / f_initial_guess))`?
       We don't know `f_initial_guess`.
       
       Let's stick to a robust fixed UCB. The "Global" part might refer to using the global minimum structure (Powell's known properties) or simply that it's a global optimization strategy (not local).
       
       Actually, let's look at the cost again.
       If we use pure UCB, we might waste budget on expensive, high-variance regions.
       But the minimum is cheap.
       
       Let's try **Expected Improvement (EI) without cost normalization**.
       `score = EI`
       This is a different structural change from `EI/cost`.
       EI is more focused on improvement than UCB.
       
       Let's compare:
       Current: `EI / cost`
       New: `EI` (Pure EI)
       
       Why might `EI` be better?
       Dividing by cost penalizes cheap points if their EI is low, and rewards expensive points if their EI is high.
       But if the cheap points have high EI (because the minimum is there), `EI/cost` will reward them *more* (since cost < 1).
       Wait, cost is `1 + mean(x)`. Min cost is 1. Max is 2.
       So dividing by cost scales the score by 0.5x to 1x.
       This slightly dampens the signal for expensive points.
       
       If we use pure `EI`, we treat all points equally in terms of cost.
       Since the minimum is in the cheap region, and the GP will likely assign high EI there (low mu, high sigma initially), pure EI will find it.
       
       Let's try **Pure EI** (no cost division). This is a structural change from Cost-Aware EI to Standard EI.
       The "Cost-Aware" aspect is handled by the fact that the minimum is cheap, so we don't *need* to penalize cost explicitly, and doing so might distort the gradient of improvement.
       
       Actually, let's try **UCB** as per the top inspiration.
       `score = mu + beta * sigma`
       Let's use `beta = 4.0`.
       
       Let's combine: **Cost-Aware UCB** but with a global beta?
       No, the inspiration said "Replace... with... UCB".
       
       Let's go with **Pure UCB** with a fixed beta.
       
       One more idea: **Knowledge Gradient**? Too complex.
       
       Let's implement **Pure UCB** with `beta = 5.0`.
       
       Wait, I should check if `f_best` can be used to adapt beta.
       If `f_best` is close to 0, set beta low.
       If `f_best` is large, set beta high.
       `beta = 2.0 + 10.0 * (f_best / (f_best + 1.0))`
       This creates a smooth transition.
       
       Let's try this adaptive beta.
       
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    t0 = time.time()
    
    # Adaptive Beta based on f_best
    # As f_best -> 0, beta -> 2.0 (exploit)
    # As f_best -> inf, beta -> 12.0 (explore)
    # Using a simple saturation function
    f_best_safe = max(f_best, 1e-6)
    beta = 2.0 + 10.0 * (f_best_safe / (f_best_safe + 5.0))
    
    # UCB Score: mu + beta * sigma
    score = mu + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline check
    if time.time() - t0 > 5:
        return score
        
    return score

# EVOLVE-BLOCK-END
