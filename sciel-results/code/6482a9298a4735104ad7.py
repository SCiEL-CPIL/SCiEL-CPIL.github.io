"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 5.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a simple fallback
    if _time_exceeded():
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    n = len(mu)
    
    # Strategy: Probabilistic Best-First Search (P-BFS) Approximation
    #
    # The current program uses LCB (mu - beta*sigma) / cost^2.
    # This is a greedy, myopic approach.
    #
    # New Strategy:
    # We model the remaining budget as a resource to be spent on future evaluations.
    # Let R be the estimated remaining budget.
    # The cost of the current candidate is c_i.
    # The remaining budget after evaluating i is R - c_i.
    # We want to maximize the probability that the *final* best value (after spending R-c_i)
    # is lower than f_best.
    #
    # Approximation:
    # 1. Estimate Remaining Budget R.
    #    The total budget is 30.
    #    Initial 5 points cost approx 5 * (1 + 0.5) = 7.5 (avg cost 1.5 for uniform [0,1] scaled).
    #    Actually, cost = 1 + mean(x_scaled). Uniform x in [0,1] has mean 0.5.
    #    So initial cost ~ 5 * 1.5 = 7.5.
    #    Current total cost is unknown directly, but we can infer it from the number of steps?
    #    No, we don't have step count.
    #    However, we can assume the evaluator has spent some cost C_spent.
    #    The `cost` array provided is for the *candidates* being scored.
    #    We don't know C_spent.
    #    
    #    Let's assume the "effective" remaining budget is proportional to the inverse of the 
    #    "urgency" or simply use a fixed heuristic based on the typical cost.
    #    If we assume the average cost of an evaluation is ~1.5, and total budget is 30,
    #    we have ~20 evaluations total. 5 are done. ~15 remain.
    #    Remaining budget R ~ 15 * 1.5 = 22.5? 
    #    But we don't know how many steps have passed.
    #    
    #    Alternative: Use the cost of the candidates themselves to guide the decision.
    #    If we pick a point with cost c, we spend c.
    #    The "value" of that point is not just its immediate EI, but how it affects the future.
    #    
    #    Let's use a "Discounted EI" where the discount factor depends on the cost.
    #    High cost points are "expensive" and leave less budget for future exploration.
    #    
    #    Score_i = EI_i * (1 - c_i / R_total)^alpha
    #    where R_total is the total budget (30).
    #    This penalizes expensive points heavily if they are close to the budget limit.
    #    But we don't know current budget usage.
    #    
    #    Let's try a different angle: **Cost-Effective Uncertainty Reduction**.
    #    The goal is to find the global minimum.
    #    The GP provides mu and sigma.
    #    The "potential" of a point is determined by how low it *could* be (mu - k*sigma)
    #    and how *cheap* it is to find out.
    #    
    #    Current LCB approach: (mu - beta*sigma) / cost^2.
    #    This favors points that are likely low AND very cheap.
    #    
    #    New Idea: **Bayesian Regret Minimization with Cost Constraint**.
    #    Expected Regret = E[f(x) - f_min_global].
    #    We don't know f_min_global. We use f_best as a proxy.
    #    Expected Improvement (EI) is a lower bound on the reduction in regret.
    #    
    #    Let's combine EI and LCB.
    #    EI is good for exploitation. LCB is good for exploration.
    #    Weighted sum: w * EI + (1-w) * (-LCB).
    #    Then divide by cost^gamma.
    #    
    #    But the directive asks for a *materially different* design family.
    #    LCB and EI are both deterministic acquisition functions.
    #    
    #    What about **Thompson Sampling** logic adapted for cost?
    #    In TS, we sample f(x) ~ N(mu, sigma^2) and pick the argmin.
    #    Here, we must return a score for all 2000 points.
    #    We can compute the *expected rank* or *expected improvement in rank*.
    #    
    #    Let's try **Expected Hypervolume Improvement (EHI)** approximation.
    #    Hypervolume is a standard multi-objective metric.
    #    Here, we have two objectives: Minimize f(x) and Minimize Cost.
    #    Reference point: (f_best, 0).
    #    For a candidate (f_i, c_i), the hypervolume improvement is the area dominated by (f_i, c_i)
    #    but not by (f_best, 0) or other non-dominated points.
    #    Since we only pick one point, we approximate the contribution.
    #    
    #    Actually, a simpler multi-objective proxy is the **Scalarized Pareto Score**.
    #    Score = - (f_best - mu) * P(f(x) < f_best) / (1 + cost)^gamma
    #    
    #    Let's look at the inspiration: "Pareto-frontier-based cost-aware score".
    #    
    #    Let's implement a **Stochastic Best-First** score.
    #    We estimate the probability that this point will be the *best* among the remaining budget.
    #    
    #    Approximation:
    #    1. Calculate "Optimistic Value" V_i = mu_i - kappa * sigma_i.
    #    2. Calculate "Cost Penalty" P_i = cost_i^gamma.
    #    3. Score = V_i / P_i.
    #    This is essentially LCB / cost^gamma.
    #    
    #    To be materially different, let's use **Monte Carlo Tree Search (MCTS)** style value estimation.
    #    But we can't run MCTS in the acquisition function for 2000 points quickly.
    #    
    #    Let's try **Information Gain (IG)** based acquisition.
    #    IG measures how much the entropy of the posterior decreases.
    #    For a GP, IG is related to the reduction in variance.
    #    Points far from existing data have high IG.
    #    But we don't have the locations of existing data.
    #    
    #    Wait, the inputs are mu, sigma, cost. We don't have x.
    #    So we can't compute distances to existing points.
    #    We must rely on mu, sigma, and cost.
    #    
    #    This limits us to functions of (mu, sigma, cost).
    #    
    #    Current: (mu - 2*sigma) / cost^2.
    #    
    #    Alternative 1: **Normalized EI**.
    #    EI / sigma. This measures "how many sigmas of improvement".
    #    Score = (EI / sigma) / cost^gamma.
    #    
    #    Alternative 2: **Probability of Improvement (PI) with Cost**.
    #    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    #    Score = PI / cost^gamma.
    #    This favors points that are *likely* to be better, regardless of how much better.
    #    It is more robust to outliers in mu.
    #    
    #    Alternative 3: **Hybrid PI + EI**.
    #    Score = (PI * EI) / cost^gamma.
    #    
    #    Let's test **Alternative 2 (PI / cost^gamma)** with a high gamma.
    #    Why? Because EI can be dominated by points with very low mu but high cost.
    #    PI is bounded [0,1]. It normalizes the "chance" of improvement.
    #    Dividing by cost^gamma (gamma=3) heavily favors cheap points that have a decent chance of being better.
    #    This aligns with the directive: "aggressively exploring unexplored, cheap parts".
    #    Unexplored parts have high sigma, which increases PI if mu is close to f_best.
    #    If mu is far above f_best, PI is low.
    #    
    #    Let's refine:
    #    z = (f_best - mu) / sigma
    #    PI = norm.cdf(z)
    #    
    #    If sigma is very small, z is large (if mu < f_best) or small (if mu > f_best).
    #    If mu < f_best, PI -> 1. Score = 1 / cost^gamma.
    #    If mu > f_best, PI -> 0. Score -> 0.
    #    
    #    This effectively ignores points that are *known* to be worse than f_best (mu > f_best + threshold).
    #    It focuses on points that *might* be better.
    #    
    #    However, PI doesn't capture *how much* better.
    #    A point with mu = f_best - 0.01 and sigma = 0.01 has PI ~ 0.5.
    #    A point with mu = f_best - 1.0 and sigma = 0.1 has PI ~ 1.0.
    #    The second is much better.
    #    
    #    So we should combine PI with a measure of potential magnitude.
    #    Let's use **Expected Improvement (EI)** but normalized by sigma to make it scale-invariant?
    #    EI/sigma is the "z-score of improvement".
    #    
    #    Let's try: Score = (EI / sigma) / cost^gamma.
    #    EI/sigma = z * Phi(z) + phi(z).
    #    This function is maximized when z is around 0.8.
    #    It goes to 0 as z -> -inf (mu >> f_best).
    #    It goes to 0 as z -> +inf (mu << f_best, sigma -> 0). Wait.
    #    If sigma -> 0 and mu < f_best, EI -> f_best - mu. EI/sigma -> inf.
    #    This would favor points with very small sigma and mu < f_best.
    #    These are "exploit" points.
    #    
    #    To encourage exploration, we want to reward high sigma.
    #    LCB (mu - beta*sigma) rewards high sigma.
    #    
    #    Let's stick to the **LCB** family but change the cost penalty structure significantly.
    #    Instead of cost^2, let's use **Exponential Cost Penalty**.
    #    Score = (mu - beta*sigma) * exp(-lambda * cost).
    #    
    #    This is materially different from polynomial penalty.
    #    It creates a "hard" cutoff for expensive points.
    #    
    #    Let's implement:
    #    Score = (mu - beta*sigma) * exp(-lambda * cost)
    #    
    #    Parameters:
    #    beta = 2.0 (standard UCB/LCB)
    #    lambda = 1.0 (tunable, but 1.0 is a good start for cost in [1, 2])
    #    
    #    If cost = 1, exp(-1) = 0.36.
    #    If cost = 2, exp(-2) = 0.13.
    #    Ratio: 0.36 / 0.13 = 2.76.
    #    With cost^2: 1^2 / 2^2 = 0.25 (ratio 4).
    #    With cost^3: 1^2 / 2^3 = 0.125 (ratio 8).
    #    
    #    Exponential penalty is stronger for high costs.
    #    
    #    Let's try **Exponential Cost Penalty**.

    beta = 2.0
    lambda_cost = 1.5  # Stronger penalty
    
    lcb = mu - beta * sigma
    
    # Apply exponential cost penalty
    # cost is in [1, 2] typically (1 + mean(x_scaled))
    # exp(-1.5 * 1) = 0.22
    # exp(-1.5 * 2) = 0.05
    # Ratio ~ 4.4
    
    score = lcb * np.exp(-lambda_cost * cost)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score

# EVOLVE-BLOCK-END
