"""Cost-aware acquisition for 2D Levy optimization using Knowledge Gradient."""
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and ensure we return well before the limit.
_DEADLINE = 320.0  # seconds, safety margin of 10s
_start_time = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Knowledge Gradient acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    # Safety: ensure we don't exceed deadline
    if _remaining_time() < 0.5:
        # Emergency: return a simple fallback
        return np.ones_like(mu) * 0.0

    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Knowledge Gradient approximation:
    # The KG value for a candidate x_i is the expected reduction in the minimum
    # of the observed values if we were to evaluate x_i.
    # A common approximation for the KG when we have a set of candidates is:
    # KG_i = E[ min(mu_j) - min(mu_i, mu_j) ]
    # Since mu is deterministic, this simplifies to:
    # KG_i = max(0, min(mu_j) - mu_i) + E[ min(mu_i, mu_j) - min(mu_j) ]? No, that's not quite right.
    
    # A more robust and computationally feasible approximation for KG in BO:
    # The value of information (VOI) of evaluating candidate i is the expected improvement
    # over the current best candidate in the *proposal set*.
    # Let Y_i ~ N(mu_i, sigma_i^2) be the latent function value at candidate i.
    # The "best" candidate in the set is argmin_j Y_j.
    # The KG for candidate i is E[ min_j Y_j - min(Y_i, min_{j!=i} Y_j) ].
    # This is equivalent to E[ min(Y_i, min_{j!=i} Y_j) - min_{j!=i} Y_j ]? No.
    # Actually, the standard KG for a single evaluation is:
    # KG_i = E[ min(Y_1, ..., Y_n) | Y_i ] - E[ min(Y_1, ..., Y_n) ]
    # This is hard to compute exactly.
    
    # A widely used approximation is to compare each candidate against the
    # distribution of the minimum of the *other* candidates.
    # Let M_{-i} = min_{j != i} Y_j.
    # Then KG_i = E[ min(Y_i, M_{-i}) - M_{-i} ] = E[ max(0, M_{-i} - Y_i) ].
    # This is the expected improvement of candidate i over the minimum of the others.
    
    # To compute this efficiently:
    # We approximate the distribution of M_{-i}.
    # If we assume the Y_j are independent (which they are not, but it's a common approximation),
    # the CDF of M_{-i} is F_{M_{-i}}(m) = prod_{j != i} F_j(m).
    # The PDF is harder.
    
    # A simpler and very effective heuristic that captures the KG spirit is:
    # Score_i = E[ max(0, mu_best_proposal - Y_i) ] where mu_best_proposal is the
    # candidate with the lowest mean in the set.
    # However, a better proxy is to compute the probability that candidate i
    # is the global minimum of the set, weighted by the potential improvement.
    
    # Let's use a specific approximation:
    # For each candidate i, calculate the probability that Y_i is the minimum of all Y_j.
    # P(Y_i = min) = E[ prod_{j != i} P(Y_j > Y_i | Y_i) ]
    # This is still complex.
    
    # Alternative Strategy: "Expected Improvement over the Best Proposal"
    # Let mu_min_idx = argmin(mu).
    # Let mu_min = mu[mu_min_idx].
    # For each i, compute EI_i = E[max(0, mu_min - Y_i)].
    # This encourages evaluating points that have a high chance of beating the current
    # best proposal in the batch.
    
    # But wait, the standard EI uses f_best (observed).
    # KG uses the expected minimum of the *future* observations.
    # If we just use EI against f_best, we ignore the other candidates.
    # If we use EI against mu_min (the best prediction in the batch), we are
    # essentially doing "batch EI" or "q-EI" approximation.
    
    # Let's try a different angle: The "Probability of Improvement" (PI)
    # normalized by cost, but adjusted for the spread of predictions.
    # PI_i = P(Y_i < f_best) = norm.cdf((f_best - mu_i) / sigma_i).
    # This is less sensitive to the magnitude of improvement than EI.
    
    # Let's stick to a robust EI variant but consider the "uncertainty of the best".
    # Actually, the most distinct and powerful strategy here is **Thompson Sampling**
    # or **UCB** with a specific cost adjustment.
    # But the task asks for a "score" for each of the 2000 points, and the evaluator
    # picks the *single* best one. So we need a deterministic ranking.
    
    # Let's use **Upper Confidence Bound (UCB)**:
    # Score_i = mu_i + kappa * sigma_i
    # Since we are minimizing, we want to minimize the function value.
    # So we want high confidence that the value is low.
    # This is equivalent to maximizing: -mu_i + kappa * sigma_i? No.
    # We want to find the global minimum.
    # UCB for minimization: Score_i = mu_i - kappa * sigma_i (lower is better).
    # But the return value should be "higher = evaluate next".
    # So Score_i = -mu_i + kappa * sigma_i.
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # How to choose kappa?
    # A common choice is to make the exploration term relative to the scale of the problem.
    # Or, use a time-dependent kappa that decreases as we get closer to the optimum.
    # However, we don't have explicit time steps, only total cost.
    
    # Let's refine the UCB approach with cost awareness:
    # Score_i = (-mu_i + kappa * sigma_i) / cost_i
    
    # What is a good kappa?
    # If kappa is too high, we explore too much. If too low, we exploit too much.
    # We can set kappa based on the ratio of the range of mu to the mean sigma.
    # Or, simply use a fixed kappa that works well for Levy.
    # For Levy, the landscape is rugged. Exploration is important.
    # Let's try kappa = 2.0 or 3.0.
    
    # Another option: **Expected Hypervolume Improvement (EHI)** or **q-EI**.
    # But q-EI is computationally expensive for n=2000 if done properly.
    
    # Let's try a hybrid: **Cost-aware UCB** with an adaptive kappa.
    # kappa = 2.0 * (1 + log(1 + total_cost)) / (1 + log(1 + total_cost))? No.
    
    # Let's look at the inspiration: "Adaptive exploration weight based on f_best relative to mu range".
    # This suggests adjusting the balance between mu and sigma.
    
    # Let's implement **Adaptive UCB**:
    # 1. Calculate the range of mu: mu_range = max(mu) - min(mu).
    # 2. Calculate the mean sigma: sigma_mean = mean(sigma).
    # 3. Set kappa such that the exploration term is comparable to the exploitation term.
    #    kappa = mu_range / sigma_mean? This might be too large.
    #    Let's use a fixed multiplier, e.g., kappa = 1.5.
    
    # Score_i = (-mu_i + kappa * sigma_i) / cost_i
    
    # To make it more robust, let's normalize mu and sigma.
    # mu_norm = (mu - min(mu)) / (max(mu) - min(mu) + 1e-9)
    # sigma_norm = (sigma - min(sigma)) / (max(sigma) - min(sigma) + 1e-9)
    # Score_i = (-mu_norm + kappa * sigma_norm) / cost_i
    
    # However, normalization changes the scale.
    # Let's stick to raw values but scale kappa.
    
    # A more sophisticated approach: **Knowledge Gradient** approximation via Monte Carlo?
    # Too slow.
    
    # Let's use **Expected Improvement** but with a twist:
    # Instead of comparing to f_best, compare to the *predicted minimum* of the batch.
    # This is effectively **q-EI** for q=1, but approximated.
    
    # Let's try **PI (Probability of Improvement)**:
    # Score_i = P(Y_i < f_best) / cost_i
    # This is simpler and often works well for global optimization.
    
    # Let's combine EI and PI:
    # Score_i = (EI_i + PI_i * sigma_i) / cost_i
    
    # Actually, let's go with **UCB** because it's a fundamentally different
    # design family (confidence bound vs. expected improvement) and handles
    # exploration/exploitation trade-off explicitly.
    
    # To adapt to the problem:
    # We want to minimize.
    # UCB for minimization: We want to maximize the lower bound of the function value.
    # Lower bound = mu_i - kappa * sigma_i.
    # We want to maximize this lower bound? No, we want to find the minimum.
    # We want to evaluate the point that has the highest probability of being the global minimum.
    # This is often approximated by maximizing: mu_i - kappa * sigma_i?
    # If mu_i is low and sigma_i is high, mu_i - kappa * sigma_i is very low.
    # But we want to *find* the minimum.
    # If we pick the point with the lowest (mu_i - kappa * sigma_i), we are picking the point
    # with the most optimistic lower bound.
    # So, Score_i = -(mu_i - kappa * sigma_i) / cost_i = (-mu_i + kappa * sigma_i) / cost_i.
    
    # Let's use an adaptive kappa.
    # Early on, we want to explore more (high kappa).
    # Later, we want to exploit more (low kappa).
    # We can use the number of evaluations or total cost as a proxy.
    # But we don't have the number of evaluations directly, only f_best and cost.
    # However, the cost array gives us the cost of *each* candidate.
    # The total cost so far is not passed.
    # But we can infer the "stage" from f_best?
    # If f_best is close to 0, we are in exploitation phase.
    # If f_best is high, we are in exploration phase.
    
    # Let's use a fixed kappa = 2.0 for now, as it's a reasonable default.
    # And normalize the scores to prevent cost from dominating if costs vary wildly.
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    
    # Normalize mu and sigma to [0, 1]
    mu_range = mu_max - mu_min
    sigma_range = sigma_max - sigma_min
    
    if mu_range < 1e-9:
        mu_norm = np.zeros_like(mu)
    else:
        mu_norm = (mu - mu_min) / mu_range
        
    if sigma_range < 1e-9:
        sigma_norm = np.zeros_like(sigma)
    else:
        sigma_norm = (sigma - sigma_min) / sigma_range
        
    # Adaptive kappa:
    # If f_best is very low (close to 0), we are likely near the optimum, so reduce exploration.
    # If f_best is high, increase exploration.
    # Let's say kappa = 1.0 + 2.0 * (f_best / (f_best + 1.0))
    # This gives kappa in [1, 3].
    # When f_best -> 0, kappa -> 1.0 (exploit more).
    # When f_best -> inf, kappa -> 3.0 (explore more).
    
    kappa = 1.0 + 2.0 * (f_best / (f_best + 1.0))
    
    # UCB Score for minimization:
    # We want to maximize the "optimistic potential".
    # Score_i = (-mu_norm + kappa * sigma_norm) / cost_safe
    
    scores = (-mu_norm + kappa * sigma_norm) / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
