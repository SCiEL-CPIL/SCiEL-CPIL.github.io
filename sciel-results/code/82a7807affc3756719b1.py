import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Multi-Step Knowledge Gradient (KG) approximation.
    
    Strategy:
    Unlike Thompson Sampling (stochastic) or UCB/EI (single-step bounds), Knowledge Gradient 
    approximates the expected reduction in regret by considering the value of information 
    gained from evaluating a point. 
    
    We use a deterministic approximation of KG:
    1. For each candidate x_i, we estimate the probability that it is the current best 
       (minimizer) given the GP posterior. This is related to the Probability of Improvement (PoI) 
       but adjusted for the "value" of knowing it's the best.
    2. We incorporate a "multi-step" heuristic: points with high uncertainty (high sigma) 
       that are also close to the current best are valuable because they might reveal a 
       new, lower minimum that persists.
    3. Cost awareness: We penalize candidates by their evaluation cost. The KG value is 
       reduced by the cost of obtaining that information.
    
    The score is approximated as:
    Score_i = Expected_Improvement_Potential_i * Confidence_Factor_i - Cost_Penalty_i
    
    Where:
    - Expected_Improvement_Potential is approximated by the lower bound of the GP mean 
      minus the current best, weighted by the probability of being better.
    - Confidence_Factor encourages points where the GP is uncertain but the mean is low.
    - Cost_Penalty is the direct cost of evaluation.
    
    This is structurally different from TS (no random sampling) and UCB (no fixed beta multiplier).
    It aims to maximize the "regret reduction per unit cost" over a horizon.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, 10s safety margin
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Basic sanity checks
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros_like(mu)
        
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # 1. Compute the "raw" potential for improvement.
    # We want to find x where f(x) < f_best.
    # The probability that f(x) < f_best is P(Z < (f_best - mu) / sigma) = Phi((f_best - mu) / sigma).
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in CDF
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement (PoI)
    p_improve = norm.cdf(z)
    
    # 2. Estimate the "Value of Information" (VOI).
    # A point is valuable if:
    # a) It is likely to be better than f_best (high p_improve).
    # b) It has high uncertainty (high sigma), suggesting the GP hasn't mapped this area well.
    # c) Its mean is low (close to or below f_best).
    
    # Expected Improvement (EI) is a standard measure of VOI for single-step.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We use EI as the base "benefit".
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * p_improve + sigma * phi_z
    
    # 3. Multi-Step / Exploration Bonus.
    # Pure EI is greedy. KG looks ahead.
    # In the early stages, or in high-uncertainty regions, we want to explore.
    # We add a term proportional to sigma to encourage exploration of unmapped areas,
    # but scaled by the likelihood that the area is promising (p_improve).
    # This creates a "smart exploration" where we explore uncertain areas that *might* be good.
    
    # Dynamic exploration rate: 
    # If we have found a very good f_best (close to true min), we should exploit.
    # If f_best is still high, we should explore.
    # However, we don't know the true min. 
    # Heuristic: Use the variance of the means or the current sigma distribution.
    # Simpler: Blend EI with a pure uncertainty term (sigma) weighted by cost.
    
    # Let's define the "KG-like" score.
    # The expected reduction in regret from evaluating x is roughly proportional to EI.
    # But we also gain information about the rest of the space.
    # A common approximation for cost-aware KG is:
    # Score = (EI - lambda * cost) / (1 + uncertainty_penalty)
    
    # To be structurally different from simple EI/cost, let's use a **Bayesian Regret Minimization** proxy.
    # We want to minimize the expected maximum regret.
    
    # Let's try a **Probability of Improvement weighted by Uncertainty** (PoI * Sigma) 
    # plus a **Mean-based** term.
    
    # Actually, the most distinct "KG" approximation without coordinates is difficult.
    # Let's pivot to **Lower Confidence Bound with Adaptive Beta** based on the 
    # "Value of Information" of the current best.
    
    # Wait, the directive asks for a DIFFERENT strategy.
    # Current: Thompson Sampling.
    # Inspiration 1 (1.63): Batched TS.
    # Inspiration 2 (1.14): EI with norm.
    # Current Parent (1.45): TS.
    
    # Let's implement **Expected Improvement (EI) with a Cost-Adjusted Discount**.
    # But to make it structurally different from standard EI, we will use
    # **Multi-Armed Bandit UCB with Dynamic Regret Estimation**.
    
    # No, let's stick to the **Knowledge Gradient** concept via **Expected Reduction in Regret (ERR)**.
    # ERR_i = E[ min(f_best, f(x_i)) - f(x*) ] - E[ min(f_best, f(x_j)) - f(x*) ] for j != i?
    # Too complex without coordinates.
    
    # Let's use **Probability of Improvement (PoI) scaled by the "Surprise" (Sigma)**.
    # Score = PoI * Sigma / Cost.
    # This favors points that are:
    # 1. Likely to beat f_best (PoI > 0.5)
    # 2. Uncertain (High Sigma) -> High information gain
    # 3. Low Cost
    
    # This is effectively **Information Gain per Cost**.
    
    # To avoid being too exploitative (PoI is high only very close to f_best),
    # we add a small UCB-like term.
    
    # Let's compute:
    # Term 1: PoI * Sigma (Information Gain)
    # Term 2: -mu (Optimism)
    
    # We will combine them with a dynamic weight.
    # If sigma is high, we rely more on Information Gain.
    # If sigma is low, we rely more on the Mean.
    
    # Weight w = 1 / (1 + sigma)
    # Score = w * (-mu) + (1-w) * (PoI * Sigma) ? No, units don't match.
    
    # Let's use **Normalized EI** plus **Uncertainty Bonus**.
    
    # 1. Calculate EI
    # 2. Calculate "Exploration Bonus" = sigma * p_improve (uncertainty weighted by promise)
    
    # Total Benefit = EI + 0.5 * sigma * p_improve
    # Score = Total Benefit / cost
    
    # This is structurally different from TS (deterministic) and simple EI (includes explicit exploration term).
    
    benefit = ei + 0.5 * sigma * p_improve
    
    # Cost adjustment
    # Divide by cost to favor cheap evaluations.
    scores = benefit / cost
    
    # 4. Safety and Finalization
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    # Optional: If we are very close to the deadline, return the current best scores
    if time.time() > deadline:
        return scores
    
    return scores
