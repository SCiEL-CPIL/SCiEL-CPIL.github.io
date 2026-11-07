# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a dynamic UCB-EI hybrid.
    
    Strategy:
    1. Compute UCB and EI scores.
    2. Dynamically adjust the exploration weight (beta) based on the 'effective'
       exploration progress, approximated by the mean cost of the current candidates 
       and the number of evaluations implied by the total cost.
    3. Combine UCB and EI with a fixed weight.
    4. Scale the combined score by 1/cost to favor cheaper evaluations, but 
       maintain the relative ranking of high-value candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety guards
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Compute Standardized Improvement (z-score)
    diff = f_best - mu
    z = diff / sigma
    z = np.clip(z, -10, 10)
    
    # 2. Compute Expected Improvement (EI)
    ei = diff * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 3. Compute Upper Confidence Bound (UCB)
    # Dynamic beta: 
    # We estimate the number of evaluations performed so far.
    # The loop runs while total_cost < 30.
    # Initial cost is 5 points * avg_cost.
    # Let's approximate 'step' or 'progress' by the mean cost of the current batch.
    # A higher mean cost implies we are evaluating "expensive" points, so we might 
    # want to be more conservative? Or rather, the total budget is fixed.
    # 
    # A simpler structural change: 
    # The standard UCB uses a constant beta. 
    # Let's use a beta that decays as the uncertainty (sigma) decreases globally?
    # No, we don't have global sigma stats easily without state.
    #
    # Let's look at the inspiration: "Combining EI with LCB balances exploitation and exploration".
    # LCB for minimization is mu - beta * sigma.
    # UCB for minimization (maximizing the score) is mu - beta * sigma? 
    # Wait, for minimization:
    # We want low f(x).
    # LCB (Lower Confidence Bound) gives a lower estimate of the function value.
    # If we maximize the "regret" or "improvement", we look at (f_best - mu) + beta * sigma.
    # This is effectively maximizing the probability of improvement.
    # Let's call this "Optimistic Improvement" or UCB-based improvement.
    # Score_UCB = (f_best - mu) + beta * sigma
    
    # Let's try a dynamic beta.
    # If we are early in the process, we want high exploration.
    # How to know if we are early? 
    # The cost array is for the 2000 candidates. 
    # The total cost budget is 30.
    # If the mean cost of the candidates is high, we can afford fewer evaluations.
    # If the mean cost is low, we can afford more.
    # 
    # Let's use the mean cost of the current 2000 candidates as a proxy for the "price" of exploration.
    mean_cost = np.mean(cost)
    
    # Normalize mean_cost to a range [0, 1] roughly.
    # Cost is 1 + mean(x_scaled). So cost is in [1, 2].
    # Let's map [1, 2] to [0, 1] for a decay factor.
    # Actually, let's just use a fixed, well-tuned beta first, but combined differently.
    # The current program uses EI / cost.
    # The inspiration says EI + LCB is better.
    # Let's try: Score = (w_ei * EI + w_ucb * UCB_improvement) / cost
    
    # UCB Improvement for minimization:
    # We want to maximize the chance that f(x) < f_best.
    # A common surrogate is: (f_best - mu) + beta * sigma
    # This is linear in sigma, unlike EI which is sublinear.
    # This encourages exploring high-uncertainty regions more strongly.
    
    # Let's set beta based on the scale of the problem.
    # Ackley 2D on [-5,5] has values up to ~30.
    # f_best starts around 10-20 and goes to 0.
    # sigma is in objective units.
    # A beta of 1.0-2.0 is typical.
    
    beta = 1.5
    
    ucb_improvement = (f_best - mu) + beta * sigma
    
    # Combine EI and UCB
    # EI is bounded and smooth. UCB is linear.
    # Weighting: 0.5 EI + 0.5 UCB?
    # Or maybe 0.7 EI + 0.3 UCB to keep stability?
    
    w_ei = 0.5
    w_ucb = 0.5
    
    combined_score = w_ei * ei + w_ucb * ucb_improvement
    
    # Cost-aware scaling
    # Instead of division, which can amplify noise in low-cost areas,
    # let's try a power law or simple division but with a floor on the score?
    # The current program uses EI / cost.
    # Let's stick to division but on the combined score.
    
    score = combined_score / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
