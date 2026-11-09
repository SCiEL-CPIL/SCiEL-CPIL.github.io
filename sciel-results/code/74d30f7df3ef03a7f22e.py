"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety guard: if we are running out of time, return a deterministic safe score
    # to avoid timeouts. The evaluator kills at 330s.
    # We assume this function is called many times. 
    # If we detect we've been running for too long, we should just return something fast.
    # However, the GP computation is done outside. This function is just scoring.
    # Scoring 2000 points is fast. The risk is if the GP is slow. 
    # But we cannot control the GP. We just score.
    # So the main risk is if the loop calling this is slow.
    # We will stick to a fast vectorized implementation.
    
    # Dynamic exploration coefficient beta(t)
    # cost is the cost of evaluating each candidate.
    # Total cost so far is not directly passed, but 'cost' here is the array of costs for the 2000 candidates.
    # Wait, the prompt says: "mu, sigma and cost have shape (2000,)."
    # "Runs start with 5 uniform random points (cost counted). While total cost is below 30..."
    # The `cost` parameter in `acquisition` is the cost of evaluating *each* of the 2000 new candidates.
    # We don't know the *total* cost accumulated so far directly from the arguments.
    # However, we can infer the "stage" of optimization.
    # Early on, f_best is high. Later, f_best is low.
    # A common heuristic is to use the number of observations. We don't have N.
    # But we have f_best.
    # Let's use a decay based on the current f_best relative to the known minimum?
    # Or simply a fixed beta that is well-tuned?
    # The inspiration said "dynamic, uncertainty-driven exploration coefficient".
    # Without N, we can use the spread of sigma or f_best.
    # Let's try a beta that adapts to the uncertainty level.
    # If sigma is high, we want to explore.
    # Standard UCB: mu + beta * sigma.
    # Cost-aware: (mu + beta * sigma) / cost.
    # Note: We are minimizing. So we want low mu.
    # Wait, UCB for minimization is usually Lower Confidence Bound (LCB): mu - beta * sigma.
    # Because we want to find the minimum.
    # If we use UCB (mu + beta*sigma), we are maximizing the upper bound, which is for maximization.
    # For minimization, we want to be optimistic that the true value is low.
    # So we should use LCB: mu - beta * sigma.
    
    # Let's re-read the inspiration: "Cost-Aware Upper Confidence Bound (UCB) that uses a dynamic...".
    # In many BO contexts, people use UCB for maximization. For minimization, it's LCB.
    # Or they define the acquisition as - (mu - beta*sigma) / cost.
    # Let's stick to the minimization logic:
    # Score = (mu - beta * sigma) / cost.
    # Higher score is better.
    # If sigma is large, the score becomes more negative (worse) if beta is positive?
    # No, if we want to explore high uncertainty regions for minimization:
    # We want to pick points where the lower bound is low.
    # LCB = mu - beta * sigma.
    # If sigma is high, LCB is lower. So it gets a higher score (if we maximize LCB? No, LCB is a value).
    # We want to minimize the function. So we want the smallest possible LCB.
    # So the acquisition function should return a value such that the *minimum* LCB is chosen?
    # The prompt says: "Return 2000 finite scores (higher = evaluate next)".
    # So we need to maximize the score.
    # If we want to pick the point with the lowest LCB, we should return -LCB?
    # Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
    # Let's check EI logic again.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z).
    # If mu is low (good), EI is high.
    # If sigma is high, EI can be high due to the sigma*phi(z) term.
    # So EI naturally handles both.
    
    # Let's look at the "UCB" inspiration again.
    # If it scored 0.75, it might have been implemented as a maximization UCB incorrectly for a minimization problem, or the beta was wrong.
    # Let's try a different structural change: **Knowledge Gradient (KG)**?
    # KG is computationally expensive (requires integration). Not suitable for 2000 points per step without heavy optimization.
    
    # How about **Thompson Sampling**?
    # We can't sample the function directly because we only have mu and sigma at 2000 points.
    # But we can simulate: sample a value y_i ~ N(mu_i, sigma_i) for each candidate?
    # That's not standard TS. TS samples the *function* parameters.
    # However, a common approximation is to sample a single realization from the GP posterior at the candidate points?
    # No, that's expensive.
    
    # Let's reconsider the **LCB** approach but with a twist.
    # The problem with LCB is that it ignores f_best.
    # We can use a **Pessimistic** bound? No, we want Optimistic (Low).
    
    # Let's try a **Normalized LCB** that also considers f_best.
    # Score = (mu - beta * sigma) / cost.
    # If we maximize this, we pick the point with the most negative (best) lower bound.
    # This is a valid acquisition function for minimization.
    # The key is beta.
    # If beta is too small, it's greedy (just mu/cost).
    # If beta is too large, it picks high uncertainty points (noise).
    #
    # Let's try a **Hybrid**: 
    # Early on, we don't know f_best well. 
    # Let's use a beta that decays with the "confidence" in f_best.
    # Since we don't have N, let's use the variance of sigma or just a fixed reasonable beta.
    # The Hartmann 6D is multimodal. 
    # 
    # Alternative Structural Change: **Entropy-based exploration**?
    # No.
    #
    # Let's look at the "Blending EI with UCB" finding which improved to 1.052.
    # The directive says "Make a STRUCTURAL change... different design or modelling family".
    # Blending is a hybrid, not a new family.
    #
    # What if we use **Probability of Improvement (PI)**?
    # PI = Phi((f_best - mu) / sigma).
    # It is simpler than EI. It ignores the magnitude of improvement, only the probability.
    # This might be more robust to noise?
    # Cost-aware PI: PI / cost.
    #
    # Let's try **Cost-Effective PI**.
    #
    # Another idea: **Maximizing the Lower Confidence Bound (LCB)** is structurally different from EI.
    # EI is an expectation. LCB is a bound.
    # Let's implement LCB with a fixed beta that is tuned for Hartmann 6D.
    # Typical beta for UCB/LCB in BO is around 1.0 to 5.0.
    # Let's try beta = 2.0.
    # Score = (mu - 2.0 * sigma) / cost.
    # Wait, if we maximize (mu - beta*sigma)/cost, we are picking the point with the lowest predicted value adjusted for uncertainty.
    #
    # Let's add a safety check for time.
    #
    # One more structural idea: **Multi-objective style**?
    # No.
    #
    # Let's go with **LCB (Lower Confidence Bound)** normalized by cost.
    # This is a "confidence bound" family, distinct from "expected improvement" (moment-based) or "probability of improvement" (tail-probability based).
    
    # To make it "dynamic" without N:
    # We can use the current f_best.
    # If f_best is very high (far from -3.32), we should explore more.
    # Let's map f_best to beta.
    # f_best range: ~0 to -3.32.
    # If f_best > -1.0, beta = 3.0 (explore).
    # If f_best < -3.0, beta = 0.5 (exploit).
    # Linear interpolation.
    
    f_best_val = float(np.mean(f_best)) # f_best is a scalar? "f_best is the lowest observed value". It's a scalar.
    # The prompt says "f_best is the lowest observed value". It is likely a scalar.
    # But `cost` is an array. `mu` and `sigma` are arrays.
    
    # Map f_best to beta
    # f_best = 0.0 -> beta = 4.0
    # f_best = -3.32 -> beta = 0.5
    # Clamp f_best
    fb = np.clip(f_best_val, -3.32, 0.0)
    # Linear map
    # t = (fb - 0.0) / (-3.32 - 0.0) = fb / -3.32
    # t is in [0, 1] where 0 is best (-3.32) and 1 is worst (0.0)
    t = fb / -3.32
    # beta = 0.5 + (4.0 - 0.5) * t
    beta = 0.5 + 3.5 * t
    
    # Calculate LCB score
    # Score = (mu - beta * sigma) / cost
    # We want to MAXIMIZE this score.
    # Since we are minimizing the function, a lower (mu - beta*sigma) is better.
    # So we should return - (mu - beta * sigma) / cost ?
    # Let's check.
    # If point A has mu=0, sigma=0. Point B has mu=-1, sigma=0.
    # LCB_A = 0. LCB_B = -1.
    # We want to pick B.
    # If Score = LCB / cost, Score_A = 0, Score_B = -1.
    # Max(Score) picks A. That's wrong.
    # So Score = - LCB / cost = (beta * sigma - mu) / cost.
    # Score_A = 0. Score_B = 1.
    # Max picks B. Correct.
    
    # So: Score = (beta * sigma - mu) / cost
    
    # Ensure no division by zero or negative costs (cost is 1 + mean(x), so >= 1)
    scores = (beta * sigma - mu) / cost
    
    # Handle potential numerical issues
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
