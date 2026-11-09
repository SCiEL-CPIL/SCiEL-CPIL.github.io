"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return early if close
    t0 = time.time()
    deadline = t0 + 325  # 5s safety margin
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware Knowledge Gradient (KG) approximation
    # The standard KG for minimization is complex (involving integrals of minima).
    # A common and effective approximation is the "Expected Improvement" (EI) 
    # which is the KG for the case where we only care about the immediate improvement 
    # over f_best, ignoring the possibility that a future point might be better 
    # than the current best but not the global best. However, EI is often too greedy.
    
    # A better structural change: Use the **Probability of Improvement (PI)** 
    # combined with a **Thompson Sampling**-like stochastic weight, or simply 
    # use a **Cost-Aware EI** but with a different normalization that better 
    # reflects the value of information.
    
    # Let's try a different family: **Expected Minimum (EM)** approximation.
    # The expected minimum after evaluating x is E[min(f_best, f(x))].
    # min(f_best, f(x)) = f_best - max(0, f_best - f(x)).
    # So E[min] = f_best - E[max(0, f_best - f(x))] = f_best - EI.
    # Minimizing E[min] is equivalent to maximizing EI.
    # So standard EI *is* the one-step KG for the minimum.
    
    # However, the current program uses a hybrid EI/UCB.
    # The UCB term encourages exploration even when EI is low (far from f_best).
    # The inspiration suggests that a pure cost-normalized EI (score 1.0) is good.
    # The current hybrid (score 0.72) is worse.
    
    # Structural Change: Instead of a fixed linear blend, use a **Bayesian Optimization 
    # with a Cost-Aware GP** proxy? No, we only get mu/sigma.
    
    # Let's look at the "Probability of Being the Best" (POB) or **Probability of Improvement**.
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    # This is a binary signal. It ignores the magnitude of improvement.
    
    # A more sophisticated approach: **Expected Improvement with Uncertainty Penalty**.
    # Or, use **Knowledge Gradient** properly?
    # Approximate KG by sampling from the GP posterior? Too slow for 2000 points in 300s if done naively, 
    # but we can use a fast approximation.
    
    # Actually, the most robust "different" strategy that often outperforms EI in high dimensions 
    # or when cost is involved is **Expected Improvement divided by the square root of cost** 
    # or **EI / (1 + log(cost))**?
    
    # Let's try **Cost-Effective EI**:
    # Score = EI / (cost ** gamma)
    # If gamma=1, it's EI/cost.
    # If gamma=0.5, it's EI/sqrt(cost).
    # The cost is in [1, 2]. sqrt(cost) is in [1, 1.414]. cost is in [1, 2].
    # Using sqrt(cost) penalizes expensive points less severely, potentially allowing 
    # the algorithm to take a "risk" on a high-EI expensive point if the EI is large enough.
    
    # But the inspiration says EI/cost (gamma=1) scored 1.0.
    # The current hybrid scored 0.72.
    # This implies the hybrid UCB term is harmful.
    
    # Let's try a **Pure EI** but with a **dynamic exploration factor** based on the 
    # **number of evaluations** or **cost spent**?
    # We don't have the number of evaluations, only `cost` (which is per-point cost, not total).
    # Wait, the signature is `acquisition(mu, sigma, f_best, cost)`.
    # `cost` has shape (2000,). This is the cost of evaluating *each* candidate point x.
    # The loop runs while *total* cost < 30.
    # We don't know the total cost spent so far inside this function, only f_best.
    
    # However, we can infer the stage of the search from f_best?
    # Early on, f_best is high (bad). Later, f_best is low (good).
    # If f_best is close to the known minimum (-3.32237), we should exploit.
    # If f_best is far, we should explore.
    
    # Let's define a "confidence" parameter based on f_best.
    # Hartmann 6D min is -3.32237.
    # Initial random points might be around -1 or -2?
    # If f_best > -2.5, we are early. If f_best < -3.0, we are late.
    
    # Strategy: **Adaptive EI/UCB Blend**
    # alpha = sigmoid(k * (f_best - threshold))
    # If f_best is high (bad), alpha is low -> more UCB (exploration).
    # If f_best is low (good), alpha is high -> more EI (exploitation).
    
    # But the pure EI/cost scored 1.0. The hybrid scored 0.72.
    # This suggests that *adding* UCB was bad, or the specific blend was bad.
    # Maybe UCB is good *early* and bad *late*?
    
    # Let's try: **Pure EI / cost** but with a **floor** on the score to prevent 
    # numerical issues, and maybe a slight **randomness** (Thompson Sampling style) 
    # to break ties? No, we need deterministic scores for the evaluator.
    
    # Let's try a different acquisition family: **Expected Regret** vs **Expected Improvement**.
    # They are similar.
    
    # How about **Probability of Improvement (PI)**?
    # PI = Phi(z).
    # Score = PI / cost.
    # PI is bounded [0, 1]. EI is unbounded.
    # PI might be more robust to outliers in mu/sigma.
    
    # Let's try **PI / cost**.
    
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Score = PI / cost
    score = pi / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
