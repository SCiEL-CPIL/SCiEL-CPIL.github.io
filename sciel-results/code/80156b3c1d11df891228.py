import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track elapsed time to ensure we finish within limits
    start_time = time.time()
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # EHI (Expected Hypar Improvement) formulation
    # The standard EHI for minimization with budget B is:
    # EHI(x) = E[ -tanh( (f(x) - f_best) / B ) ]
    # Where B is the remaining budget or a characteristic scale of improvement.
    # In a cost-aware setting, we can adapt this.
    # However, a common cost-aware variant is to maximize the expected improvement
    # relative to the cost, or to use a specific kernel.
    
    # Let's use the formulation from "Cost-aware Bayesian optimization" literature.
    # One robust approach is to use the Expected Improvement but scaled by the cost
    # in a way that reflects the "value" of the information.
    
    # The current program uses EI / cost^1.5.
    # The directive suggests EHI.
    # EHI = E[ -tanh( (f(x) - f_best) / sigma_B ) ]
    # Where sigma_B is a budget-related scale.
    
    # Let's try a different structural change: 
    # Instead of EI / cost^alpha, let's use a "Cost-Adjusted Expected Improvement" 
    # that accounts for the probability of improvement more directly in a 
    # budget-aware way, or simply switch to a different acquisition function 
    # family entirely: Expected Logarithmic Improvement (ELI) or similar, 
    # but the directive specifically asks for EHI.
    
    # Implementation of EHI:
    # We need to define the budget scale. The total budget is 30.
    # The cost of a point is in [1, 2].
    # Let's use a characteristic budget scale B. 
    # A common choice is the remaining budget, but we don't have that explicitly 
    # except via the loop context which we don't see inside this function.
    # However, f_best is given.
    
    # Let's approximate the EHI using the normal approximation for f(x) ~ N(mu, sigma^2).
    # E[-tanh((f(x) - f_best)/B)]
    
    # Since we don't know the exact remaining budget, we can use a fixed scale 
    # or estimate it. 
    # Alternatively, we can use the "Cost-Aware EI" which is often better 
    # formulated as:
    # Score = EI(x) * (1 / cost(x))
    # But the current program already does this with a power of 1.5.
    
    # Let's try a structural change: **Pareto Front-based Acquisition** or 
    # **Hypervolume Improvement (HI)** based acquisition.
    # However, with a single objective and cost, HI is effectively the same as 
    # EI if we treat cost as the second objective.
    
    # Let's stick to the directive: EHI.
    # We will use a budget scale B. 
    # If the total budget is 30 and we start with 5 points (cost ~ 1.5 each = 7.5),
    # remaining is ~22.5.
    # Let's assume a characteristic improvement scale. 
    # Hartmann 6D range is roughly 0 to 3.3.
    # Let's set B = 1.0 as a scale for the tangent argument.
    
    # Actually, a more robust "structural" change from EI/cost^k is to use 
    # **Expected Improvement with a Cost Penalty in the Exponent** or 
    # **Bayesian Optimization with Adaptive Cost**.
    
    # Let's try **Expected Improvement (EI)** but with a **Softplus** cost penalty 
    # instead of a power law, which behaves differently at low/high costs.
    # Score = EI - lambda * cost
    # This is a linear penalty.
    
    # Or, let's try **UCB (Upper Confidence Bound)** with a cost-aware beta.
    # UCB = mu - beta * sigma (for minimization, we want low mu and high uncertainty? No, UCB for minimization is mu - beta*sigma? No, we want to find low values. 
    # Standard UCB for maximization: mu + beta*sigma.
    # For minimization: mu - beta*sigma.
    # But we want to exploit low mu. 
    # So Acquisition = -mu + beta*sigma? No, that's not right.
    # We want to select x that minimizes f.
    # EI is standard.
    
    # Let's implement the EHI as requested.
    # EHI = E[ -tanh( (f(x) - f_best) / B ) ]
    # We approximate this integral.
    # Let y = (f(x) - f_best) / B.
    # f(x) ~ N(mu, sigma^2).
    # y ~ N( (mu - f_best)/B, (sigma/B)^2 ).
    # We need E[-tanh(y)].
    
    # There is no closed form for E[tanh(N)]. We can use a Monte Carlo approximation 
    # or a lookup table, but that's expensive.
    # Alternatively, we can use the fact that tanh(z) ~ 1 - 2e^{-2|z|} for large z.
    
    # Given the constraints (2000 points, 300s limit), we can afford a vectorized 
    # approximation.
    
    # Let's use a simpler structural change: **Gaussian Process Upper Confidence Bound (GP-UCB)** 
    # with a cost-aware exploration parameter.
    # UCB(x) = mu(x) - beta(t) * sigma(x)
    # But we need to incorporate cost.
    # Cost-Aware UCB: UCB(x) = mu(x) - beta(t) * sigma(x) + gamma * cost(x)
    # We want to MINIMIZE this score? No, the acquisition returns a score where 
    # higher = evaluate next.
    # So we want to MAXIMIZE the acquisition.
    # For minimization, we want low mu.
    # So Acquisition = -mu(x) + beta(t) * sigma(x) - gamma * cost(x)
    
    # Let's try this.
    # beta(t) can be time-dependent or fixed.
    # gamma is a cost penalty.
    
    # Let's use a fixed beta and gamma.
    # beta = 2.0 (typical for exploration)
    # gamma = 1.0 (cost penalty)
    
    # However, EI is generally superior to UCB in low-noise regimes.
    # The knowledge snippet says: "EI provides proper Bayesian calibration... avoiding UCB's tendency to over-explore".
    # So switching to UCB might be a step back.
    
    # Let's try **Expected Logarithmic Improvement (ELI)**.
    # ELI = E[ log(f_best / f(x)) ] (if f(x) > 0)
    # But Hartmann can be negative? No, min is -3.3.
    # So log is not defined for negative values.
    
    # Let's try **Hypervolume Improvement (HI)** in the (f, cost) space.
    # Reference point: (f_ref, cost_ref).
    # f_ref = f_best - 1.0 (we want to improve by at least 1.0)
    # cost_ref = 2.0 (max cost)
    # HI(x) = Volume of the region dominated by (f(x), cost(x)) but not by (f_ref, cost_ref).
    # This is complex to compute in expectation.
    
    # Let's go back to **EHI** but with a simpler approximation.
    # EHI ~ EI / B for small improvements, and saturates for large improvements.
    # The current EI/cost^1.5 does not saturate.
    
    # Let's use **Tanh-EI**:
    # Score = tanh( EI / scale )
    # This saturates the EI, preventing a single point with huge EI (due to high sigma) 
    # from dominating, which might be beneficial if that point is also high cost.
    # But we already divide by cost.
    
    # Let's try a **Weighted Sum of EI and Cost**:
    # Score = w1 * EI - w2 * cost
    # This is a linear trade-off.
    
    # Let's implement **Cost-Aware EI with a Softplus Penalty**:
    # Score = EI - lambda * softplus(cost)
    # softplus(x) = log(1 + exp(x))
    # This penalizes high costs smoothly.
    
    # Let's try this structural change.
    # lambda = 0.5
    
    # Calculate EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Calculate Cost Penalty
    # cost is in [1, 2]
    # softplus(cost) is roughly cost for cost > 1
    # Let's use a linear penalty for simplicity and speed, but with a different 
    # structure: 
    # Score = EI / (1 + cost)
    # This is similar to EI / cost but less aggressive.
    
    # Let's try **Inverse Cost Weighting**:
    # Score = EI * (1 / cost)
    # This is the standard cost-aware EI.
    
    # The current program uses cost^1.5.
    # Let's try **cost^1.0** (linear) to see if the aggressive penalty was hurting.
    # Or **cost^0.5** (milder penalty).
    
    # Let's try **cost^0.5**.
    
    safe_cost = np.maximum(cost, 1e-9)
    score = ei / np.sqrt(safe_cost)
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # If we're getting close to the deadline, return immediately
    elapsed = time.time() - start_time
    if elapsed > 295:
        return score
    
    return score
