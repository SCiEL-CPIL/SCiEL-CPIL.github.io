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
    
    # Safety check: if we are running out of time, return a deterministic fallback
    if _time_exceeded():
        # Fallback: return a simple EI score
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    n = len(mu)
    
    # Strategy: Batch-Consistent Expected Improvement (Batch EI) with Cost Awareness
    # 
    # The standard EI assumes independence of candidates. In BO, we often evaluate
    # one point at a time, but the GP posterior implies correlations.
    # A more structural approach is to approximate the "Batch EI" for a small batch size B=2 or 3.
    # However, computing exact Batch EI for 2000 points is O(N^3) or worse.
    # 
    # Instead, we use a "Correlation-Aware" heuristic that penalizes candidates
    # that are highly correlated with the *current best candidate* in the set of top-K
    # points, effectively encouraging diversity (batch consistency) without full batch optimization.
    #
    # Alternative Structural Change: **Thompson Sampling (TS) with Cost Normalization**
    # TS is a different design family from EI. It samples from the posterior.
    # For minimization, we sample f ~ N(mu, sigma^2) and pick the lowest.
    # To make it cost-aware, we can use a "Cost-Adjusted Thompson Sampling" where we
    # divide the sampled value by the cost, or more subtly, we can bias the sampling
    # towards lower cost regions by adjusting the mean or variance?
    # No, that breaks the GP posterior.
    #
    # Let's try **Probability of Improvement (PI) with a Cost-Dependent Threshold**.
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    # This is less sensitive to the magnitude of improvement than EI.
    # If we combine PI with a cost penalty, we get:
    # Score = PI / cost^alpha
    #
    # But a more structural change is to use **Knowledge Gradient (KG)** approximation.
    # KG measures the expected reduction in the final objective value (after remaining budget).
    # An approximation for KG in budget-constrained BO is related to the "Value of Information".
    # 
    # Let's implement a **Hybrid of EI and Kriging Distance (KD)** but with a
    # **Dynamic Weighting** based on the variance of the cost.
    #
    # Actually, the most promising structural change from the inspirations is
    # **Batch EI approximation via Cholesky or Monte Carlo**.
    # Since we cannot pass coordinates, we cannot compute distances directly.
    # We only have mu, sigma, cost.
    #
    # Wait, the prompt says "No coordinates are passed."
    # This is a crucial constraint. We cannot compute Kriging Distance (KD) or
    # pairwise correlations because we don't know the x-locations.
    # We only know the GP predictions (mu, sigma) and the costs.
    #
    # Therefore, any method requiring spatial information (like KD, or explicit
    # correlation matrices based on x) is INVALID.
    #
    # We must rely solely on the distributional properties provided: mu, sigma, cost.
    #
    # So, what structural changes are possible without x?
    # 1. Different Acquisition Function Family: PI, UCB, EI, KG, TS.
    # 2. Different Cost Integration: 1/cost, 1/cost^2, cost in denominator, etc.
    # 3. Different Handling of Uncertainty: e.g., using sigma to adjust the threshold.
    #
    # The current program uses EI + small bonus / cost.
    # Inspiration 1 (score 1.22) used a hybrid EI + KD. But KD requires x.
    # Did Inspiration 1 have x? The prompt says "No coordinates are passed" for MY task.
    # Let's re-read carefully: "No coordinates are passed."
    # If Inspiration 1 scored 1.22, maybe it didn't use KD? Or maybe the "KD" was
    # computed differently? Or maybe the prompt for Inspiration 1 was different?
    # The prompt says "INSPIRATIONS FROM OTHER SOLUTIONS".
    # If the task contract is the same, Inspiration 1 also had no coordinates.
    # So "Kriging Distance" in Inspiration 1 might be a misnomer for something else,
    # or they approximated it?
    #
    # Let's assume we CANNOT use spatial information.
    #
    # New Structural Idea: **Expected Hypervolume Improvement (EHVI) approximation**
    # In the (value, cost) space, we want to maximize the area under the Pareto front.
    # For a single point, the "contribution" to the hypervolume is roughly
    # (f_best - f(x)) * (cost_best - cost(x)) if it dominates.
    # But we don't know cost_best (total cost so far). We only have candidate costs.
    #
    # Let's try **Cost-Aware Probability of Improvement (CA-PI)**.
    # Standard PI: P(f(x) < f_best).
    # CA-PI: P(f(x) < f_best) / cost(x)^gamma.
    # This is structurally different from EI because EI weights by the *magnitude* of improvement.
    # PI treats all improvements equally (binary: improved or not).
    # This might be better if the landscape is rugged and small improvements are hard to distinguish.
    #
    # Let's try **Upper Confidence Bound (UCB) for Minimization**.
    # UCB_min = mu - beta * sigma.
    # We want to minimize this value.
    # So Score = -UCB_min / cost = (beta * sigma - mu) / cost.
    # This is structurally different from EI. EI is non-linear and bounded.
    # UCB is linear in mu and sigma.
    #
    # Let's try **Thompson Sampling (TS) Score**.
    # Sample y ~ N(mu, sigma^2).
    # Score = -y / cost.
    # Since we need a deterministic function (or we can seed the RNG), we can use a
    # "Deterministic TS" approximation or just use the expectation?
    # E[-y / cost] = -mu / cost. That's just Exploitation.
    # TS usually involves sampling. But acquisition functions are typically deterministic.
    # However, we can use a "Pessimistic" or "Optimistic" TS variant.
    #
    # Let's go with **UCB with Adaptive Beta**.
    # The current program uses EI. UCB is a different family.
    # UCB encourages exploration by subtracting a term proportional to sigma.
    # EI also encourages exploration but in a more complex way.
    #
    # Let's implement **Cost-Aware UCB**.
    # Score = (mu - beta * sigma) is the "optimistic" value (lower is better).
    # We want to maximize the score, so Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
    #
    # How to choose beta?
    # A common choice is beta = sqrt(log(1 + t)) or similar.
    # We don't have t (time step).
    # We can use a fixed beta or one based on the range of mu/sigma.
    #
    # Let's try **beta = 1.0** as a starting point for UCB.
    # Or **beta = 2.0** for more exploration.
    #
    # Another structural change: **Expected Improvement with a Cost-Dependent Threshold**.
    # Instead of f_best, use a threshold that depends on the cost?
    # No, f_best is the observed best.
    #
    # Let's try **Probability of Improvement (PI)** with a **Quadratic Cost Penalty**.
    # Score = PI / cost^2.
    # This is structurally different from EI/cost because PI is bounded [0,1] and
    # does not scale with the magnitude of improvement. It might be more robust
    # to outliers in the GP mean.
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Compute PI
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    pi = np.clip(pi, 0.0, 1.0)
    
    # Compute EI for comparison
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Let's try a **Mixture of PI and EI**.
    # Score = (w * PI + (1-w) * EI) / cost^2
    # If w=1, it's pure PI. If w=0, it's pure EI.
    # Let's try w=0.5.
    
    # But the directive says "Make a STRUCTURAL change".
    # PI is a different function family than EI.
    # Let's go with **Pure PI with Quadratic Cost Penalty**.
    
    score = pi / (cost_safe ** 2)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
