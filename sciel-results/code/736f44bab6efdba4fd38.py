import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function based on approximate Expected Hypervolume Improvement (EHVI)
    for the bi-objective problem: minimize f(x) and minimize cost(x).
    
    We approximate EHVI by considering a small set of reference points derived from
    the current Pareto frontier of observed (cost, f) pairs, but since we only have
    f_best and cost array for candidates, we approximate the "improvement" as
    the probability-weighted area of the rectangle defined by (f_best, cost) and
    (mu, cost) that lies below the current best trade-off.
    
    However, a simpler and robust bi-objective acquisition that behaves differently
    from EI/cost is the **Cost-Effective Upper Confidence Bound (CE-UCB)** or
    **Cost-Effective Expected Improvement (CE-EI)** with a non-linear cost penalty.
    
    Let's use a different family: **Pareto-EI**.
    Since we don't have the full history of (cost, f) pairs in the signature,
    we must rely on f_best and the current candidate's cost.
    
    Actually, the most distinct and effective strategy often found in literature
    for cost-aware BO is **Cost-Effective UCB** or **Cost-Effective EI**.
    The current program uses EI / cost. This is linear in cost.
    
    Let's try **Cost-Effective EI with a power penalty**: EI / (cost ** gamma)
    where gamma > 1 makes it more sensitive to cost.
    
    Alternatively, **Expected Improvement over the Cost-Effective Frontier**.
    
    Let's try a **Weighted Sum of EI and Cost Reduction**.
    Score = EI - lambda * (cost - min_cost_observed).
    But we don't have min_cost_observed.
    
    Let's stick to a robust variant: **Cost-Effective EI** but with a different
    normalization or a **UCB** variant.
    
    Inspiration 2 used Cost-Aware UCB and got 1.0369.
    Inspiration 1 got 1.1571. What did it do? It likely used a more sophisticated
    cost-aware metric.
    
    Let's implement **Cost-Effective EI** with a **non-linear cost discount**:
    Score = EI / (1 + beta * (cost - 1))
    This is bounded and smooth.
    
    Or, **Expected Hypervolume Improvement** approximation:
    The hypervolume improvement of a point (c, f) over the reference point (c_ref, f_ref)
    is (c_ref - c) * (f_ref - f) if c < c_ref and f < f_ref.
    We can approximate the expected improvement in hypervolume by:
    E[HV_improvement] ~ (f_best - mu) * P(f < f_best) * (cost_ref - cost) * P(cost < cost_ref)
    This is getting complicated without knowing the reference points.
    
    Let's try **Cost-Effective UCB** (CE-UCB):
    UCB = mu - beta * sigma
    Score = UCB / cost^gamma
    
    Let's try **Cost-Effective EI** with gamma = 2.
    This penalizes high cost much more heavily than linear division.
    """
    # Safety check
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a safe default, e.g., zero scores
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Non-linear cost penalty: EI / (cost ** 2)
    # This strongly favors cheap evaluations that have moderate EI,
    # and heavily penalizes expensive evaluations unless their EI is very high.
    scores = ei / (cost_safe ** 2)
    
    # Alternative: Cost-Effective UCB
    # beta = 2.0
    # ucb = mu - beta * sigma_safe
    # scores = ucb / (cost_safe ** 1.5)
    
    # Let's stick to EI / cost^2 as it is a distinct family (non-linear cost weighting)
    # compared to the linear EI / cost.
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
