# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a modified Expected Improvement (EI)
    with a soft cost penalty and robustness against zero/low sigma.
    
    The current seed uses EI/cost. We propose a different approach:
    1. Compute standard EI.
    2. Apply a non-linear cost penalty that is less aggressive than 1/cost for low costs,
       but still penalizes high costs. Specifically, we use EI / (1 + cost) or similar,
       but actually the problem states cost = 1 + mean(x_scaled).
       The range of cost is [1, 2].
       1/cost ranges from 1 to 0.5.
       
       Alternative: Use a "Cost-Weighted" approach where we maximize EI but subtract a term
       proportional to cost, or divide by a softer function.
       
       However, a very strong baseline is often just maximizing EI, or EI with a small cost penalty.
       Since the cost is bounded [1, 2], dividing by cost is reasonable.
       
       Let's try a different family: **Probability of Improvement (PI)** with cost weighting?
       Or **Upper Confidence Bound (UCB)**?
       
       UCB: mu + kappa * sigma.
       Cost-aware UCB: (mu + kappa * sigma) / cost? No, that doesn't make sense for minimization.
       For minimization, we want low mu and high sigma (exploration).
       UCB for minimization: mu - kappa * sigma? No, that's LCB.
       LCB: mu - kappa * sigma. We want to minimize, so we prefer low values.
       But in BO, we usually maximize an acquisition function.
       So Acquisition = -(mu - kappa * sigma) / cost? Or just kappa * sigma - mu?
       
       Let's stick to Improvement-based metrics as they are generally more robust.
       
       The seed uses EI / cost.
       Let's try **Expected Improvement with a cost-aware beta parameter** or a **different normalization**.
       
       Another idea: **Knowledge Gradient (KG)** is expensive.
       
       Let's try **Cost-Adjusted EI** where the cost penalty is applied differently.
       Instead of dividing by cost, we subtract a cost term:
       EI - lambda * (cost - 1)
       But EI is in objective units, cost is in [1,2]. The scales might not match.
       
       Better: **Maximize EI / (1 + cost^2)**? No.
       
       Let's look at the cost definition: cost = 1 + mean(x_scaled).
       x_scaled is in [0,1]. So cost is in [1,2].
       The variation in cost is small (factor of 2).
       The variation in EI can be large.
       
       Perhaps the seed is already quite good. To improve, we might need a better balance of exploration/exploitation.
       The standard EI uses f_best. 
       
       Let's try **Local Penalized EI** or **Scaled EI**?
       
       Actually, a common improvement is to use **Expected Improvement with a small epsilon** in the denominator to avoid division by zero, which is done.
       
       What if we use **Probability of Improvement (POI)**?
       POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
       Cost-aware POI = POI / cost
       
       Or **UCB** (Upper Confidence Bound for minimization, which is Lower Confidence Bound):
       LCB = mu - kappa * sigma
       We want to maximize this? No, LCB is a lower bound on the function value. We want to find low values.
       So we want to minimize LCB? No, in BO we maximize an acquisition function.
       So Acquisition = -LCB = kappa * sigma - mu.
       Cost-aware: (kappa * sigma - mu) / cost?
       This encourages exploration (high sigma) and exploitation (low mu).
       
       Let's try **UCB** with a tuned kappa. A common choice is kappa=1 or kappa=2.
       Since we have 5 instances and need to be robust, let's try kappa=2.
       
       Acquisition = (kappa * sigma - mu) / cost
       
       Wait, the seed is EI/cost. EI is generally considered superior to UCB in practice for BO.
       
       Let's try a **Hybrid** or a **Different EI variant**.
       
       How about **Expected Improvement with a cost-aware threshold**?
       No, that's complex.
       
       Let's try **Maximizing the ratio of Improvement to Cost**, but using a **softened cost**.
       The cost is 1 + mean(x). The "mean(x)" part is the variable part.
       Maybe we should penalize the variable part more?
       cost_var = cost - 1 = mean(x_scaled). Range [0,1].
       Acquisition = EI / (1 + cost_var) = EI / cost. This is the same.
       
       What if we use **EI^2 / cost**? This would favor larger improvements more strongly.
       
       Let's try **EI / (cost ** 0.5)****. This reduces the penalty for high cost, encouraging exploration in high-cost regions if the EI is high enough.
       
       Or **EI / (1 + 0.5 * cost)**?
       
       Let's go with a **UCB** approach as a distinct family from EI.
       UCB (for minimization, maximizing acquisition):
       Acq = kappa * sigma - mu
       We need to make it cost-aware.
       Acq_cost = (kappa * sigma - mu) / cost
       
       Let's choose kappa. If kappa is too small, it's exploitation. If too large, exploration.
       With length scale 0.2 and 2000 candidates, we have fine granularity.
       A moderate kappa like 1.0 or 2.0 is typical.
       
       Let's try kappa = 2.0.
       
       However, UCB can be unstable if sigma is very large (early on).
       EI is more stable.
       
       Let's try **Expected Improvement** but with a **different cost weighting**.
       The seed uses 1/cost.
       Let's try **1 / (1 + cost)**? No, cost is already >=1.
       
       How about **EI * (1 / cost)** is the seed.
       
       Let's try **EI - lambda * cost**.
       We need to calibrate lambda.
       EI is in objective units (Rastrigin values).
       Cost is in [1,2].
       If f_best is around 10, EI might be around 1-5.
       lambda * cost would be lambda * [1,2].
       If lambda = 0.1, penalty is 0.1-0.2. This is small.
       If lambda = 1.0, penalty is 1-2. This is significant.
       
       Since the cost variation is only 2x, and EI variation is likely larger, a multiplicative penalty (1/cost) might be better than additive.
       
       Let's try a **Power Law** on the cost:
       Acquisition = EI / (cost ** 0.5)
       This is less penalizing than 1/cost for high costs, and same for low costs (if cost=1, 1/1=1).
       Actually, if cost=1, 1/1^0.5 = 1. If cost=2, 1/2 = 0.5 vs 1/sqrt(2) = 0.707.
       So this encourages exploring high-cost regions more than the seed.
       
       Let's try this: **EI / sqrt(cost)**
       
       This is a simple change from the seed (EI / cost) to a new approach (EI / sqrt(cost)).
       It belongs to the "Cost-Adjusted Improvement" family but with a different exponent.
       
       Another option: **UCB** is a distinct family.
       Let's implement UCB.
       
       def acquisition(mu, sigma, f_best, cost):
           kappa = 2.0
           # LCB = mu - kappa * sigma
           # We want to minimize f, so we want low LCB.
           # But we maximize acquisition.
           # So Acquisition = -LCB = kappa * sigma - mu
           # Normalize by cost?
           # (kappa * sigma - mu) / cost
           
           # However, mu can be negative? No, Rastrigin is non-negative.
           # But mu is the GP mean, which can be anything.
           # If mu is very negative, kappa*sigma - mu is very positive. Good.
           # If mu is very positive, it's negative.
           
           # One issue: UCB doesn't use f_best. It's a relative measure.
           # EI uses f_best.
           
           # Let's stick to EI-based but modify the cost term.
           # Let's try **EI / (1 + cost)**? No.
           
           # Let's try **Expected Improvement** with **cost-aware sigma**?
           # No.
           
           # Let's try **Max-Probability of Improvement** (MPI) which is similar to EI but simpler.
           # MPI = (f_best - mu) * Phi(z)
           # Cost-aware: MPI / cost
           
           # Let's go with **EI / sqrt(cost)** as a simple, distinct variation.
           # It reduces the penalty for expensive points, potentially allowing the algorithm to explore more broadly if the expected improvement is high.
           
           z = (f_best - mu) / (sigma + 1e-9)
           ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
           
           # Ensure ei is non-negative? EI is theoretically non-negative.
           # Numerical issues might make it slightly negative.
           ei = np.maximum(ei, 0)
           
           # Cost-aware: divide by sqrt(cost)
           return ei / np.sqrt(cost)
       
       Let's double check if this is "from scratch" and "new approach family".
       The seed is EI/cost.
       This is EI/sqrt(cost).
       It's a modification of the cost weighting.
       
       Is there a more distinct family?
       **Knowledge Gradient** is too expensive.
       **Thompson Sampling** requires sampling from the GP, which is not provided (only mu, sigma).
       
       **UCB** is distinct.
       Let's try UCB.
       
       def acquisition(mu, sigma, f_best, cost):
           kappa = 1.0
           # For minimization, we want to find low values.
           # UCB for minimization is often defined as mu - kappa * sigma (LCB).
           # We want to maximize the acquisition function.
           # So Acq = kappa * sigma - mu.
           # But we need to incorporate f_best? UCB doesn't.
           # This might be less focused on improvement.
           
           # Let's try **Cost-Weighted UCB**:
           # Acq = (kappa * sigma - mu) / cost
           
           # But what if mu > kappa * sigma? Acq is negative.
           # EI is always >= 0.
           # This might cause issues if all scores are negative.
           # We can add a constant or use max(0, ...).
           
           # Let's try **EI** with a **different cost exponent**.
           # I will go with **EI / (cost ** 0.5)**.
           
           z = (f_best - mu) / (sigma + 1e-9)
           ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
           ei = np.maximum(ei, 0)
           
           return ei / np.sqrt(cost)
       
       This is a safe, simple change that might improve performance by slightly encouraging exploration in higher-cost regions (which might be where the minimum is, if the minimum is moved to the inner 80%, and cost is 1+mean(x), so center is cost ~1.5, corners cost ~1 or 2? No, x in [-5.12, 5.12]. Scaled to [0,1].
       x=-5.12 -> 0. x=5.12 -> 1.
       cost = 1 + mean(x_scaled).
       If x is in the middle (0,0 in scaled), cost = 1.
       If x is at the edge (0,1 or 1,0), cost = 1.5.
       If x is at the corner (1,1), cost = 2.
       
       Wait, the problem says: "minimizer is moved to a uniform point in the inner 80% of each range".
       So the minimum is not at the center necessarily.
       The cost is 1 + mean(x_scaled).
       So low cost is at x_scaled=0 (which is x=-5.12).
       High cost is at x_scaled=1 (which is x=5.12).
       
       If the minimum is moved to a uniform point in the inner 80%, it could be anywhere in [ -4.1, 4.1 ].
       So the minimum could be at low cost or high cost.
       
       The seed EI/cost penalizes high cost strongly.
       EI/sqrt(cost) penalizes high cost less.
       This might allow the algorithm to find minima that are in high-cost regions.
       
       Let's go with this.
"""
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    return ei / np.sqrt(cost)
# EVOLVE-BLOCK-END
