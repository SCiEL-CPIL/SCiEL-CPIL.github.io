import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Computes acquisition scores using Cost-Adjusted Expected Improvement (EI).
    
    Strategy:
    1. Standardize the mean and standard deviation relative to the current best.
    2. Compute the Expected Improvement (EI) for each candidate.
       EI = E[max(f_best - f(x), 0)]
       This provides a deterministic measure of the expected gain over the current best.
    3. Apply an adaptive exploitation parameter xi.
       xi scales with f_best to ensure that in the early stages (high f_best), 
       we require a larger expected improvement to justify the cost, 
       and in late stages (low f_best), we are more sensitive to small improvements.
       Specifically, we compute EI for (f_best - xi) instead of f_best.
       This effectively shifts the threshold for "improvement" and can be tuned
       to balance exploration vs exploitation. A larger xi makes the function
       more exploratory (it ignores candidates that only slightly beat f_best).
       However, for minimization, standard EI is max(f_best - f, 0).
       To make it more exploratory, we can use a "soft" EI or adjust xi.
       
       Let's use a standard EI but with a cost-aware normalization.
       Score = EI / cost.
       
       To introduce a structural change from TS:
       TS is stochastic. EI is deterministic.
       Also, we will implement an adaptive xi that starts small and grows, 
       or simply use a fixed reasonable xi relative to the scale of the problem.
       Given the range of Rastrigin is [0, ~200], and we start with random points,
       f_best might be high initially.
       
       A robust cost-aware metric is:
       score = EI / (cost + epsilon)
       
       To encourage exploration in the early stages when f_best is not yet good,
       we can add a small exploration bonus based on sigma, but EI already captures
       the potential. 
       
       Let's stick to pure Cost-Adjusted EI with an adaptive xi.
       xi = 0.01 * max(1, abs(f_best))
       This ensures that if f_best is 100, xi is 1. If f_best is 0.1, xi is 0.01.
       This prevents the algorithm from getting stuck in local minima by ignoring
       very small expected improvements that aren't worth the cost in late stages,
       but in early stages, it allows for broader search.
       
       Actually, for minimization, EI is defined as:
       EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
       where z = (f_best - mu(x)) / sigma(x)
       
       If we want to be more exploratory, we can reduce the threshold.
       But EI is already a strong exploit metric.
       
       Let's try a hybrid: Cost-Adjusted EI with a small exploration term.
       Or simply Cost-Adjusted EI. The stochasticity of TS was the main differentiator.
       Deterministic EI is often more stable.
       
       To ensure we don't just exploit the current best immediately, we can use
       a "Look-ahead" or "Optimistic" EI by using a slightly pessimistic mean?
       No, standard EI is good.
       
       Let's implement standard Cost-Adjusted EI.
    """
    start_time = time.time()
    deadline = 330.0
    
    # Ensure arrays are numpy float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # f_best is a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # Check time limit
    if time.time() > deadline - 5.0:
        # Fallback: simple LCB
        return (1.5 * sigma - mu) / cost

    # Adaptive exploitation parameter xi
    # Scale xi with the current best value to maintain relative sensitivity
    # If f_best is large (early search), xi is larger, requiring bigger improvements.
    # If f_best is small (late search), xi is smaller, allowing finer improvements.
    # This helps avoid premature convergence by not chasing tiny gains that cost too much.
    xi = 0.05 * max(1.0, abs(f_best))
    
    # Calculate z-score for EI
    # z = (f_best - xi - mu) / sigma
    # We use (f_best - xi) as the threshold. If mu > f_best - xi, the improvement is likely negative or small.
    # Standard EI uses f_best. Using f_best - xi makes it "optimistic" in a way? 
    # No, EI(x) = E[max(f_best - f(x), 0)].
    # If we use threshold T = f_best - xi, then we are calculating E[max(T - f(x), 0)].
    # Since T < f_best, this EI will be smaller than standard EI.
    # This effectively penalizes candidates that don't beat f_best by at least xi.
    # This is a good way to control exploitation.
    
    T = f_best - xi
    
    z = (T - mu) / sigma
    
    # Handle numerical stability in division
    z = np.clip(z, -10, 10) # Clip to avoid overflow in exp/phi
    
    # Phi(z) is the CDF of standard normal
    # phi(z) is the PDF of standard normal
    # scipy.stats.norm.cdf and pdf
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (T - mu) * Phi(z) + sigma * phi(z)
    # Note: If T < mu, (T-mu) is negative. Phi(z) will be small.
    # The term sigma * phi(z) is always positive.
    # This formula is valid for any z.
    
    ei = (T - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (due to numerical errors, it might be slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted score
    # We want to maximize EI per unit cost.
    scores = ei / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
