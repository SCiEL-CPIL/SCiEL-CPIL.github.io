import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a hybrid approach:
    1. Compute Expected Improvement (EI) for exploitation.
    2. Compute Upper Confidence Bound (UCB) for exploration.
    3. Combine them with an adaptive weight that shifts from exploration to exploitation
       as the number of evaluations increases (estimated via f_best improvement or just fixed schedule).
       
    Since we don't have the exact evaluation count, we use a heuristic:
    - Early on: more exploration (higher UCB weight)
    - Later: more exploitation (higher EI weight)
    
    However, a simpler and robust approach is to use a fixed combination or a
    well-tuned UCB. Given the stagnation directive, let's try a structural change:
    Use a Gaussian Process posterior to compute a more sophisticated acquisition
    that accounts for the shape of the distribution, not just mean and std.
    
    Actually, the task says: "mu and sigma ... are in objective units."
    And we only have mu and sigma for 2000 points. We don't have the GP kernel or data.
    So we can only use functions of mu and sigma.
    
    Let's try a different structural idea: 
    Instead of linear UCB, use a non-linear combination that better handles the
    relationship between uncertainty and improvement.
    
    One such method is the "Probability of Improvement" (PI) or a variant.
    But PI tends to be too exploitative.
    
    Let's try "Knowledge Gradient" (KG) approximation. For a single point,
    KG is related to the expected improvement in the best value.
    A simple approximation for KG in 1D (per point) is:
    KG ≈ E[max(f_best - f(x), 0)] = EI
    But this ignores the uncertainty in f_best.
    
    Another idea: Use a "safe" UCB that ensures the candidate is likely to be better than f_best.
    score = -mu + beta * sigma
    But constrain it so that we don't pick points with very low probability of improvement.
    
    Let's try a different structural change: 
    Use a "random restart" style acquisition? No, that's not an acquisition function.
    
    What about using the ratio of improvement to uncertainty?
    score = (f_best - mu) / (sigma + epsilon)
    This is like a t-statistic for improvement.
    
    Or, use a "hybrid" of EI and UCB:
    score = w * EI + (1-w) * UCB
    
    Let's implement EI properly:
    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    And UCB = -mu + beta * sigma
    
    Then combine: score = alpha * EI + (1-alpha) * UCB
    
    We need to choose alpha and beta.
    Given the stagnation, let's try a fixed alpha=0.5, beta=1.0.
    
    But wait, the current program scores 1.5379 with beta=1.0 UCB.
    The inspiration 3 with beta=1.5 scored 1.0933 (worse).
    So lower beta is better?
    
    Let's try the EI + UCB hybrid.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # For minimization, EI = E[max(f_best - f, 0)]
    # where f ~ N(mu, sigma^2)
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB
    beta = 1.0
    UCB = -mu + beta * sigma
    
    # Normalize EI and UCB to have similar scales before combining
    # EI can be zero or small, UCB can be large.
    # Let's use a simple weighted sum.
    # Based on previous results, UCB with beta=1.0 worked well.
    # Let's try a 50/50 mix, or even 70/30 in favor of UCB.
    alpha = 0.5  # weight for EI
    scores = alpha * EI + (1 - alpha) * UCB
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
