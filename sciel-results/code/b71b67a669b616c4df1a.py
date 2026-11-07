"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    n = len(mu)
    
    # Knowledge Gradient (KG) Acquisition
    # KG(x) = E[ min( min_{y != x} f(y), f(x) ) - min( f_best, min_{y != x} f(y) ) | F ]
    # Approximation using independence assumption between candidates:
    # For a candidate i, the value is min(mu_i, sigma_i * Z_i).
    # The new best is min( f_best, min_{j != i} (mu_j, sigma_j * Z_j) ).
    # Since we don't have correlations, we approximate the distribution of the minimum of other points.
    
    # A common efficient approximation for KG in discrete sets:
    # Score_i = E[ min(f_i, M_{-i}) - min(f_best, M_{-i}) ]
    # where M_{-i} is the minimum of all other points.
    # Under independence, P(M_{-i} > t) = prod_{j != i} P(f_j > t).
    # This is computationally expensive (O(N^2) or O(N^3) if done naively with CDFs).
    
    # Alternative: Use the fact that for minimization, 
    # E[min(X, Y)] = E[X] - E[(X-Y)^+]
    # Here X = min(f_best, f_i) and Y = M_{-i}? No, that's not quite right.
    
    # Let's use a simpler, robust KG approximation often used in practice:
    # The expected improvement of the global best if we evaluate i.
    # Current best: B = f_best.
    # Next best: B' = min(B, f_i, min_{j != i} f_j).
    # Since we evaluate only one point, the "min_{j != i} f_j" is a random variable representing the best of the rest.
    # However, in the context of the acquisition function which scores candidates to pick the next one,
    # we assume we pick i. Then the new best is min(f_best, f_i).
    # But KG accounts for the fact that even if we don't pick i, the "value" of knowing i might be that it reveals
    # the landscape. But here we must pick one.
    
    # Actually, the standard discrete KG for selecting the next point x_i from a set X is:
    # KG(x_i) = E[ min( min_{j != i} f_j, f_i ) - min( min_{j != i} f_j, f_best ) ]
    # Wait, the baseline is the current best known value f_best.
    # So, KG(x_i) = E[ min(f_i, M_{-i}) - min(f_best, M_{-i}) ]
    # where M_{-i} = min_{j \neq i} f_j.
    
    # To compute this efficiently:
    # Let G(t) = P(M_{-i} > t) = prod_{j \neq i} (1 - Phi((t - mu_j)/sigma_j))
    # This is hard to compute for all i.
    
    # Approximation:
    # Assume the "rest" of the points have a distribution characterized by their global min stats?
    # Or, simpler: Assume the other points are fixed at their posterior means? No, that's greedy.
    
    # Let's use a well-known fast approximation for KG:
    # Score_i = E[ max(0, min(f_best, f_i) - min(f_best, M_{-i})) ]? No.
    
    # Let's stick to the definition:
    # KG_i = E[ min(f_i, M_{-i}) - min(f_best, M_{-i}) ]
    # = E[ min(f_i, M_{-i}) ] - E[ min(f_best, M_{-i}) ]
    
    # Note that min(a, b) - min(c, b) = 
    # If b < c: a - c (if a > c? no)
    # Let's expand:
    # min(f_i, M) - min(f_best, M)
    # Case 1: M < f_best. Then min(f_best, M) = M.
    #   Term = min(f_i, M) - M = min(f_i - M, 0).
    # Case 2: M >= f_best. Then min(f_best, M) = f_best.
    #   Term = min(f_i, M) - f_best.
    
    # This expectation is complex.
    
    # Alternative Strategy: 
    # The prompt asks for a "DIFFERENT strategy". 
    # Inspiration 1 (Thompson Sampling) scored 1.42. 
    # Inspiration 2 (Quantile-Guided) scored 1.32.
    # Current (PoI) scored 17.4.
    
    # Let's try a **Lower Confidence Bound (LCB)** variant, which is distinct from PoI/EI.
    # LCB = mu - beta * sigma.
    # Higher score means "more promising" (lower expected value with uncertainty).
    # But we need to return scores where HIGHER is better for selection.
    # So Score = -LCB = -mu + beta * sigma.
    
    # How to choose beta?
    # Adaptive beta based on number of evaluations or spread of sigma?
    # Or fixed beta?
    # Branin is multi-modal. Exploration is key.
    # PoI failed (17.4) likely because it was too greedy or got stuck? 
    # Actually, PoI score of 17.4 is very bad (regret is high). 
    # Wait, "score (lower is better): mean over seeds of simple regret".
    # "search score is seed/value ... 1.0 equals seed program".
    # The current program score is 17.4036. This is a BAD score (high regret).
    # The inspirations have scores 1.42 and 1.32, which are MUCH BETTER (lower regret).
    # So I need to beat 1.32 or at least get close to it.
    
    # Why did PoI fail? 
    # PoI = P(f < f_best). 
    # If f_best is not yet very low, PoI might be high everywhere.
    # If f_best is low, PoI is low everywhere except near minima.
    # But PoI ignores the magnitude of improvement.
    
    # LCB is a strong contender for multi-modal functions if beta is tuned.
    # Let's implement LCB with an adaptive beta.
    # Beta can be scaled by the standard deviation of the observed values or just a constant.
    # Let's try beta = 2.0 or 3.0.
    
    # However, there is a risk: if sigma is small in unexplored areas (due to GP length scale), 
    # LCB might not explore enough. 
    # But the GP has length scale 0.2 on [0,1] scaled coords. 
    # Domain is [-5, 10] x [0, 15]. 
    # Scaled to [0,1]: x1 in [0, 2/3], x2 in [0, 1].
    # Length scale 0.2 is relatively large compared to the domain size (0.66 and 1.0).
    # So the GP will be smooth and sigma will not drop to zero too quickly in unexplored areas.
    
    # Let's try LCB with a beta that decays slightly or is constant.
    # Constant beta = 2.5.
    
    # Score = -mu + beta * sigma.
    
    # Let's add a small safety check for time.
    
    # Calculate LCB scores
    beta = 2.5
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
