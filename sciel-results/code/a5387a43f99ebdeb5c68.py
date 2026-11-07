import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Lookahead Expected Improvement (Approximate).
    
    Standard EI maximizes E[improvement after 1 step].
    This strategy approximates a 2-step lookahead: 
    1. Estimate the new f_best' after evaluating at x.
    2. Estimate the expected improvement at the *best* of the remaining candidates 
       given that new state.
    
    However, a simpler and more robust "Lookahead" proxy in 1D BO with limited info
    is to use the **Expected Improvement of the Improvement** (EII) or simply 
    boost the exploration term based on the potential to reduce the *variance* of 
    the best value, not just the mean.
    
    A distinct family is **Probability of Improvement (PI)** combined with 
    **Expected Simple Regret (ESR)** minimization.
    
    Let's try **Expected Simple Regret (ESR)** minimization directly.
    ESR(x) = E[|f(x) - f*_true|] is hard.
    
    Alternative Distinct Strategy: **Knowledge Gradient (KG)** approximation.
    KG(x) = E[ min( f_best, f(x) ) - min( f_best, f(x) + noise_reduction ) ]
    This is complex.
    
    Let's look at **Thompson Sampling** approximation.
    Sample f(x) ~ N(mu, sigma^2).
    The score should reflect the probability that x is the global minimum 
    among the 2000 candidates.
    
    Score(x) = P( f(x) < min_{j!=x} f(j) )
    
    This is a distinct design family: **Bayesian Optimization via Thompson Sampling 
    surrogate**. Instead of a deterministic score, we simulate the stochastic nature 
    of the function.
    
    Since we must return deterministic scores, we can compute the *expected* 
    rank or the probability of being the minimum.
    
    P(f(x) is min) = E[ 1(f(x) < min_{j} f(j)) ]
    
    For Gaussian variables, this is related to the multivariate normal CDF, which 
    is expensive for 2000 points.
    
    Approximation: 
    P(f(x) < f(j)) = Phi( (mu_j - mu_x) / sqrt(sigma_j^2 + sigma_x^2) )
    P(f(x) is min) ≈ prod_{j != x} P(f(x) < f(j))
    log P = sum_{j != x} log( Phi( (mu_j - mu_x) / sqrt(sigma_j^2 + sigma_x^2) ) )
    
    This is O(N^2) which is 4,000,000 operations. NumPy can handle this in ~0.1s.
    This is a materially different strategy: **Probability of Being the Global Min** 
    (Thompson Sampling proxy).
    """
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # We want to maximize the probability that f(x) is the minimum.
    # P(f(x) < f(j)) for all j.
    # Using the approximation that f(x) and f(j) are independent (which they are not, 
    # but it's a common heuristic for this score) or using the exact pairwise 
    # probability for the minimum.
    
    # Actually, a better approximation for "Probability of Improvement" in a 
    # multi-candidate setting is the **Expected Rank** or just the **Probability 
    # of Improvement over current best** but adjusted for competition.
    
    # Let's use the **Probability of Being the Minimum** among the candidates.
    # P(x is min) = E[ 1(f(x) < min_{y in candidates, y!=x} f(y)) ]
    
    # Approximation:
    # The probability that f(x) is less than f(j) is Phi( (mu_j - mu_x) / sqrt(sigma_j^2 + sigma_x^2) ).
    # Assuming independence of f(x) and f(j) (crude, but fast),
    # P(x is min) ≈ ∏_{j≠x} P(f(x) < f(j))
    
    # To avoid O(N^2) memory if N is huge, we can compute in chunks or just do it 
    # since N=2000 is small. 2000x2000 matrix is 4M floats = 32MB. Fine.
    
    mu_col = mu[:, np.newaxis]
    mu_row = mu[np.newaxis, :]
    sigma_col = sigma_safe[:, np.newaxis]
    sigma_row = sigma_safe[np.newaxis, :]
    
    # z_ij = (mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2)
    # We want P(f_i < f_j) = Phi(z_ij)
    # Note: if i==j, z=0, Phi(0)=0.5. We want to exclude self.
    
    denom = np.sqrt(sigma_col**2 + sigma_row**2)
    # Avoid division by zero
    denom = np.maximum(denom, 1e-8)
    
    z = (mu_row - mu_col) / denom
    
    # Probability that i is less than j
    p_i_less_j = norm.cdf(z)
    
    # Set diagonal to 1 (or ignore) because we multiply over j != i.
    # Actually, if we include j=i, we multiply by 0.5 for each i, which is a 
    # constant factor for all i, so it doesn't change the argmax.
    # But it might underflow.
    # log(p_i_less_j)
    
    log_p = np.log(p_i_less_j + 1e-15)
    
    # Sum over j for each i
    log_prob_min = np.sum(log_p, axis=1)
    
    # The score is the probability of being the minimum.
    # Higher score -> more likely to be the global minimum among candidates.
    # This naturally balances exploration (high sigma makes it harder to be min? 
    # No, high sigma makes the distribution wider. 
    # If sigma_i is high, the probability that f_i < f_j is more sensitive to mu.
    # Actually, if sigma_i is very high, P(f_i < f_j) averages to 0.5.
    # If sigma_i is low and mu_i is low, P(f_i < f_j) is high.
    # So this favors low mu and moderate sigma?
    # Let's check: if mu_i is very low, z is large positive, Phi ~ 1.
    # If mu_i is very high, z is large negative, Phi ~ 0.
    # So it favors low mu.
    # What about sigma?
    # If sigma_i is large, denom is large, z is small, Phi ~ 0.5.
    # If sigma_i is small, denom is small (if sigma_j is also small), z is large.
    # So it favors low mu and low sigma (exploitation) if the model is confident.
    # If the model is uncertain (high sigma everywhere), it's a coin flip.
    
    # This might be too exploitative.
    # However, "Probability of Being the Minimum" is a valid acquisition function 
    # (related to PI but competitive).
    
    # To encourage exploration, we can mix this with a pure exploration term?
    # Or just use it as is. The current best is 3.0 (Adaptive UCB).
    # If this is too greedy, it might get stuck.
    
    # Let's try a variant: **Expected Improvement of the Minimum**.
    # But the "Probability of Being the Minimum" is distinct.
    
    # To prevent underflow/overflow and ensure finite scores:
    scores = log_prob_min
    
    # Normalize to [0, 1] range? Not necessary, just need finite and higher is better.
    # But large negative numbers might cause issues? No, just rank matters.
    
    # One issue: if all sigmas are 0, z is inf.
    # We handled sigma_safe.
    
    # Return scores
    return scores
