import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Knowledge Gradient (KG) acquisition for minimization
    # KG approximates the expected reduction in the minimum value after the next evaluation.
    # For a single candidate x, KG(x) = E[min(f_best, f(x)) - f_best]
    # Since we want to maximize the expected improvement, we compute:
    # KG(x) = E[max(0, f_best - f(x))] = EI(x)
    # But KG is more nuanced. For a single point, it reduces to EI.
    # However, the true KG considers the value of information for the entire search.
    # A practical approximation for KG in the context of GP-BO is to use a modified EI
    # or to use the fact that KG is related to the probability of improvement and the expected improvement.
    
    # A common approximation for KG when we only have mu and sigma is to use EI.
    # EI(x) = E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu(x), sigma(x)), then:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # To make it more like KG, we can use a slightly different formulation.
    # Some sources suggest that for a single evaluation, KG is equivalent to EI.
    # Let's implement EI with a small epsilon to avoid division by zero.
    
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
