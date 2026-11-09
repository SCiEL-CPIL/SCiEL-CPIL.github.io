import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Sharpened Expected Improvement
    # Temperature tau < 1 makes the acquisition more exploitative/greedy
    # Temperature tau > 1 makes it more exploratory
    tau = 0.5 
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Standard EI is ((f_best - mu) * cdf(z) + sigma * pdf(z))
    # With temperature tau, we scale the argument to the normal distribution
    # This effectively changes the "softness" of the improvement estimate
    # Note: The standard derivation assumes the predictive distribution is N(mu, sigma).
    # Sharpening is a heuristic modification.
    
    z_scaled = z / tau
    cdf_val = norm.cdf(z_scaled)
    pdf_val = norm.pdf(z_scaled)
    
    # The term (f_best - mu) * cdf + sigma * pdf is the expected improvement.
    # When we change z to z/tau, we are effectively evaluating the expectation
    # under a distribution with std dev sigma*tau? No, it's a heuristic.
    # Let's stick to a simpler heuristic: Weight the uncertainty term.
    # EI = Exploit + Explore
    # Exploit = (f_best - mu) * cdf(z)
    # Explore = sigma * pdf(z)
    # Let's reduce the Explore term.
    
    exploit = (f_best - mu) * norm.cdf(z)
    explore = sigma * norm.pdf(z)
    
    # Reduce exploration
    alpha = (exploit + 0.5 * explore) / cost
    
    return alpha
