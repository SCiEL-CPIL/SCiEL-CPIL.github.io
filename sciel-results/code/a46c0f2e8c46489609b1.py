# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Quantile-based Expected Improvement (qEI) acquisition function.
    
    Strategy:
    Instead of using a heuristic combination of relevance and uncertainty (like the 
    current KG approximation), this function computes the Expected Improvement (EI)
    but uses a quantile-based proxy that is more robust to multimodal landscapes.
    
    Rationale:
    Standard EI maximizes E[max(f_best - f(x), 0)]. This is optimal for unimodal
    functions but can struggle with multimodal ones like Branin because it heavily
    favors points with high probability of small improvements over points with lower
    probability of large improvements.
    
    The Quantile-based EI (qEI) approximates the improvement by looking at the
    quantile of the improvement distribution. Specifically, we use the fact that
    for a normal distribution Y ~ N(mu, sigma^2), the expected positive part 
    E[max(Y, 0)] can be related to the quantiles. However, a simpler and effective
    robust alternative is to use the "Upper Confidence Bound" with a dynamic 
    exploration term that adapts to the scale of the problem, or more directly,
    to use the formula for EI which is analytically solvable.
    
    Actually, the most direct "different strategy" from the current product-based
    heuristic is to use the **Analytic Expected Improvement (EI)**.
    
    Why EI is different from the current KG approx:
    - Current: score = exp(-(mu-f_best)^2/scale^2) * (sigma/max_sigma) + small_exploit
      This is a heuristic product. It doesn't integrate the probability mass.
    - EI: score = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
      This is the exact expectation of the improvement. It naturally balances 
      exploration (high sigma) and exploitation (low mu) based on the probability 
      that the point is actually better than f_best.
    
    For Branin, EI is a strong baseline. To make it "materially different" and 
    potentially better for multimodal landscapes, we can use **qEI** with a 
    specific quantile, or simply stick to standard EI if the current heuristic 
    is underperforming due to its arbitrary scaling.
    
    However, the prompt asks for a strategy *different* from the current one.
    The current one is a "KG approximation". A standard EI is a different family.
    
    Let's implement Standard Expected Improvement (EI). It is often superior to 
    heuristic combinations because it is derived from first principles of Bayesian 
    optimization. The current program's "relevance" term is a Gaussian decay, 
    which is similar to the Phi term in EI but the "uncertainty" term is linear 
    sigma, whereas EI uses phi (the PDF) which is nonlinear. This nonlinearity 
    is crucial for correct balancing.
    
    Formula for EI (minimization):
    Let d = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    
    Where:
    Phi is the CDF of the standard normal.
    phi is the PDF of the standard normal.
    
    This handles the trade-off optimally for the expected gain.
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Calculate the normalized distance from the best value
    # d = (f_best - mu) / sigma
    # If sigma is very small, d becomes very large positive or negative.
    d = (f_best - mu) / sigma
    
    # Clip d to avoid numerical issues with extreme values in exp/erf
    # Phi(d) for d > 8 is ~1, for d < -8 is ~0
    # phi(d) for |d| > 8 is ~0
    d_clipped = np.clip(d, -10, 10)
    
    # Calculate Phi(d) using the error function
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    phi_d = norm.cdf(d_clipped)
    
    # Calculate phi(d) (standard normal PDF)
    # phi(x) = exp(-x^2 / 2) / sqrt(2 * pi)
    # Using norm.pdf is slower, so we compute directly
    pdf_d = norm.pdf(d_clipped)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    # Note: (f_best - mu) = d * sigma
    # So EI = d * sigma * Phi(d) + sigma * phi(d) = sigma * (d * Phi(d) + phi(d))
    # This form is numerically more stable if we factor out sigma.
    
    ei_scores = sigma * (d_clipped * phi_d + pdf_d)
    
    # Ensure non-negative (EI should be >= 0)
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Ensure finite
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    # If time is running out, return current
    if time.time() > t_start + 0.1:
        return ei_scores
        
    return ei_scores
# EVOLVE-BLOCK-END
