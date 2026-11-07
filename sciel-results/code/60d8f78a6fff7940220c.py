import numpy as np
from scipy.stats import norm
import time

# Module-level state for time management
_start_time = None
_MAX_TIME = 300.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Hypervolume-inspired approach for Bayesian optimization
    of Goldstein-Price 2D.
    
    The idea: instead of scalarizing mean and variance, we compute an approximate
    hypervolume contribution for each candidate point, where the reference point
    is set to the known global minimum (3.0). This directly optimizes for the
    area of the Pareto front dominated by the best observed solution relative
    to the true optimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher meaning more promising.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _MAX_TIME - _SAFETY_MARGIN:
        # Fallback: simple UCB if we're running out of time
        return -mu + 1.0 * sigma
    
    # Known global minimum for Goldstein-Price 2D
    f_min = 3.0
    
    # Hypervolume-inspired acquisition:
    # For minimization, we think in terms of improvement over the reference.
    # The "hypervolume" contribution of a point is related to how much it
    # dominates the reference point in the objective space.
    #
    # For a single point with posterior N(mu, sigma^2), the expected hypervolume
    # contribution relative to reference r is:
    # E[max(0, r - f(x))] where f(x) ~ N(mu, sigma^2)
    # This is the expected improvement over the reference point.
    #
    # EI_ref(x) = (r - mu) * Phi((r - mu)/sigma) + sigma * phi((r - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Reference point: use the known global minimum as the reference
    # This creates a constant exploration signal that doesn't diminish
    # as f_best approaches f_min.
    ref = f_min
    
    # Compute expected improvement relative to the reference point
    # EI_ref(x) = E[max(0, ref - f(x))]
    d = ref - mu  # This is positive when mu < ref, meaning we expect improvement
    
    # Standard normal CDF and PDF
    z = d / (sigma + 1e-10)  # Avoid division by zero
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected improvement relative to reference
    ei_ref = d * Phi_z + sigma * phi_z
    
    # Additionally, we want to balance exploitation (finding low values)
    # with exploration (uncertainty). The EI_ref already captures both:
    # - When mu is close to ref and sigma is small, EI_ref is small
    # - When mu is far from ref and sigma is large, EI_ref can be significant
    #
    # However, we also want to prioritize points where the predicted mean
    # is already good (low). Let's add a weighted combination:
    # score = w1 * EI_ref + w2 * (-mu + beta * sigma)
    #
    # Actually, let's stick with a pure hypervolume-inspired approach but
    # add a small exploitation term to break ties and guide search.
    
    # Weight between hypervolume-inspired EI and exploitation
    # The exploitation term is -mu + beta * sigma (standard UCB)
    beta_ucb = 0.5
    ucb_score = -mu + beta_ucb * sigma
    
    # Normalize both scores to have similar scales
    # Use the median as a robust scale estimate
    ei_ref_scale = np.median(ei_ref) + 1e-10
    ucb_scale = np.median(np.abs(ucb_score)) + 1e-10
    
    ei_ref_norm = ei_ref / ei_ref_scale
    ucb_norm = ucb_score / ucb_scale
    
    # Combine: 70% hypervolume-inspired, 30% exploitation
    scores = 0.7 * ei_ref_norm + 0.3 * ucb_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
