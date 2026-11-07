import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Combines the structural exploration bias of Probability of Improvement (PI)
    with the exploitation signal of Expected Improvement (EI).
    
    The PI term provides a robust exploration bias that favors points likely to
    surpass the current best, regardless of magnitude. The EI term provides the
    expected magnitude of improvement. Blending them creates a smoother
    acquisition surface than pure EI, which can be too greedy or spiky.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score: (f_best - mu) / sigma
    # For minimization, we want f(x) < f_best
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    pi = np.where(np.isfinite(pi), pi, 0.0)

    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where phi is the standard normal PDF
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)

    # Normalize EI to be comparable to PI (which is in [0, 1])
    # Use f_best as a natural scale for the objective function
    # If f_best is very small, we use a small floor to avoid division by zero
    # This ensures that when f_best is large, EI is scaled down to match PI's range
    scale = max(abs(f_best), 1e-6)
    ei_normalized = ei / scale

    # Blend PI and EI
    # PI provides the exploration bias (structural change from pure EI)
    # EI provides the exploitation signal (refinement machinery)
    # A 50/50 blend is a robust starting point, but we can weight PI slightly
    # higher to ensure sufficient exploration in the early stages
    score = 0.5 * ei_normalized + 0.5 * pi

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard: if computation took too long, return a safe fallback
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return EI only (fastest path)
        return ei

    return score
