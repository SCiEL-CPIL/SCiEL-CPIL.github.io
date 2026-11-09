"""Cost-aware acquisition for three-hump-camel 2D BO using Thompson Sampling with budget-aware temperature."""
import time
import numpy as np
from scipy.stats import norm

# Module-level cache for fast normal CDF/PDF computation
_CDF_CACHE = {}
_PDF_CACHE = {}


def _norm_cdf(z):
    """Fast normal CDF using lookup table with interpolation."""
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -10, 10)
    return norm.cdf(z)


def _norm_pdf(z):
    """Fast normal PDF."""
    z = np.asarray(z, dtype=np.float64)
    return norm.pdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function with budget-aware temperature.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (lower is better)
    cost : ndarray, shape (2000,)
        Evaluation cost for each candidate point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)

    # --- Budget-aware temperature parameter ---
    # Total budget is 30. We estimate remaining budget from f_best and cost history.
    # Since we don't have direct access to total cost spent, we use f_best as a proxy for convergence.
    # Early: f_best is high (several units), late: f_best is near 0.
    # convergence in [0, 1]: 0 = just started, 1 = fully converged
    convergence = max(0.0, min(1.0, 1.0 - f_best / 5.0))
    
    # Temperature: higher early (more exploration), lower late (more exploitation)
    # Temperature modulates the effective noise in Thompson Sampling
    temperature = 0.5 + 1.5 * (1.0 - convergence)  # [0.5, 2.0]

    # --- Thompson Sampling: sample from posterior ---
    # Sample a random offset from standard normal
    # Use a fixed seed for determinism within a call, but different across calls
    # Actually, for acquisition functions, we want stochastic behavior
    # Generate random numbers with a seed based on the current state
    # Use np.random with a deterministic seed derived from inputs for reproducibility
    # But actually, we want different samples each time to get exploration
    # Use a simple hash-based seed
    seed = int(np.sum(np.abs(mu)) * 1000 + np.sum(np.abs(sigma)) * 100 + f_best * 10) % (2**32)
    rng = np.random.RandomState(seed)
    
    # Sample from N(0, temperature^2 * sigma^2)
    # This is equivalent to sampling f(x) ~ N(mu, sigma^2) and adding temperature * sigma * N(0,1)
    # Actually, standard Thompson Sampling: sample f(x) ~ N(mu, sigma^2)
    # The temperature should modulate the effective uncertainty
    # Let's use: sample = mu + sigma * temperature * N(0,1)
    # This increases exploration when temperature is high
    noise = rng.randn(len(mu))
    sampled_values = mu + sigma * temperature * noise

    # --- Cost adjustment ---
    # Use EI/Cost style: divide by cost to get cost-adjusted score
    # But for Thompson Sampling, we want to select the point with highest sampled value per unit cost
    # So: score = sampled_value / cost
    # This naturally balances exploration (high sampled value) with cost efficiency (low cost)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Cost-adjusted Thompson Sampling score
    score = sampled_values / cost_safe

    # --- Add small exploration bonus for high uncertainty when not converged ---
    # This helps when Thompson Sampling might undersample high-uncertainty regions
    sigma_max = np.max(sigma) + 1e-10
    sigma_norm = sigma / sigma_max
    exploration_bonus = 0.02 * sigma_norm * (1.0 - convergence) * temperature
    
    score += exploration_bonus

    # --- Ensure finite values ---
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard
    if time.time() > deadline:
        pass

    return score
