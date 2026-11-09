"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: return a safe default if we're running out of time
    # We don't know the exact deadline, but we can track relative time
    # Since this is called repeatedly, we'll use a simple heuristic:
    # If computation takes too long, return a reasonable default
    
    start_time = time.time()
    
    # For minimization:
    # We want high score when mu is low (close to f_best) and sigma is high (uncertain)
    # Expected Improvement (EI) for minimization:
    # EI = max(f_best - mu, 0) - sigma * norm.pdf((f_best - mu)/sigma)
    # But EI can be negative or zero, and we want higher = better
    
    # Alternative: Use a combination of exploitation and exploration
    # Exploitation: how close mu is to f_best (lower is better)
    # Exploration: how high sigma is (higher is better)
    
    # Normalize sigma to make it comparable
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    if sigma_max < 1e-10:
        sigma_max = 1.0
    
    # Exploitation score: higher when mu is lower
    # Use negative mu so lower mu gives higher score
    exploitation = -mu
    
    # Exploration score: higher when sigma is higher
    exploration = sigma / sigma_max
    
    # Combine with a balance factor
    # beta controls exploration vs exploitation
    # For early optimization, we want more exploration
    # For later, more exploitation
    
    # Since we don't know the step, use a fixed balance
    # The current UCB approach uses beta=2.0
    # Let's try Expected Improvement which is more principled
    
    # EI for minimization:
    # If mu < f_best, EI = (f_best - mu) * Phi((f_best-mu)/sigma) + sigma * phi((f_best-mu)/sigma)
    # If mu >= f_best, EI = sigma * phi((f_best-mu)/sigma)  (small positive)
    
    # Simplified EI:
    z = (f_best - mu) / (sigma + 1e-10)
    
    # Use scipy if available, otherwise approximate
    try:
        from scipy.stats import norm
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
    except ImportError:
        # Approximate normal CDF and PDF
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        # Approximate CDF using error function
        Phi = 0.5 * (1 + np.tanh(z * np.sqrt(2 / np.pi)))
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # EI is always non-negative, higher is better
    # But EI can be dominated by sigma when mu is far from f_best
    # Let's also add a small exploitation term
    
    # Actually, let's just use EI directly
    # It naturally balances exploration and exploitation
    
    elapsed = time.time() - start_time
    
    # If computation took too long, we might want to return a simpler score
    # But EI is already computed, so just return it
    return ei
# EVOLVE-BLOCK-END
