"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # 330s limit, 5s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Adaptive LCB acquisition function for minimization.
    
    Structural Change: Replaces standard EI with an Adaptive Lower Confidence Bound (LCB)
    where the exploration coefficient (beta) scales with the mean uncertainty of the model.
    This helps avoid premature convergence in multimodal landscapes like Hartmann 3D.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return safe finite scores to avoid hanging
        return np.random.rand(len(mu))

    # Convert to numpy arrays if not already
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical safeguards
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Calculate adaptive beta
    # Base exploration rate
    beta_base = 1.5
    
    # Scale of the objective function for normalization
    # Hartmann 3D values are roughly in [-4, 0]. 
    # We use the spread of mu to estimate the current scale of interest.
    mu_spread = np.max(mu) - np.min(mu)
    if mu_spread < 1e-6:
        mu_spread = 1.0
    
    # Mean uncertainty
    sigma_mean = np.mean(sigma)
    
    # Adaptive component: increases beta when average uncertainty is high relative to the function scale
    # This encourages exploration in the early stages or when the model is unsure
    beta_adapt = 0.5 * (sigma_mean / mu_spread)
    
    # Clamp beta to prevent extreme exploration
    beta = np.clip(beta_base + beta_adapt, 1.0, 3.0)

    # LCB Score for minimization: We want to maximize the lower bound of the function value.
    # Lower bound estimate: mu - beta * sigma
    # To maximize this, we maximize: -mu + beta * sigma
    lcb_scores = -mu + beta * sigma

    # Cost-normalized score
    scores = lcb_scores / cost

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    # If all scores are invalid (should not happen), return zeros
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)

    return scores
# EVOLVE-BLOCK-END
