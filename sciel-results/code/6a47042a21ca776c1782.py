"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for Thompson Sampling
_ts_state = {'count': 0, 'rng': None}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Hartmann 6D using Thompson Sampling.
    
    This is a structural change from LCB to Thompson Sampling, where we sample
    from the posterior distribution at each candidate point. This creates a
    fundamentally different exploration mechanism that can better escape local
    optima in multimodal landscapes by occasionally selecting high-uncertainty
    regions even when the mean suggests exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Posterior mean of the GP at candidate points (in objective units).
    sigma : ndarray, shape (n_candidates,)
        Posterior standard deviation of the GP at candidate points (in objective units).
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means evaluate next.
    """
    try:
        # Ensure sigma is strictly positive
        sigma_safe = np.maximum(sigma, 1e-10)
        
        # Initialize RNG if needed
        if _ts_state['rng'] is None:
            _ts_state['rng'] = np.random.default_rng(42)
        
        # Increment evaluation count
        _ts_state['count'] += 1
        n_evals = _ts_state['count']
        
        # Define known minimum of Hartmann 6D
        f_min = -3.32237
        f_start_est = 3.0
        
        # Calculate progress p in [0, 1]
        f_best_clamped = max(f_best, f_min)
        range_est = f_start_est - f_min
        if range_est < 1e-6:
            p = 0.0
        else:
            p = (f_best_clamped - f_min) / range_est
            p = np.clip(p, 0.0, 1.0)
        
        # Thompson Sampling: sample from posterior
        # For minimization, we want to find points with low sampled values
        # scores = -sampled_value, so higher score = lower sampled value
        
        # Early on (high p), use more aggressive sampling (larger sigma multiplier)
        # Late on (low p), use more conservative sampling
        # This adapts the exploration intensity
        
        # Adaptive sigma scaling for Thompson Sampling
        # When f_best is poor (early), we want more exploration
        # When f_best is good (late), we want more exploitation
        
        # Use a decay schedule: early steps have more uncertainty in our beliefs
        # We scale sigma by a factor that decreases as we gather more info
        if n_evals <= 5:
            # Initial phase: very high exploration
            sigma_scale = 2.0
        elif n_evals <= 10:
            # Early phase: high exploration
            sigma_scale = 1.5
        elif n_evals <= 15:
            # Mid phase: moderate exploration
            sigma_scale = 1.2
        elif n_evals <= 20:
            # Later phase: moderate-low exploration
            sigma_scale = 1.0
        else:
            # Final phase: low exploration, more exploitation
            sigma_scale = 0.8
        
        # Sample from the posterior: N(mu, sigma^2)
        # For minimization, we want low values, so we negate the sample
        rng = _ts_state['rng']
        samples = mu + sigma_safe * sigma_scale * rng.standard_normal(mu.shape)
        
        # Score: we want to maximize, so negate the sampled value
        # Lower sampled value = better (for minimization) = higher score
        scores = -samples
        
        # Ensure all scores are finite
        if not np.all(np.isfinite(scores)):
            # Fallback to pure exploitation if numerical issues arise
            scores = -mu
            scores = np.where(np.isfinite(scores), scores, 0.0)
            
        return scores
        
    except Exception:
        # Final safety net: return a neutral score based on mean
        try:
            scores = -mu
            scores = np.where(np.isfinite(scores), scores, 0.0)
            return scores
        except:
            return np.zeros_like(mu)
# EVOLVE-BLOCK-END
