# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0  # Leave margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition for Bayesian optimization of Beale 2D.
    
    Uses a deterministic approximation of Thompson Sampling by computing
    expected improvement from sampling the GP posterior. This naturally balances
    exploration and exploitation without hand-tuned constants.
    
    The score is based on the expected value of the minimum of multiple samples
    from the GP posterior, which encourages exploration where uncertainty is high
    and exploitation where the mean is low.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Emergency fallback: pure exploitation
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling approximation:
    # Sample from the GP posterior: f(x) ~ N(mu(x), sigma(x)^2)
    # For minimization, we want to find x that minimizes f(x)
    # The expected minimum of n samples from N(mu, sigma^2) is:
    # E[min] = mu - sigma * E[min of n standard normals]
    # 
    # We use a fixed number of samples n=5 for computational efficiency
    # E[min of n standard normals] can be approximated using the inverse CDF
    # For n=5, the expected minimum of standard normals is approximately -1.163
    # More precisely, we can compute this using the formula for expected order statistics
    
    n_samples = 5
    # Expected minimum of n standard normal samples
    # This is E[min(Z1, ..., Zn)] where Zi ~ N(0,1)
    # For n=5, this is approximately -1.163
    # We can compute it more precisely using numerical integration or known values
    # For simplicity, use the approximation that works well for n=5
    
    # Expected minimum of n i.i.d. N(0,1) samples
    # Can be computed as: integral of (1 - Phi(x))^n * phi(x) dx from -inf to inf
    # But for efficiency, use precomputed values or approximation
    
    # For n=5, E[min] ≈ -1.163
    # For n=3, E[min] ≈ -0.846
    # For n=10, E[min] ≈ -1.539
    
    # Use a lookup or formula. For n=5:
    expected_min_std_normal = -1.163
    
    # The expected minimum of n samples from N(mu, sigma^2) is:
    # mu + sigma * expected_min_std_normal
    # (since scaling by sigma and shifting by mu)
    
    expected_min = mu + sigma_safe * expected_min_std_normal
    
    # For minimization, we want to maximize the score
    # The score should be higher where the expected minimum is lower
    # So: score = -expected_min = -mu - sigma * expected_min_std_normal
    # Since expected_min_std_normal is negative, this becomes:
    # score = -mu + sigma * |expected_min_std_normal|
    
    # This is equivalent to UCB with beta = |expected_min_std_normal| = 1.163
    # But the key insight is that this comes from a principled Thompson Sampling
    # approximation rather than hand-tuning
    
    # However, pure UCB with fixed beta may not adapt well to optimization progress
    # Let's use an adaptive version where the effective number of samples changes
    # based on how close we are to the optimum
    
    # When f_best is large (early), use more samples (more exploration)
    # When f_best is small (late), use fewer samples (more exploitation)
    
    f_best_clamped = max(f_best, 1e-12)
    
    # Adaptive number of samples: more when far from optimum
    # n = 2 + 3 / (1 + f_best)
    # At f_best=0: n=5 (max exploration)
    # At f_best=10: n=2.3
    # At f_best=100: n=2.03
    
    n_adaptive = 2.0 + 3.0 / (1.0 + f_best_clamped)
    
    # Expected minimum of n standard normals as a function of n
    # Use a smooth approximation
    # For integer n: n=1: 0, n=2: -0.564, n=3: -0.846, n=4: -1.029, n=5: -1.163
    # For non-integer n, interpolate smoothly
    
    # Approximation: E[min of n N(0,1)] ≈ -sqrt(2 * log(n)) for large n
    # But for small n, use a different formula
    # Let's use a piecewise linear interpolation for n in [1, 10]
    
    n_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    e_min_vals = np.array([0.0, -0.564, -0.846, -1.029, -1.163, -1.267, -1.355, -1.431, -1.497, -1.539])
    
    # Clamp n_adaptive to valid range
    n_clamped = np.clip(n_adaptive, 1.0, 10.0)
    
    # Interpolate
    e_min_adaptive = np.interp(n_clamped, n_vals, e_min_vals)
    
    # Score: -mu - sigma * e_min_adaptive
    # Since e_min_adaptive is negative, this is -mu + sigma * |e_min_adaptive|
    score = -mu - sigma_safe * e_min_adaptive
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
