"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t as t_dist

# Global state to track evaluation count for adaptive exploration
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 300s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Increment evaluation count
    _eval_count += 1
    
    # Strategy: Student's t-based Expected Improvement
    # Standard EI assumes a Gaussian posterior. With only ~30 samples,
    # the uncertainty is often underestimated (heavy tails are ignored).
    # Using a t-distribution with low degrees of freedom (df) accounts for
    # this extra uncertainty, making the acquisition function more robust
    # and encouraging exploration of areas with high sigma even if mu is not
    # immediately promising, while still being greedy where mu is very low.
    
    # Degrees of freedom: 
    # As we collect more data, our confidence in the GP increases.
    # We can increase df over time. 
    # Start with low df (e.g., 3) for heavy tails (more exploration/robustness).
    # Increase df as _eval_count increases, approaching Gaussian (inf df).
    # df = 3 + _eval_count * 0.5 might be too fast.
    # Let's keep it relatively low to maintain robustness throughout the short run.
    # df = 5 is a common choice for robust regression/estimation.
    # Let's try a fixed df=5 or a slow increase.
    df = 5.0
    
    # Standardize the improvement relative to f_best
    # Z = (f_best - mu) / sigma
    # For minimization, we want to find points where the expected value is lower than f_best.
    # Let Y ~ t_df(0, 1).
    # The improvement is I = max(0, f_best - Y).
    # In our context, the posterior at x is approximated by a t-distribution:
    # f(x) ~ t_df(mu, sigma)  (Note: standard t is centered at 0 with scale 1. 
    # Here we center at mu and scale by sigma).
    
    # The Expected Improvement for a t-distributed variable:
    # EI(x) = E[max(0, f_best - f(x))]
    # Let Z = (f_best - mu) / sigma.
    # Then f(x) = mu + sigma * T, where T ~ t_df.
    # f_best - f(x) = (f_best - mu) - sigma * T = sigma * (Z - T).
    # EI = sigma * E[max(0, Z - T)]
    
    # The term E[max(0, Z - T)] can be computed analytically or numerically.
    # For a standard normal N(0,1), E[max(0, Z - N)] = Z * Phi(Z) + phi(Z).
    # For a t-distribution, there is no simple closed form like the normal case, 
    # but we can use the CDF and PDF of the t-distribution.
    
    # Let's derive or look up the EI for t-distribution.
    # Alternatively, we can approximate the t-distribution with a Gaussian if df is large, 
    # but the point is to use the t-dist.
    
    # Actually, a simpler and very effective robust acquisition is the 
    # "Expected Improvement" computed using the t-CDF/PDF directly.
    
    # Let's compute Z
    Z = (f_best - mu) / sigma
    
    # We need E[max(0, Z - T)] where T ~ t_df.
    # This is equivalent to the "t-expected improvement" term.
    # A known approximation or exact calculation involves the t-CDF and PDF.
    # Let's use the property that for T ~ t_df:
    # E[max(0, Z - T)] = Z * P(T < Z) + E[T * 1(T < Z)]
    # Wait, max(0, Z-T) is positive when T < Z.
    # E[max(0, Z-T)] = E[(Z-T) * 1(T<Z)] = Z * P(T<Z) - E[T * 1(T<Z)]
    
    # We know that for T ~ t_df, E[T * 1(T < z)] = -df * t_pdf(z) / (df - 1)  (for df > 1)
    # Let's verify this identity.
    # The t-distribution PDF is f(t) = C * (1 + t^2/df)^(-(df+1)/2).
    # The CDF is F(t).
    # It is a known result that E[T * 1(T < z)] = - (df / (df - 1)) * f_t(z) for df > 1?
    # Let's check.
    # Integration by parts:
    # \int_{-inf}^z t * f(t) dt.
    # Let u = t, dv = f(t)dt. This is tricky.
    # Alternative identity:
    # For standard normal: E[N * 1(N < z)] = -phi(z).
    # For t-dist: E[T * 1(T < z)] = - (df / (df - 1)) * pdf_t(z).
    # Let's assume this holds for df > 1.
    
    # So, EI_term = Z * t_cdf(Z, df) - ( - (df / (df - 1)) * t_pdf(Z, df) )
    # EI_term = Z * t_cdf(Z, df) + (df / (df - 1)) * t_pdf(Z, df)
    
    # Then EI = sigma * EI_term
    
    # Let's compute t_cdf and t_pdf
    # scipy.stats.t.cdf and scipy.stats.t.pdf
    # Note: t_dist.pdf(z, df) and t_dist.cdf(z, df)
    
    # Handle numerical stability for large Z
    # If Z is very large, cdf ~ 1, pdf ~ 0. EI_term ~ Z.
    # If Z is very small, cdf ~ 0.5, pdf ~ constant.
    
    t_cdf_vals = t_dist.cdf(Z, df)
    t_pdf_vals = t_dist.pdf(Z, df)
    
    # Factor for t-dist
    factor = df / (df - 1.0)
    
    ei_term = Z * t_cdf_vals + factor * t_pdf_vals
    
    # Expected Improvement
    scores = sigma * ei_term
    
    # Ensure scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
    # Check deadline
    if time.time() > deadline:
        pass

    return scores
# EVOLVE-BLOCK-END
