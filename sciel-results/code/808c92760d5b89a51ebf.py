"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t


def acquisition(mu, sigma, f_best):
    """
    Student's t-based Expected Improvement acquisition function.
    
    Uses a Student's t-distribution with 5 degrees of freedom to model
    the posterior predictive distribution, accounting for heavy-tailed
    uncertainty inherent in the limited 30-sample budget.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    dof = 5.0
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize: the t-distribution has mean 0 and variance dof/(dof-2)
    # We scale the GP posterior to match the t-distribution's unit variance
    # Actually, for EI with t-distribution, we use:
    # P(f < f_best | x) and E[max(f_best - f, 0)]
    # where f ~ f_best - (f_best - mu) * t_dof scaled appropriately
    
    # The t-distribution with ν dof has:
    # - Mean: 0 (for ν > 1)
    # - Variance: ν/(ν-2) (for ν > 2)
    # - Standard deviation: sqrt(ν/(ν-2))
    
    # We model the posterior as:
    # f(x) ~ mu + sigma * t_ν / sqrt(ν/(ν-2))
    # This way the posterior has mean mu and standard deviation sigma.
    
    # So: f(x) = mu + sigma * (t_ν / sqrt(ν/(ν-2)))
    # Let Z = t_ν / sqrt(ν/(ν-2)), which has mean 0 and variance 1.
    
    # Then: f(x) < f_best  <=>  mu + sigma * Z < f_best
    #       <=>  Z < (f_best - mu) / sigma
    #       <=>  Z < z where z = (f_best - mu) / sigma
    
    # EI = E[max(f_best - f, 0)]
    #    = E[max(f_best - mu - sigma*Z, 0)]
    #    = E[max(f_best - mu - sigma*Z, 0)]
    
    # Let a = (f_best - mu) / sigma
    # Then EI = sigma * E[max(a - Z, 0)]
    
    # For Z ~ t_ν / sqrt(ν/(ν-2)):
    # E[max(a - Z, 0)] = ∫_{-∞}^{a} (a - z) * pdf_Z(z) dz
    #                  = a * CDF_Z(a) - ∫_{-∞}^{a} z * pdf_Z(z) dz
    
    # Now, pdf_Z(z) = pdf_t(ν * z / sqrt(ν/(ν-2)) * sqrt(ν-2)/ν... let me redo this.
    
    # Let T ~ t_ν (standard Student's t with ν dof)
    # Z = T / sqrt(ν/(ν-2))  =>  T = Z * sqrt(ν/(ν-2))
    # pdf_Z(z) = pdf_T(z * sqrt(ν/(ν-2))) * sqrt(ν/(ν-2))
    
    # CDF_Z(a) = CDF_T(a * sqrt(ν/(ν-2)))
    
    # ∫_{-∞}^{a} z * pdf_Z(z) dz = ∫_{-∞}^{a} z * pdf_T(z * sqrt(ν/(ν-2))) * sqrt(ν/(ν-2)) dz
    # Let w = z * sqrt(ν/(ν-2)), so z = w / sqrt(ν/(ν-2)), dz = dw / sqrt(ν/(ν-2))
    # = ∫_{-∞}^{a*sqrt(ν/(ν-2))} (w / sqrt(ν/(ν-2))) * pdf_T(w) * sqrt(ν/(ν-2)) * (dw / sqrt(ν/(ν-2)))
    # = ∫_{-∞}^{a*sqrt(ν/(ν-2))} w * pdf_T(w) * (1 / sqrt(ν/(ν-2))) dw
    # = (1 / sqrt(ν/(ν-2))) * ∫_{-∞}^{a*sqrt(ν/(ν-2))} w * pdf_T(w) dw
    
    # The t-distribution is symmetric, so ∫_{-∞}^{b} w * pdf_T(w) dw is the partial first moment.
    # For a symmetric distribution, we can use:
    # ∫_{-∞}^{b} w * pdf_T(w) dw = -∫_{b}^{∞} w * pdf_T(w) dw  (by symmetry, since mean is 0)
    # Actually, let's just compute it numerically or use the identity.
    
    # For standard t_ν, the CDF is T_ν(cdf) and PDF is T_ν.pdf.
    # The partial moment ∫_{-∞}^{b} w * pdf_T(w) dw doesn't have a simple closed form,
    # but we can compute it as:
    # ∫_{-∞}^{b} w * pdf_T(w) dw = -ν * (1 + b²/(ν-1)) * pdf_T(b) / (ν-1) ... no, that's not right.
    
    # Let me just compute EI directly using the formula:
    # EI = sigma * [a * CDF_Z(a) - E[Z * 1(Z <= a)]]
    
    # Where E[Z * 1(Z <= a)] = ∫_{-∞}^{a} z * pdf_Z(z) dz
    
    # Since Z = T / s where s = sqrt(ν/(ν-2)):
    # pdf_Z(z) = s * pdf_T(s * z)
    # CDF_Z(a) = CDF_T(s * a)
    # E[Z * 1(Z <= a)] = ∫_{-∞}^{a} z * s * pdf_T(s*z) dz
    #   Let u = s*z, z = u/s, dz = du/s
    #   = ∫_{-∞}^{s*a} (u/s) * s * pdf_T(u) * (du/s)
    #   = (1/s) * ∫_{-∞}^{s*a} u * pdf_T(u) du
    
    # So EI = sigma * [a * CDF_T(s*a) - (1/s) * ∫_{-∞}^{s*a} u * pdf_T(u) du]
    
    # Now, for the integral I(b) = ∫_{-∞}^{b} u * pdf_T(u) du:
    # The pdf of t_ν is: pdf_T(u) = Γ((ν+1)/2) / (sqrt(νπ) * Γ(ν/2)) * (1 + u²/ν)^(-(ν+1)/2)
    # u * pdf_T(u) is an odd function, so I(b) = -∫_{b}^{∞} u * pdf_T(u) du (since total integral is 0)
    
    # There's no simple closed form, but we can compute it numerically.
    # Alternatively, we can use the fact that for large b, the integral converges.
    
    # For efficiency with 2000 points, let's precompute or use a vectorized approach.
    # We can use scipy.integrate.quad, but that might be slow for 2000 points.
    # 
    # Alternative: use the relationship with the incomplete beta function or just
    # compute the EI using a different formulation.
    
    # Actually, let's use a simpler approach: compute EI using the formula
    # EI = sigma * (a * T_ν_cdf(s*a) - (1/s) * partial_first_moment_T(s*a))
    
    # where partial_first_moment_T(b) = ∫_{-∞}^{b} u * pdf_T(u) du
    
    # We can compute this using scipy.integrate.quad for each point, but that's slow.
    # Let's use a numerical integration approach or approximate.
    
    # Actually, for ν=5, we can use the fact that:
    # ∫_{-∞}^{b} u * pdf_T(u) du = -pdf_T(b) * (ν + b²) / (ν - 1) ... let me verify.
    
    # The derivative of pdf_T(u) is:
    # d/du pdf_T(u) = pdf_T(u) * (-(ν+1) * u / (ν + u²))
    
    # So: d/du [pdf_T(u) * (ν + u²)/(ν-1)] ... this is getting complicated.
    
    # Let's just use numerical integration with scipy.integrate.quad for robustness.
    # With 2000 points, it should still be fast enough.
    
    from scipy.integrate import quad
    
    s = np.sqrt(dof / (dof - 2.0))
    
    # Vectorized computation
    a = (f_best - mu) / sigma_safe
    
    # We need to compute for each a_i:
    # EI_i = sigma_i * [a_i * CDF_T(s * a_i) - (1/s) * I(s * a_i)]
    # where I(b) = ∫_{-∞}^{b} u * pdf_T(u) du
    
    # Let's compute I(b) using quad for each point
    n = len(mu)
    ei_scores = np.zeros(n)
    
    for i in range(n):
        b = s * a[i]
        # Compute I(b) = ∫_{-∞}^{b} u * pdf_T(u) du
        # Use quad with a suitable lower limit
        # For large negative b, the integral is close to 0 (actually negative)
        # Let's integrate from -abs(b) - 10 to b, or use a fixed range
        
        # The t-distribution has heavy tails, but for practical purposes,
        # integrating from -50 to b should be sufficient for |b| < 50.
        lower = min(b - 10, -50)
        upper = max(b, 50)
        
        if b < -50:
            # The integral is very close to 0 (slightly negative)
            I_val = 0.0
        elif b > 50:
            # The integral is close to the total, which is 0 (since mean is 0)
            # Actually, ∫_{-∞}^{∞} u * pdf_T(u) du = 0
            # So for large b, I(b) ≈ 0
            I_val = 0.0
        else:
            integrand = lambda u: u * student_t.pdf(u, dof)
            I_val, _ = quad(integrand, lower, b, limit=100)
        
        cdf_val = student_t.cdf(b, dof)
        
        ei_scores[i] = sigma_safe[i] * (a[i] * cdf_val - I_val / s)
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
