<<<<<<< SEARCH
def _ei_fast(mu, sigma, f_best):
    """Fast Gaussian Expected Improvement for minimization using erfc."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    # norm.cdf(z) = 0.5 * erfc(-z / sqrt(2))
    cdf_z = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = d * cdf_z + sigma_safe * pdf_z
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    return ei
=======
def _ei_fast(mu, sigma, f_best):
    """Fast Gaussian Expected Improvement for minimization using erfc."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    cdf_z = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = d * cdf_z + sigma_safe * pdf_z
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    return ei


def _knowledge_gradient(mu, sigma, f_best, cost, n_candidates=2000):
    """
    Approximate Knowledge Gradient: expected reduction in f_best.
    KG(x) = E[max(f_best - f(x), 0)] - E[max(f_best - f'(x'), 0)]
    where f'(x') is the best of a set of future evaluations.
    
    We approximate this using the current GP: the probability that x is the new best
    times the expected improvement, plus a term for reducing uncertainty in the
    region around the current best.
    
    Simplified KG ≈ EI * P(x is promising) + exploration_bonus
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    cdf_z = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = d * cdf_z + sigma_safe * pdf_z
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    
    # KG approximation: weight EI by the probability that this point beats f_best
    # and by how much it reduces expected best value
    p_beat = cdf_z  # P(f(x) < f_best)
    
    # Expected reduction in f_best if we evaluate x:
    # E[f_new_best | x] ≈ f_best - EI(x) * p_beat(x) / (p_beat(x) + small)
    # Simplified: KG ≈ EI * (0.5 + 0.5 * p_beat) + sigma * p_beat * (1 - p_beat)
    # The second term rewards points that are uncertain AND likely to be good
    
    kg = ei * (0.5 + 0.5 * p_beat) + 0.3 * sigma_safe * p_beat * (1.0 - p_beat)
    kg = np.where(sigma < eps, np.maximum(d, 0.0), kg)
    return kg
>>>>>>> REPLACE

