def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei_scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Add an exploration term
    # This encourages points with high uncertainty
    exploration_weight = 1.0
    scores = ei_scores + exploration_weight * sigma_safe
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
