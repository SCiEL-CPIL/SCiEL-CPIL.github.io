import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # mu and sigma have shape (2000,)
    # We want to compute the probability that each candidate is the minimum among the candidates.
    # Assumption: Candidates are independent (approximation).
    # P(f_i < f_j) = Phi((mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2))
    
    n = len(mu)
    # Create matrices for mu and sigma
    # mu_i vs mu_j
    # diff = mu_j - mu_i
    # denom = sqrt(sigma_i^2 + sigma_j^2)
    
    # To avoid O(N^2) memory if N is large, but 2000 is small.
    # 2000x2000 is 4 million floats, ~32MB. Acceptable.
    
    mu_col = mu[:, np.newaxis]
    mu_row = mu[np.newaxis, :]
    
    sigma_col = sigma[:, np.newaxis]
    sigma_row = sigma[np.newaxis, :]
    
    diff = mu_row - mu_col  # Shape (N, N), diff[i, j] = mu_j - mu_i
    denom = np.sqrt(sigma_col**2 + sigma_row**2) # Shape (N, N)
    
    # Avoid division by zero
    denom = np.where(denom < 1e-9, 1e-9, denom)
    
    z = diff / denom
    p = norm.cdf(z)
    
    # For i=j, p=0.5. We should exclude self-comparison or set it to 1 (log 0).
    # We want product over j != i.
    # Set diagonal to 1 so log is 0.
    np.fill_diagonal(p, 1.0)
    
    log_pbb = np.sum(np.log(p), axis=1)
    
    # Return log_pbb. Higher is better.
    # Ensure finite
    log_pbb = np.where(np.isfinite(log_pbb), log_pbb, -np.inf)
    
    return log_pbb
