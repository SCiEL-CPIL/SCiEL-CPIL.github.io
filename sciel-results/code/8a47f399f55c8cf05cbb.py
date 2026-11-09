import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
       sigma = np.maximum(sigma, 1e-9)
       
       # Candidate kappas
       kappas = [1.0, 2.0, 3.0]
       
       best_ei_at_peak = -np.inf
       best_scores = None
       
       for kappa in kappas:
           # UCB scores for minimization: -mu + kappa * sigma
           scores = -mu + kappa * sigma
           
           # Find the best candidate
           idx = np.argmax(scores)
           
           # Compute EI at this candidate
           mu_star = mu[idx]
           sigma_star = sigma[idx]
           z = (f_best - mu_star) / sigma_star
           ei_star = (f_best - mu_star) * norm.cdf(z) + sigma_star * norm.pdf(z)
           
           if ei_star > best_ei_at_peak:
               best_ei_at_peak = ei_star
               best_scores = scores
               
       # If the best EI is very low/negative, maybe fall back to EI?
       # Or just return the best UCB scores found.
       
       # Ensure finite
       best_scores = np.where(np.isfinite(best_scores), best_scores, -np.inf)
       
       return best_scores
