import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 5.0  # 5 second safety margin within 300s budget
    
    # Stochastic Expected Improvement (SEI) with log-normal cost model
    # We model cost as log-normal: C ~ LogNormal(0, 1) scaled by observed cost
    # This integrates cost uncertainty into the acquisition integral
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Standardize the improvement: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI for minimization: E[max(f_best - f, 0)] = sigma * (z * Phi(z) + phi(z))
    # where Phi is CDF and phi is PDF of standard normal
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    
    # Cost-aware EI: E[EI / C] where C ~ LogNormal(0, 1) * cost
    # E[1/C] for C ~ LogNormal(mu_c, sigma_c) = exp(-mu_c + sigma_c^2/2)
    # Here C = cost * exp(N(0,1)), so mu_c = log(cost), sigma_c = 1
    # E[1/C] = exp(-log(cost) + 1/2) = exp(0.5) / cost
    # So E[EI/C] = EI * E[1/C] = EI * exp(0.5) / cost
    
    # This is equivalent to EI / cost up to a constant factor
    # But we can add a small exploration bonus based on sigma/cost ratio
    # to balance exploitation and exploration
    
    # Use a weighted combination: primarily EI/cost, with a small UCB-like term
    # to encourage exploration in uncertain regions
    beta_explore = 0.5
    ucb_term = beta_explore * sigma / cost
    
    scores = ei / cost + ucb_term
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
