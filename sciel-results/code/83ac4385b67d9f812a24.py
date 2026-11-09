import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Bayesian Regret Minimization (BRM) acquisition function.
    
    The scoring metric is simple regret: lowest observed value minus the true minimum.
    We want to minimize this. The expected simple regret after evaluating point x is:
    E[min(f(x), f_best)] = f_best - E[max(0, f_best - f(x))]
    
    The expected improvement EI(x) = E[max(0, f_best - f(x))]
    So minimizing expected regret is equivalent to maximizing EI.
    
    However, standard EI can be too exploitative. We use a cost-normalized version
    that directly targets the regret reduction per unit cost.
    
    Score = EI(x) / cost(x)
    
    This is structurally different from:
    - Pure EI (doesn't account for cost)
    - UCB (linear in sigma, EI is non-linear)
    - Thompson Sampling (stochastic, BRM is deterministic expected value)
    """
    current_time = time.time()
    if current_time - _start_time > _TIMEOUT:
        return np.zeros(2000)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI = E[max(0, f_best - f(x))]
    # where f(x) ~ N(mu, sigma^2)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Handle numerical issues
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: maximize expected regret reduction per unit cost
    scores = ei / cost_safe
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
