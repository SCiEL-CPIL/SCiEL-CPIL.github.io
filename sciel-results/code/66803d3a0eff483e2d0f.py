import numpy as np
import time
from scipy.special import erf

# Constants
SQRT_2PI = np.sqrt(2 * np.pi)
INV_SQRT_2 = 1 / np.sqrt(2)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG) approximation.
    
    KG estimates the expected reduction in the observed minimum after evaluating
    a candidate. We use a competitive approximation: for each candidate, compute
    EI but then apply a softmax-like competition among top candidates to favor
    points that are likely to be the new global best, not just locally improving.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    t_start = time.time()
    deadline = t_start + 0.050
    
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Compute EI components
    z = (f_best - mu) / sigma_safe
    z_erf_arg = z * INV_SQRT_2
    phi_z = 0.5 * (1.0 + erf(z_erf_arg))
    z_sq_half = 0.5 * z * z
    pdf_z = np.exp(-z_sq_half) / SQRT_2PI
    ei = (f_best - mu) * phi_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # KG competitive approximation:
    # The key idea is that the value of evaluating a point depends on how likely
    # it is to be the NEW best among all candidates. We approximate this by
    # considering the "rank" of each candidate's predicted minimum.
    
    # Predicted minimum at each point: mu - k*sigma (optimistic prediction)
    # Use a temperature-scaled softmax to create competition weights
    # Candidates with lower predicted minima get higher weights
    
    # Temperature adapts: early on, more exploration (higher temp); later, exploitation
    # We don't have step count, so use spread of mu to adapt
    mu_range = np.max(mu) - np.min(mu) + 1e-10
    
    # Optimistic scores: lower is better
    optimistic = mu - 1.5 * sigma_safe
    
    # Convert to competition weights via softmax (higher weight for better candidates)
    # Use negative optimistic so that lower optimistic -> higher weight
    temp = 0.1 * mu_range + 1e-6
    neg_opt = -optimistic / temp
    # Shift for numerical stability
    neg_opt_max = np.max(neg_opt)
    weights = np.exp(neg_opt - neg_opt_max)
    weights_sum = np.sum(weights)
    
    # KG score: EI weighted by competition weight, plus a small exploration bonus
    # This favors points that are both promising (high EI) and likely to be global best
    kg_scores = ei * weights / weights_sum * len(mu)
    
    # Add a small uncertainty bonus to maintain some exploration
    # This prevents the acquisition from being too greedy
    uncertainty_bonus = 0.1 * sigma_safe / (np.max(sigma_safe) + 1e-10)
    kg_scores = kg_scores + uncertainty_bonus * np.max(ei)
    
    # Ensure finite
    scores = np.nan_to_num(kg_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
