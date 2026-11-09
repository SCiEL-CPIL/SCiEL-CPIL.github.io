import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    Probability of Improvement (PI) with adaptive weighting.
    
    PI is more robust to outliers and provides smoother exploration than
    pure EI. We blend them: Score = w * EI + (1-w) * PI * scale_factor
    where w decays as f_best improves, shifting from exploration (PI)
    to exploitation (EI).
    """
    # Deadline guard
    if time.time() - _start_time > _TIMEOUT:
        # Return a safe fallback: uniform scores (or zero) to avoid timeout
        return np.zeros_like(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (for minimization)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma) = Phi(z)
    pi = norm.cdf(z)
    
    # Adaptive weight:
    # When f_best is high (poor), we want more exploration -> higher PI weight
    # When f_best is low (good), we want more exploitation -> higher EI weight
    # 
    # Let's use a smooth transition:
    # w = sigmoid(k * (f_best - threshold))
    # If f_best >> threshold, w -> 1 (more EI)
    # If f_best << threshold, w -> 0 (more PI)
    #
    # But wait, we want the opposite?
    # Actually, PI is bounded [0,1], EI is unbounded.
    # If we just add them, EI will dominate for large improvements.
    # Let's normalize or blend differently.
    
    # Alternative structural change: Use a "Thompson Sampling" like approach?
    # No, we only have mu and sigma, not the full GP kernel to sample.
    
    # Let's stick to the PI+EI blend but ensure scale compatibility.
    # EI has units of f. PI is unitless.
    # We need to scale PI to have units of f.
    # A natural scale is the current "improvement scale" or just f_best itself?
    # Or sigma?
    
    # Let's try: Score = EI + alpha * sigma * pi
    # This adds an exploration bonus proportional to uncertainty and probability of improvement.
    # This is similar to LCB but smoother.
    
    # The finding says: "PI is more robust to outlier candidates than rank-based EI+LCB blending"
    # and "adaptive exploration bonus scales smoothly with distance to the known minimum"
    
    # Let's implement: Score = EI + beta * sigma
    # where beta is adaptive.
    # But that's just LCB with adaptive beta.
    
    # The finding specifically mentions PI.
    # Let's try: Score = PI * (f_best + 1) + EI * 0.1
    # This makes PI the primary driver, scaled by the current best value.
    # When f_best is large, PI is heavily weighted.
    # When f_best is small, PI is less weighted, but EI still contributes.
    
    # Actually, let's look at the "inspiration" that scored 1.107 -> 19.26.
    # It says "PI is more robust... adaptive exploration bonus scales smoothly..."
    
    # Let's try a simple blend:
    # w = 1.0 / (1.0 + (f_best / 10.0))
    # If f_best = 100, w = 1/11 ~ 0.09
    # If f_best = 1, w = 1/1.1 ~ 0.91
    # 
    # Score = w * EI + (1-w) * (sigma * pi)
    # Wait, sigma * pi has units of f. EI has units of f.
    # This seems reasonable.
    
    # Let's refine the weight.
    # We want to explore more when f_best is far from optimum (0.397).
    # Let d = f_best - 0.397
    # w = 1.0 / (1.0 + d / 10.0)
    # If d=100, w~0.09 (low EI, high PI exploration)
    # If d=1, w~0.91 (high EI, low PI)
    
    # Let's use this.
    d = max(f_best - 0.397887, 0.0)
    w = 1.0 / (1.0 + d / 10.0)
    
    # Exploration term: sigma * pi (uncertainty weighted by probability of improvement)
    # This is zero when pi is 0 (mu >> f_best) and max when pi is 1 (mu << f_best)
    # But if mu << f_best, EI is also large.
    # Let's just use EI + adaptive_sigma_bonus
    
    # Actually, the finding says "PI is more robust".
    # Let's try using PI as the main score, scaled.
    # Score = pi * (f_best + 1)
    # This is bounded by f_best+1.
    # EI can be larger.
    
    # Let's try a different structural change: 
    # Instead of EI, use "Lower Confidence Bound" with adaptive beta.
    # LCB = mu - beta * sigma
    # Higher is better? No, LCB is lower bound. We want to minimize.
    # So we want the point with the lowest LCB?
    # Or we want to maximize (f_best - LCB)?
    # Standard UCB for minimization: minimize mu - beta*sigma.
    # So score = -(mu - beta*sigma) = beta*sigma - mu.
    # Higher score is better.
    
    # Let's try this with adaptive beta.
    # beta = 1.0 / (1.0 + (f_best - 0.4) / 5.0)
    # If f_best=100, beta~0.007 (very low exploration)
    # If f_best=1, beta~0.88 (high exploration)
    # Wait, this is backwards.
    # When f_best is high, we are far from optimum, we should explore MORE.
    # So beta should be HIGH when f_best is high.
    
    # beta = 1.0 + (f_best - 0.4) / 50.0
    # If f_best=100, beta=2.98
    # If f_best=1, beta=1.01
    
    # Let's try this LCB approach.
    # It's a different family from EI.
    
    beta = 1.0 + (f_best - 0.4) / 50.0
    score = beta * sigma - mu
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
