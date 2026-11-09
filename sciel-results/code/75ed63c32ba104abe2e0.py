"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard variables
# The function is called many times, but the constraint is on the *evaluation* step.
# The prompt says "Each evaluation must finish within 300 s".
# The acquisition function itself is called once per step.
# We should ensure it's fast. The 330s kill switch is for the whole program.
# We don't need a global deadline inside a single call unless it's very slow, 
# but we can add a safety check if it takes too long.

def _adaptive_ei(mu, sigma, f_best):
    """
    Calculate Adaptive Expected Improvement.
    
    Standard EI for minimization:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    However, in BO with standardized targets, the scale of mu and sigma changes 
    relative to f_best as the GP updates. A static EI can become overly exploitative 
    or exploratory depending on the current spread of the posterior.
    
    This strategy uses a "temperature" or scaling factor that adapts based on 
    the current uncertainty and progress. 
    Since we don't know the exact number of steps, we use the current sigma 
    distribution to normalize the exploration.
    
    A more robust approach for Hartmann 6d (multimodal, deceptive) is to use 
    a slightly more exploratory variant early on, but since we don't have 
    step count, we use the magnitude of sigma to modulate.
    
    Actually, a simpler and often more effective strategy for this specific 
    contract (fixed 30 evals, 2000 candidates) is to use the standard EI 
    but ensure numerical stability and perhaps a slight boost to exploration 
    when sigma is high relative to the current best improvement.
    
    Let's stick to the mathematically sound EI but with a twist: 
    We normalize the scores to [0, 1] or ensure they are comparable? 
    No, the task says "Return 2000 finite scores, higher meaning evaluate next". 
    The evaluator picks the argmax. So relative scaling within the candidate set 
    matters, but absolute scaling doesn't strictly matter for argmax, 
    UNLESS we mix different acquisition functions (which we won't).
    
    Wait, the previous program used a portfolio. The inspiration 1 (score 1.0) 
    is likely the seed or a very simple EI. The current program (score 0.9047) 
    uses a portfolio. The portfolio might be causing issues if the scales 
    of EI, UCB, and PI are very different, leading to the "max" operation 
    picking a candidate that is high in UCB (exploration) but poor in EI 
    (exploitation) simply because its UCB score is numerically larger.
    
    A materially different strategy: **Single, well-tuned EI with numerical safeguards.**
    The Hartmann 6d function is well-behaved enough for EI to perform well.
    The main failure mode of EI is getting stuck in local optima if the GP 
    hasn't explored enough. With 5 initial points and 25 steps, EI should be 
    sufficient if the GP is accurate.
    
    Let's implement a clean, fast EI.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Handle numerical issues
    ei[~np.isfinite(ei)] = 0.0
    
    # Ensure non-negative EI (should be by definition, but floating point errors can occur)
    ei = np.maximum(ei, 0.0)
    
    return ei

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Calculate Adaptive EI
    scores = _adaptive_ei(mu, sigma, f_best)
    
    # Final safety check: ensure all scores are finite
    scores[~np.isfinite(scores)] = 0.0
    
    return scores

# EVOLVE-BLOCK-END
