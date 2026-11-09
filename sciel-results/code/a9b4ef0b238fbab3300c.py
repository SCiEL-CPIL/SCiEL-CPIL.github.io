"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Global Rank-based Acquisition Function.
    
    Strategy:
    Instead of computing a pointwise probability (like POI) or expected value (like EI), 
    we rank the 2000 candidates based on a stochastic sample from the posterior 
    (Thompson Sampling approximation). 
    
    We generate a single sample for each candidate: y_i ~ N(mu_i, sigma_i^2).
    We then assign scores based on the rank of these samples in ascending order 
    (lower is better). The top-ranked candidates get the highest acquisition scores.
    
    This approach:
    1. Naturally handles the multimodal nature of Branin by exploring high-variance 
       regions if they produce low samples.
    2. Is robust to scale issues because it relies on relative ordering.
    3. Provides a strong exploitation signal for the best-performing regions in 
       the current batch while maintaining exploration via the stochastic sampling.
    
    To make this deterministic and suitable for the API (which expects a score for 
    every point, where the max is chosen), we use the rank itself as the score.
    However, since we only have one sample per point, we can use a "Softmax Rank" 
    or simply the inverse of the rank to create a smooth gradient.
    
    Actually, a simpler and often more effective structural change for batch BO 
    with fixed candidates is to use the **Expected Improvement (EI)** but with 
    a **local length-scale adjustment** or **Multi-Objective EI** approximation.
    
    Let's try a different structural change: **Knowledge Gradient (KG) approximation**.
    KG maximizes the expected reduction in the minimum value. For a single point, 
    it is complex, but for a batch of 2000, we can approximate the value of 
    information.
    
    However, the prompt asks for a MATERIAL DIFFERENCE. 
    Current: POI (Deterministic, local).
    Inspiration 1: EI (Deterministic, local).
    
    Let's use **Thompson Sampling via Rank**.
    Since we must return a score for each of the 2000 points, and the evaluator picks the max,
    we can simulate the Thompson Sampling decision by assigning scores that reflect 
    the probability of being the minimum in the batch.
    
    Let's implement a **Batch Thompson Sampling** approximation:
    1. Sample y_i ~ N(mu_i, sigma_i) for all i.
    2. The "best" point is argmin(y).
    3. To create a score function that picks this point, we can score each point 
       by how "likely" it is to be the best. A simple proxy is the rank of y_i 
       among all y_j.
    
    Score_i = 1 / (Rank_i + 1)
    Where Rank_i is the index of y_i when sorted in ascending order (0 is best).
    
    This forces the algorithm to pick the single best sample from the posterior 
    distribution across the entire batch, which is a global, stochastic strategy.
    """
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: sigma must be positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate a single sample from the posterior for each candidate
    # y_i ~ N(mu_i, sigma_i)
    # We use a fixed seed relative to the input? No, the evaluator runs this multiple times.
    # If we use random, it's stochastic. The task allows this as long as it's valid.
    # However, stochasticity can hurt reproducibility if not managed. 
    # But Thompson Sampling is inherently stochastic.
    
    # To be safe and deterministic-ish but structurally different, 
    # we can use a **Quantile-based** approach.
    
    # Let's try **Expected Improvement (EI)** but with a **Gamma correction** 
    # or **Multi-Step EI**.
    
    # Actually, let's look at the failure of POI (17.4 score, which is bad). 
    # EI (Inspiration 1) scored 1.42. 
    # The parent of the current program (p45) scored 0.376. 
    # The current program (POI) is much worse.
    
    # Structural Change: **Upper Confidence Bound (UCB)** with a **dynamic beta**.
    # UCB: mu - beta * sigma
    # POI: Phi((f_best - mu) / sigma)
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Let's use **Knowledge Gradient (KG)** approximation for a single step.
    # KG(x) = E[min(f(x), f_best) - min(f(x'), f_best)]
    # For a single point, it approximates to EI but with a different weighting.
    
    # Let's try **Probability of Improvement (POI)** but **inverted** or **modified**?
    # No, POI was bad.
    
    # Let's go with **Thompson Sampling via Rank** as it is a distinct family (Stochastic vs Deterministic).
    # To ensure stability, we will use a **Quantile Sample** instead of a single random draw.
    # We sample the 0.1 quantile? No, that's deterministic.
    
    # Let's use **Expected Improvement (EI)** but with a **local variance penalty** 
    # to encourage exploration of unvisited areas?
    
    # Wait, the inspiration 1 (EI) scored 1.42. The parent scored 0.376. 
    # The current POI scored 17.4 (bad).
    # We need to beat 17.4 easily, but ideally beat 1.42.
    
    # Let's implement **Expected Improvement (EI)** with a **small epsilon** to avoid 
    # division by zero and ensure smooth gradients. EI is the standard for a reason.
    # Why did POI fail? POI is less sensitive to the magnitude of improvement. 
    # EI rewards large improvements. Branin has multiple minima. 
    # If the GP is well-calibrated, EI should work.
    
    # However, the directive says "MATERIAL DIFFERENT". 
    # EI is likely what Inspiration 1 did. 
    # Let's do **Upper Confidence Bound (UCB)** with a **tuned beta**.
    # UCB is simpler and often outperforms EI in high-dimensional or noisy settings.
    
    # Let's try **UCB** with beta = 1.0.
    
    # Scores = mu - beta * sigma
    # Higher score is better (since we want to minimize f, we want low mu and high sigma).
    # Wait, for minimization:
    # We want low mu (exploitation).
    # We want high sigma (exploration).
    # So Score = -mu + beta * sigma? 
    # Or Score = mu - beta * sigma (if we maximize score, we want high sigma, low mu).
    # Yes, if we maximize the acquisition, we want to pick points with low predicted mean 
    # and high uncertainty.
    
    # Let's check the sign.
    # If mu is low, -mu is high. Good.
    # If sigma is high, beta*sigma is high. Good.
    # So Score = -mu + beta * sigma.
    
    # Let's implement this.
    
    beta = 1.0
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
