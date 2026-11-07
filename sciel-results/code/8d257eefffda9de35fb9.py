# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Probability of Improvement (CA-POI).
    
    Strategy:
    1. Use Probability of Improvement (POI) as the base signal. POI is robust 
       to multi-modal landscapes because it only cares about the probability 
       of finding a point better than f_best, not the magnitude of the potential 
       improvement (which can be misleading in high-noise or high-variance regions).
    2. Cost-awareness: Divide by cost to strictly prefer cheaper evaluations. 
       This is more aggressive than linear division in some contexts, but POI 
       naturally decays as the probability of improvement drops, preventing 
       the selection of expensive points with negligible upside.
    3. To ensure we don't get stuck in local exploitation, we add a small 
       exploration bonus based on uncertainty (sigma) but weighted by the 
       probability that this uncertainty actually leads to improvement. 
       This is effectively a "Confidence-Weighted POI".
       
    Formula:
    P(x < f_best) = Phi((f_best - mu) / sigma)
    Expected Improvement (for context, though we use POI primarily) = (f_best - mu)Phi(z) + sigma phi(z)
    
    We use a hybrid: Score = (POI * (f_best - mu) + sigma * POI) / cost
    Actually, simpler and often better for multi-modal:
    Score = POI / cost + small_exploration_term
    
    Let's stick to a robust CA-POI:
    Score = P(x < f_best) / cost
    
    However, pure POI can be too greedy. Let's add a small UCB-like term 
    only for points where POI is significant, to break ties in high-uncertainty 
    regions that are cheap.
    
    Revised Score:
    POI = Phi((f_best - mu) / sigma)
    Base_Score = POI
    Exploration_Bonus = 0.1 * sigma * POI  # Uncertainty weighted by chance of success
    Total_Score = (Base_Score + Exploration_Bonus) / cost
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-score: how many std devs away is the mean from the best?
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive. We want high z (high probability of being better).
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(x < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # To avoid numerical issues where poi is extremely close to 1 or 0,
    # we clip it slightly, though Phi is stable.
    poi = np.clip(poi, 1e-6, 1.0 - 1e-6)
    
    # Exploration term: 
    # We want to explore regions with high sigma, BUT only if there is a reasonable 
    # chance they contain the global minimum. 
    # If POI is high, we are confident this region is good. 
    # If POI is low, we shouldn't explore there even if sigma is high (it's likely bad).
    # So, weight sigma by POI.
    # This creates a "smart" exploration term that vanishes in bad regions.
    exploration_weight = 0.2
    exploration_term = exploration_weight * sigma * poi
    
    # Base term: The probability itself.
    # This drives exploitation towards known good areas.
    
    raw_score = poi + exploration_term
    
    # Cost adjustment:
    # Divide by cost to prefer cheaper points.
    # cost is in [1, 2].
    # Dividing by cost effectively scales the score down for expensive points.
    scores = raw_score / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
