import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 280.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge case where f_best is very small
    # If f_best is 0, we are done, but let's be safe.
    if f_best <= 1e-9:
        # Pure exploitation
        return -mu
    
    # Ensure sigma is non-negative and safe for normalization
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Strategy: Quantile-based Probability of Improvement (POI)
    # We compute the CDF of the posterior distribution at the current best value,
    # P(f < f_best). This ranks candidates by their likelihood of beating the
    # incumbent rather than the magnitude of improvement or uncertainty.
    #
    # The posterior is assumed to be Gaussian: f ~ N(mu, sigma^2)
    # P(f < f_best) = Phi((f_best - mu) / sigma)
    #
    # This decouples selection pressure from the absolute scale of the objective.
    # However, pure POI can be too exploitative in early stages when f_best is far
    # from the optimum. To balance this, we blend POI with a quantile-normalized
    # exploration term (similar to the inspirations) that encourages visiting
    # high-uncertainty regions early on.
    
    # Compute POI scores
    # z = (f_best - mu) / sigma
    # POI = Phi(z)
    # We want higher scores to be better. Phi(z) is in [0, 1].
    # A high POI means high probability of improvement.
    z = (f_best - mu) / sigma_safe
    poi_scores = norm.cdf(z)
    
    # Quantile normalization of sigma for exploration bonus
    # This maps sigma values to [0, 1] based on their rank.
    n = len(sigma_safe)
    if n > 1:
        ranks = np.argsort(np.argsort(sigma_safe))
        sigma_norm = ranks / (n - 1)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Dynamic exploration weight
    # We want to explore more when f_best is large (early) and exploit when f_best is small (late).
    # The inspirations used beta = 1.0 * f_best.
    # However, since POI is already bounded in [0, 1], adding a large exploration term
    # might dominate the POI signal.
    # Let's use a smaller, fixed weight for the exploration term to act as a tie-breaker
    # or a gentle push towards uncertain regions, while POI drives the main selection.
    # Alternatively, we can blend them.
    # Let's try: score = poi_scores + alpha * sigma_norm
    # If alpha is small, POI dominates. If alpha is larger, exploration matters more.
    # Given the stagnation warning and the desire for a structural change, let's rely
    # primarily on POI but keep a small exploration component to avoid getting stuck.
    
    # Let's experiment with a blend.
    # The inspirations scored ~2.19 with UCB + quantile sigma.
    # The current program scores ~1.54 with raw UCB.
    # The directive suggests POI.
    
    # Let's use a weighted sum.
    # POI is in [0, 1]. sigma_norm is in [0, 1].
    # We want the score to be high for good candidates.
    # Let's set alpha such that exploration is secondary but present.
    # A value like 0.1 or 0.2 might be appropriate.
    
    # However, pure POI might be too greedy. Let's look at the structure.
    # The directive says: "Implement a quantile-based Probability of Improvement (POI) acquisition function by computing the CDF of the posterior distribution at the current best value, P(f < f_best), which ranks candidates by their likelihood of beating the incumbent rather than the magnitude of improvement or uncertainty, thereby decoupling the selection pressure from the absolute scale of the objective function."
    
    # This suggests POI is the primary signal.
    # Let's start with pure POI, but add a small exploration bonus to handle ties or early diversity.
    # Actually, the directive says "quantile-based POI". This might imply using the quantile of the POI scores?
    # Or does it mean POI computed on quantile-normalized sigma? No, POI uses mu and sigma directly.
    # "Quantile-based" might refer to the fact that we are using the CDF (which is a quantile function inverse) or that we rank by POI.
    # Let's stick to standard POI: Phi((f_best - mu) / sigma).
    
    # To ensure we don't get stuck in local minima, we can add a small exploration term.
    # Let's use a small alpha.
    alpha = 0.1
    scores = poi_scores + alpha * sigma_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
