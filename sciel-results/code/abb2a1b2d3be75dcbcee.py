import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 320:
        # Fallback to EI if time is tight
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    n = len(mu)
    
    # Hypervolume-based acquisition:
    # The idea is to maximize the expected increase in hypervolume of the Pareto front.
    # For a single-objective minimization problem, the "Pareto front" is just the best point.
    # The hypervolume indicator for a single point f is the area/volume dominated by f
    # relative to a reference point.
    #
    # A more practical approach for single-objective: use a modified acquisition that
    # balances exploration and exploitation by considering the expected improvement
    # weighted by the uncertainty in a way that captures the hypervolume concept.
    #
    # Hypervolume-based acquisition for single-objective:
    # The expected hypervolume improvement (EHVI) for a single point can be approximated
    # by considering the probability that the point improves upon f_best and the magnitude
    # of that improvement, but weighted by the "volume" it would add to the Pareto front.
    #
    # For single-objective minimization, the EHVI simplifies to something close to EI,
    # but we can add a hypervolume-inspired weighting that favors points in unexplored regions.
    #
    # A practical approach: Use a combination of EI and a bonus for high uncertainty
    # in regions where the current best is not well-established.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Hypervolume-inspired term:
    # The hypervolume contribution of a point is related to how much it dominates
    # the current front. For a single best point f_best, a new point x contributes
    # to the hypervolume if f(x) < f_best. The "volume" added is proportional to
    # (f_best - f(x)) in a normalized sense.
    #
    # We can model this by considering the expected improvement but with a
    # hypervolume-weighted probability. The key insight is that in multimodal
    # landscapes, we want to explore regions with high uncertainty even if the
    # current mean is not promising, because they might contain better minima.
    #
    # Hypervolume-based weighting: weight the EI by a factor that increases
    # with the uncertainty relative to the range of observed values.
    #
    # Let's use a normalized uncertainty bonus:
    mu_range = np.ptp(mu)
    if mu_range < 1e-12:
        mu_range = 1e-12
    
    # Normalize sigma to [0, 1] relative to the range of mu
    sigma_norm = sigma_safe / (mu_range + 1e-12)
    
    # Hypervolume-inspired acquisition:
    # Combine EI with a term that rewards high uncertainty in regions
    # where the mean is not too far from f_best (i.e., potentially competitive)
    #
    # The hypervolume of a point relative to the reference point f_ref = f_best
    # is (f_ref - f) if f < f_ref, else 0.
    # The expected hypervolume contribution is E[max(0, f_best - f(x))]
    # which is exactly EI! So for single-objective, EHVI = EI.
    #
    # However, the key difference is in how we handle the reference point and
    # the weighting. Let's try a different approach:
    #
    # Use a UCB-like term with a hypervolume-inspired beta that adapts
    # based on the number of evaluations.
    #
    # Actually, let's try a simpler but effective approach:
    # Use a mixture of EI and a pure exploration term (max sigma) with
    # a hypervolume-inspired weighting.
    
    # Exploration term: pure uncertainty
    exploration = sigma_safe
    
    # Weighting: in early stages, explore more; in later stages, exploit more
    # The call count gives us a proxy for the number of evaluations
    t = _call_count
    # Decay exploration weight over time
    beta = 1.0 / (1.0 + 0.1 * t)
    
    # Hypervolume-inspired: weight exploration by the potential to improve
    # the hypervolume, which is higher when the mean is close to f_best
    # (i.e., the point is competitive)
    competitiveness = np.exp(-np.abs(f_best - mu) / (mu_range + 1e-12))
    
    # Combine: EI + beta * exploration * competitiveness
    hv_acq = ei + beta * exploration * competitiveness
    
    # Ensure finite scores
    hv_acq = np.where(np.isfinite(hv_acq), hv_acq, 0.0)
    
    return hv_acq
