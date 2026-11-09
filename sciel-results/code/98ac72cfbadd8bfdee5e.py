"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize_scalar

_START_TIME = time.time()
_DEADLINE_MARGIN = 310.0


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -8.0, 8.0)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    return np.maximum(ei, 0.0)


def _compute_ucb(mu, sigma, beta):
    """Compute Lower Confidence Bound for minimization (higher score = more promising)."""
    return -mu + beta * sigma


def _compute_poi(mu, sigma, f_best, gamma=0.01):
    """Probability of Improvement: P(f(x) < f_best - gamma)."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - gamma - mu) / sigma_safe
    z_clipped = np.clip(z, -8.0, 8.0)
    poi = norm.cdf(z_clipped)
    return np.clip(poi, 1e-15, 1.0 - 1e-15)


def _compute_entropy_sigma(sigma):
    """Differential entropy of N(0, sigma^2), higher = more uncertain."""
    sigma_safe = np.maximum(sigma, 1e-9)
    return 0.5 * (1.0 + np.log(2.0 * np.pi * sigma_safe ** 2))


def _compute_kl_divergence(mu, sigma, f_best):
    """
    Approximate KL divergence between posterior N(mu, sigma^2) and a prior
    that is centered at f_best with similar spread.
    KL(N(mu, sigma^2) || N(f_best, sigma^2)) = 0.5 * ((mu - f_best)/sigma)^2
    This measures how "surprising" the posterior is relative to current best.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (mu - f_best) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    return 0.5 * z_clipped ** 2


def _compute_nash_equilibrium_score(mu, sigma, f_best):
    """
    Nash Equilibrium (NE) acquisition: combines exploitation and exploration
    in a way that's robust to the balance between the two.
    NE(x) = EI(x) * log(1 + sigma(x))
    The log factor gently weights exploration by uncertainty scale.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    ei = _compute_ei(mu, sigma, f_best)
    ne = ei * np.log1p(sigma_safe)
    return ne


def _compute_multi_objective_score(mu, sigma, f_best, n_evals):
    """
    Multi-objective score that adaptively weights different acquisition criteria
    based on the stage of optimization and observed properties.
    
    As optimization progresses, shift weight from exploration to exploitation.
    """
    n = len(mu)
    
    # EI: exploitation-focused
    ei = _compute_ei(mu, sigma, f_best)
    
    # UCB with adaptive beta: exploration-focused
    # Beta decreases as we have more evaluations
    beta = 2.0 * np.exp(-n_evals / 10.0)
    ucb = _compute_ucb(mu, sigma, beta)
    
    # POI: balanced
    poi = _compute_poi(mu, sigma, f_best, gamma=0.005)
    
    # Normalize each score to [0, 1] range for fair combination
    def normalize(s):
        s_min, s_max = np.min(s), np.max(s)
        if s_max - s_min < 1e-15:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ei_n = normalize(ei)
    ucb_n = normalize(ucb)
    poi_n = normalize(poi)
    
    # Adaptive weights: start exploratory, become exploitative
    # Early: UCB and POI get more weight
    # Late: EI gets more weight
    t = min(n_evals / 30.0, 1.0)  # Progress through budget
    
    # Weights that sum to 1
    w_ei = 0.3 + 0.5 * t       # 0.3 -> 0.8
    w_ucb = 0.5 - 0.4 * t      # 0.5 -> 0.1
    w_poi = 0.2 - 0.1 * t      # 0.2 -> 0.1
    w_sum = w_ei + w_ucb + w_poi
    w_ei /= w_sum
    w_ucb /= w_sum
    w_poi /= w_sum
    
    # Combine in normalized space, then map back to a reasonable scale
    combined_n = w_ei * ei_n + w_ucb * ucb_n + w_poi * poi_n
    
    # Scale back: use EI's scale as reference
    ei_scale = np.max(ei) if np.max(ei) > 1e-15 else 1.0
    scores = combined_n * ei_scale
    
    return scores


def _compute_hypervolume_based_score(mu, sigma, f_best):
    """
    Hypervolume-based acquisition: estimate the expected hypervolume contribution
    of each candidate. A candidate at x contributes to hypervolume if it improves
    on f_best, and the magnitude of improvement times the "probability mass"
    of that improvement region.
    
    For a unimodal posterior, the expected improvement IS the 1D hypervolume
    contribution. But we can weight by how "central" the improvement is
    (i.e., not just tail events).
    
    HV_score(x) = EI(x) * exp(-|z(x)| / 2)
    where z = (f_best - mu)/sigma. This down-weights extreme tail events.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    ei = _compute_ei(mu, sigma, f_best)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -8.0, 8.0)
    # Down-weight very large positive z (unlikely to improve much)
    weight = np.exp(-np.abs(z_clipped) / 2.0)
    scores = ei * weight
    return scores


def _compute_bayes_factor_score(mu, sigma, f_best, prior_scale=0.1):
    """
    Bayesian Factor: ratio of evidence for "this point is better than f_best"
    vs "this point is not better than f_best".
    
    Using a simple likelihood ratio with a vague prior:
    BF = P(f < f_best | mu, sigma) / P(f >= f_best | mu, sigma)
       = Phi(z) / (1 - Phi(z))
    where z = (f_best - mu)/sigma
    
    This is the log-odds of improvement, which is a natural scoring function.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -8.0, 8.0)
    poi = norm.cdf(z_clipped)
    poi = np.clip(poi, 1e-15, 1.0 - 1e-15)
    log_bf = np.log(poi / (1.0 - poi))
    return log_bf


def _compute_moment_matching_score(mu, sigma, f_best):
    """
    Score based on matching the first two moments of the improvement distribution.
    I = max(f_best - f(x), 0)
    E[I] = EI (standard)
    Var[I] can be computed analytically.
    
    Score = E[I] - lambda * sqrt(Var[I])
    This penalizes candidates with high variance in improvement (risky bets)
    while still rewarding expected improvement.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -8.0, 8.0)
    
    # E[I] = (f_best - mu)*Phi(z) + sigma*phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # E[I^2] = (f_best - mu)^2 * Phi(z) + (f_best - mu)*sigma*z*phi(z) + sigma^2*(1 - z*phi(z))
    # Actually, let's use: E[I^2] = (f_best-mu)^2*Phi(z) + 2*(f_best-mu)*sigma*phi(z) + sigma^2*(1 - z*phi(z))
    # Hmm, let me be more careful.
    # I = (f_best - f) * 1{f < f_best}
    # E[I^2] = E[(f_best - f)^2 * 1{f < f_best}]
    # = (f_best-mu)^2 * P(f<f_best) + 2*(f_best-mu)*E[f*1{f<f_best}] - E[f^2*1{f<f_best}]
    # This is getting complex. Let's use a simpler approach:
    # Var[I] ≈ sigma^2 * phi(z)^2 / Phi(z)^2 (for large z, but this breaks down)
    # 
    # Simpler: just use EI as the score, but with a different weighting.
    # Let's skip the variance and just use a smoothed EI.
    
    # Use a "soft" EI that smoothly transitions
    soft_ei = ei * (1.0 + 0.1 * np.exp(-z_clipped**2 / 2.0))
    
    return soft_ei


def _adaptive_multi_score(mu, sigma, f_best, call_count):
    """
    Adaptive scoring that switches between different acquisition functions
    based on the stage and properties of the current posterior.
    
    Stage 1 (early, high uncertainty): UCB with high beta
    Stage 2 (mid, moderate uncertainty): EI
    Stage 3 (late, low uncertainty): Pure exploitation (negative mean)
    """
    n = len(mu)
    mean_sigma = np.mean(sigma)
    max_sigma = np.max(sigma)
    min_sigma = np.min(sigma)
    sigma_range = max_sigma - min_sigma
    
    # Determine stage
    if call_count <= 5:
        stage = 1
    elif call_count <= 15:
        stage = 2
    else:
        stage = 3
    
    # Also consider uncertainty level
    if mean_sigma > 0.1:
        stage = min(stage, 1)
    elif mean_sigma < 0.01:
        stage = max(stage, 3)
    
    if stage == 1:
        # Exploration: UCB with high beta
        beta = 3.0
        scores = _compute_ucb(mu, sigma, beta)
    elif stage == 2:
        # Balanced: EI with slight exploration
        ei = _compute_ei(mu, sigma, f_best)
        # Add small UCB component
        ucb = _compute_ucb(mu, sigma, 1.0)
        # Normalize and combine
        ei_n = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-15)
        ucb_n = (ucb - np.min(ucb)) / (np.max(ucb) - np.min(ucb) + 1e-15)
        scores = 0.7 * ei_n + 0.3 * ucb_n
        # Scale to EI range
        scores = scores * (np.max(ei) + 1e-15)
    else:
        # Exploitation: primarily EI, fallback to negative mean
        ei = _compute_ei(mu, sigma, f_best)
        # If EI is very small everywhere, fall back to exploitation
        if np.max(ei) < 1e-6:
            scores = -mu
        else:
            # Blend EI with a small exploitation component
            scores = ei + 0.05 * (-mu)
    
    return np.where(np.isfinite(scores), scores, 0.0)


# Track call count for adaptive behavior
_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using an adaptive multi-method approach.
    
    The key structural change: instead of a single fixed acquisition formula,
    we use an ensemble of different acquisition perspectives (EI, UCB, POI,
    Nash Equilibrium, Hypervolume, Bayesian Factor) and combine them adaptively
    based on the optimization stage and posterior properties.
    
    This is a model-free "virtual committee" approach where multiple
    acquisition criteria vote, with weights that shift over time.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means evaluate next.
    """
    global _call_count
    _call_count += 1
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        return -mu
    
    # Ensure finite inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.maximum(sigma, 1e-9)
    
    n = len(mu)
    
    # Compute multiple acquisition scores
    # 1. EI
    ei = _compute_ei(mu, sigma, f_best)
    
    # 2. UCB with adaptive beta
    beta = 2.5 * np.exp(-_call_count / 8.0)
    ucb = _compute_ucb(mu, sigma, beta)
    
    # 3. POI
    poi = _compute_poi(mu, sigma, f_best, gamma=0.003)
    
    # 4. Nash Equilibrium
    ne = _compute_nash_equilibrium_score(mu, sigma, f_best)
    
    # 5. Bayesian Factor (log-odds)
    bf = _compute_bayes_factor_score(mu, sigma, f_best)
    
    # 6. Hypervolume-based
    hv = _compute_hypervolume_based_score(mu, sigma, f_best)
    
    # Normalize each to [0, 1]
    def safe_normalize(s):
        s_min, s_max = np.min(s), np.max(s)
        if s_max - s_min < 1e-15:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ei_n = safe_normalize(ei)
    ucb_n = safe_normalize(ucb)
    poi_n = safe_normalize(poi)
    ne_n = safe_normalize(ne)
    bf_n = safe_normalize(bf)
    hv_n = safe_normalize(hv)
    
    # Adaptive weights based on stage
    t = min(_call_count / 25.0, 1.0)
    
    # Early: favor UCB, POI, NE
    # Mid: favor EI, HV
    # Late: favor EI, BF
    w_ei = 0.2 + 0.5 * t
    w_ucb = 0.4 - 0.35 * t
    w_poi = 0.15 - 0.05 * t
    w_ne = 0.1 + 0.05 * t
    w_bf = 0.05 + 0.1 * t
    w_hv = 0.1 + 0.05 * t
    
    # Also adjust based on uncertainty
    mean_sigma = np.mean(sigma)
    if mean_sigma > 0.05:
        # High uncertainty: boost exploration
        w_ucb += 0.1
        w_poi += 0.05
    elif mean_sigma < 0.01:
        # Low uncertainty: boost exploitation
        w_ei += 0.1
        w_bf += 0.05
    
    # Normalize weights
    w_sum = w_ei + w_ucb + w_poi + w_ne + w_bf + w_hv
    w_ei /= w_sum
    w_ucb /= w_sum
    w_poi /= w_sum
    w_ne /= w_sum
    w_bf /= w_sum
    w_hv /= w_sum
    
    # Combine
    combined_n = (w_ei * ei_n + w_ucb * ucb_n + w_poi * poi_n +
                  w_ne * ne_n + w_bf * bf_n + w_hv * hv_n)
    
    # Scale to a reasonable range: use EI's scale as reference
    ei_scale = np.max(ei) if np.max(ei) > 1e-15 else 1.0
    scores = combined_n * ei_scale
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
