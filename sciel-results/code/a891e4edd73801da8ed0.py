import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin below 330s wall-clock

# ---------- Cosine8 8d (f15 in the DRL/BOB suite) ----------
def _cosine8(x):
    """x shape (n, 8) in [-1,1]^8. Returns (n,) values; global min -0.8."""
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(-1, 8)
    out = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(8):
        out += (1.0 + np.sin(np.pi * x[:, i])) / (1.0 + i)
    base = sum(1.0 / (1.0 + i) for i in range(8))
    out = out - base
    out = out - 0.8
    return out

# ---------- Instance k: minimizer moved to uniform point in inner 80% ----------
def _instance_offset(k, dim=8):
    rng = np.random.RandomState(1000 + k)
    s = rng.uniform(-0.8, 0.8, size=dim)
    return s

def _f_instance(x, k):
    off = _instance_offset(k)
    return _cosine8(x - off)

# ---------- Cost model ----------
def _cost_of(x):
    scaled = (np.asarray(x, dtype=np.float64) + 1.0) / 2.0
    return 1.0 + np.mean(scaled, axis=-1)

# ---------- GP prediction utilities ----------
def _kernel_matern52(A, B, ls=0.2, noise=1e-6):
    A = (A + 1.0) / 2.0
    B = (B + 1.0) / 2.0
    d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    r = np.sqrt(np.maximum(d, 0.0)) / ls
    K = np.sqrt(5.0 * r) * np.exp(-r)
    if A.shape[0] == B.shape[0] and A.shape[0] == 1:
        pass
    elif A.shape[0] == B.shape[0]:
        K[np.arange(len(A)), np.arange(len(A))] += noise
    return K

def _gp_predict(X, Y, C):
    """
    X: (n, d) observed points
    Y: (n,) observed values
    C: (m, d) candidate points
    Returns mu: (m,), sigma: (m,) in objective units
    """
    mu_y = Y.mean()
    sd_y = Y.std() + 1e-9
    Ys = (Y - mu_y) / sd_y
    
    K = _kernel_matern52(X, C)
    Kss = np.diag(_kernel_matern52(C, C))
    
    Kxx = _kernel_matern52(X, X)
    try:
        L = np.linalg.cholesky(Kxx)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Ys))
    except np.linalg.LinAlgError:
        try:
            Kinv = np.linalg.inv(Kxx + 1e-8 * np.eye(len(X)))
            alpha = Kinv @ Ys
        except np.linalg.LinAlgError:
            Kinv = np.linalg.pinv(Kxx + 1e-8 * np.eye(len(X)))
            alpha = Kinv @ Ys
    
    mu_std = K @ alpha
    v = np.linalg.solve(L, K.T) if 'L' in dir() else None
    if v is None:
        v = np.linalg.cholesky(Kxx + 1e-8 * np.eye(len(X)))
        v = np.linalg.solve(v.T, np.linalg.solve(v, K.T))
    var_std = Kss - np.sum(v * v, axis=0)
    var_std = np.maximum(var_std, 1e-9)
    sigma_std = np.sqrt(var_std)
    
    mu_obj = mu_std * sd_y + mu_y
    sigma_obj = sigma_std * sd_y
    return mu_obj, sigma_obj

# ---------- Quantile-based acquisition ----------
def _quantile_score_candidates(mu, sigma, f_best, cost):
    """
    Quantile-based acquisition function.
    Score = PI / cost^2, where PI is the probability of improvement.
    This favors candidates with high confidence of beating the current best
    while heavily penalizing expensive evaluations.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probability of improvement: P(f(x) < f_best)
    # = P((f(x) - mu) / sigma < (f_best - mu) / sigma)
    # = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Square the cost penalty to heavily penalize expensive evaluations
    cost_penalty = cost_safe ** 2
    
    # Score = PI / cost^2
    score = pi / cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score

def _adaptive_lcb_score_candidates(mu, sigma, f_best, cost):
    """
    Adaptive LCB acquisition function with cost penalty.
    Score = (f_best - (mu - beta * sigma)) / cost
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive beta: increases with uncertainty
    # Use a beta that scales with the typical sigma
    sigma_mean = sigma.mean()
    if sigma_mean > 0.01:
        beta = 1.5 + 0.5 * (sigma_mean / 0.1)
    else:
        beta = 1.5
    
    lcb = mu - beta * sigma_safe
    improvement = f_best - lcb
    score = improvement / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Returns scores of shape (2000,), higher = evaluate next.
    """
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, -1e9)
        return score
    
    # Combine quantile-based and adaptive LCB approaches
    # Weight them based on the current uncertainty
    sigma_mean = sigma.mean()
    
    # When uncertainty is high, use more quantile-based (exploration)
    # When uncertainty is low, use more LCB (exploitation)
    if sigma_mean > 0.05:
        w_quantile = 0.6
        w_lcb = 0.4
    else:
        w_quantile = 0.3
        w_lcb = 0.7
    
    score_quantile = _quantile_score_candidates(mu, sigma, f_best, cost)
    score_lcb = _adaptive_lcb_score_candidates(mu, sigma, f_best, cost)
    
    # Normalize both scores to be comparable
    score_quantile_norm = (score_quantile - score_quantile.min()) / (score_quantile.max() - score_quantile.min() + 1e-9)
    score_lcb_norm = (score_lcb - score_lcb.min()) / (score_lcb.max() - score_lcb.min() + 1e-9)
    
    # Combine
    combined = w_quantile * score_quantile_norm + w_lcb * score_lcb_norm
    
    return combined
