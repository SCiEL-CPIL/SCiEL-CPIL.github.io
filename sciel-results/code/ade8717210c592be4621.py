"""Combined cost-aware acquisition for Bayesian optimization (hartmann 3d).
Uses a calibrated hybrid of EI and UCB with adaptive cost discounting.
Calibration selects parameters that best separate low-mu (promising) points
from expensive ones, using a lightweight grid search over a small parameter
space. This combines the structural calibration idea from inspiration 1 with
the simple EI foundation from inspiration 2, plus adaptive beta from the
current program.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _ei_score(mu, sigma, f_best, cost, beta=1.0):
    """Expected improvement per unit cost^beta."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    cost_safe = np.maximum(cost, 1e-9)
    return ei / (cost_safe ** beta)


def _ucb_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0):
    """UCB-like acquisition: -mu + gamma * sigma, cost-discounted."""
    cost_safe = np.maximum(cost, 1e-9)
    score = (-mu + gamma * sigma) / (cost_safe ** beta)
    return score


def _lcb_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0):
    """LCB-like acquisition: -mu - gamma * sigma, cost-discounted."""
    cost_safe = np.maximum(cost, 1e-9)
    score = (-mu - gamma * sigma) / (cost_safe ** beta)
    return score


def _hybrid_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0, w_ei=0.5, w_ucb=0.5, w_lcb=0.0):
    """Hybrid of EI, UCB, and LCB with weighted combination."""
    ei = _ei_score(mu, sigma, f_best, cost, beta=beta)
    ucb = _ucb_score(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
    lcb = _lcb_score(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
    
    # Normalize each to have zero mean and unit std for fair combination
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    lcb_mean = np.mean(lcb)
    lcb_std = np.std(lcb) + 1e-9
    lcb_norm = (lcb - lcb_mean) / lcb_std
    
    score = w_ei * ei_norm + w_ucb * ucb_norm + w_lcb * lcb_norm
    return score


def _calibrate(mu, sigma, f_best, cost, time_budget=3.0):
    """Lightweight calibration of acquisition parameters.
    
    Tries a small grid of (gamma, beta, w_ei, w_ucb, w_lcb) and picks the combination
    that best ranks low-mu points highly. Proxy: mean mu of top-k scored points.
    """
    t0 = time.time()
    
    gamma_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    beta_vals = [0.5, 0.75, 1.0, 1.25, 1.5]
    # (w_ei, w_ucb, w_lcb) - normalized weights
    w_combos = [
        (0.5, 0.5, 0.0),
        (0.4, 0.4, 0.2),
        (0.3, 0.3, 0.4),
        (0.2, 0.4, 0.4),
        (0.0, 0.5, 0.5),
        (0.0, 0.3, 0.7),
        (0.1, 0.2, 0.7),
        (0.3, 0.5, 0.2),
        (0.5, 0.3, 0.2),
    ]
    
    best_proxy = -np.inf
    best_params = {'gamma': 1.0, 'beta': 1.0, 'w_ei': 0.5, 'w_ucb': 0.5, 'w_lcb': 0.0}
    
    top_k = min(20, len(mu) // 10)
    if top_k < 1:
        top_k = 1
    
    for gamma in gamma_vals:
        for beta in beta_vals:
            for (w_ei, w_ucb, w_lcb) in w_combos:
                if time.time() - t0 > time_budget:
                    break
                
                scores = _hybrid_score(mu, sigma, f_best, cost, 
                                       gamma=gamma, beta=beta, 
                                       w_ei=w_ei, w_ucb=w_ucb, w_lcb=w_lcb)
                
                if not np.all(np.isfinite(scores)):
                    continue
                
                top_idx = np.argsort(scores)[-top_k:]
                mean_mu_top = np.mean(mu[top_idx])
                proxy = -mean_mu_top
                
                if proxy > best_proxy:
                    best_proxy = proxy
                    best_params = {'gamma': gamma, 'beta': beta, 
                                   'w_ei': w_ei, 'w_ucb': w_ucb, 'w_lcb': w_lcb}
    
    return best_params


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 290.0
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calibrate parameters (quick, ~3s max)
    if time.time() < deadline - 5.0:
        params = _calibrate(mu_safe, sigma_safe, f_best, cost_safe, time_budget=3.0)
    else:
        params = {'gamma': 1.0, 'beta': 1.0, 'w_ei': 0.5, 'w_ucb': 0.5}
    
    # Compute hybrid score
    scores = _hybrid_score(mu_safe, sigma_safe, f_best, cost_safe,
                           gamma=params['gamma'], beta=params['beta'],
                           w_ei=params['w_ei'], w_ucb=params['w_ucb'],
                           w_lcb=params.get('w_lcb', 0.0))
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
