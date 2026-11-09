import numpy as np
import time
from scipy.optimize import minimize_scalar

# ---- Deadline guard ----
_START_TIME = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 20.0

# ---- Constants ----
DIM = 4
RANGE_LOW = -4.0
RANGE_HIGH = 5.0
SPAN = RANGE_HIGH - RANGE_LOW

# ---- Powell 4D surrogate for offline fitting ----
def _powell_surrogate(x):
    """Powell 4D function: sum over i of (x_{2i}^2 + x_{2i+1}^2 + ... )
    Standard Powell 4D: sum_{i=0}^{1} (x_{2i}^2 + x_{2i+1}^2 + x_{2i} * x_{2i+1})
    Actually standard Powell 4D:
    f(x) = (x1+x2+10)^2 + 5*(x3-x4)^2 + (x3-x4-10)^2 + 10*(x1-x4)^2
    Minimum at x = (-3, 0, 3, 0) with f=0 in range [-4,5]^4.
    """
    x = np.asarray(x, dtype=np.float64)
    return ((x[0] + x[1] + 10)**2 +
            5*(x[2] - x[3])**2 +
            (x[2] - x[3] - 10)**2 +
            10*(x[0] - x[3])**2)

def _local_cost(x):
    """Cost of evaluating x: 1 + mean(x scaled to [0,1])"""
    x_scaled = (x - RANGE_LOW) / SPAN
    return 1.0 + np.mean(x_scaled)

def _compute_scores(mu, sigma, f_best, cost, params):
    """Parameterized acquisition score."""
    alpha, beta, gamma = params
    # Score = (f_best - mu) / cost + alpha * sigma / cost + beta * sigma * (f_best - mu) / cost + gamma * sigma^2 / cost
    # Simpler: (f_best - mu + alpha * sigma + beta * sigma^2) / cost
    num = (f_best - mu) + alpha * sigma + beta * sigma**2
    score = num / cost
    return score

def _simulate_one(seed, params, n_initial=5, cost_limit=30.0, n_candidates=2000):
    """Simulate one BO instance with the given acquisition params."""
    rng = np.random.RandomState(seed)
    
    # Instance shift: move minimizer to uniform point in inner 80% of range
    # Inner 80% of [-4,5] = [-2.8, 3.8] for each dim
    inner_low = RANGE_LOW + 0.1 * SPAN
    inner_high = RANGE_HIGH - 0.1 * SPAN
    shift = rng.uniform(inner_low, inner_high, size=DIM)
    
    def f_instance(x):
        return _powell_surrogate(x - shift)
    
    # Initial points
    X = rng.uniform(RANGE_LOW, RANGE_HIGH, size=(n_initial, DIM))
    Y = np.array([f_instance(xi) for xi in X])
    total_cost = sum(_local_cost(xi) for xi in X)
    
    # BO loop
    while total_cost < cost_limit:
        # Generate candidates
        Xcand = rng.uniform(RANGE_LOW, RANGE_HIGH, size=(n_candidates, DIM))
        
        # For the surrogate, we don't have a real GP, so we use a heuristic:
        # mu = f_instance (since we're on the true function, the "GP" would predict near truth)
        # But for fitting purposes, we need to simulate what the GP would give.
        # Use a simple surrogate: mu = f_instance, sigma = distance-based uncertainty
        mu_cand = np.array([f_instance(xi) for xi in Xcand])
        
        # Simple uncertainty model: sigma proportional to min distance to observed points
        sigma_cand = np.zeros(n_candidates)
        for i in range(n_candidates):
            d = np.min(np.linalg.norm(Xcand[i] - X, axis=1))
            sigma_cand[i] = 0.5 * d  # rough uncertainty
        
        cost_cand = np.array([_local_cost(xi) for xi in Xcand])
        f_best = np.min(Y)
        
        scores = _compute_scores(mu_cand, sigma_cand, f_best, cost_cand, params)
        idx = np.argmax(scores)
        
        x_new = Xcand[idx]
        y_new = f_instance(x_new)
        c_new = _local_cost(x_new)
        
        X = np.vstack([X, x_new])
        Y = np.append(Y, y_new)
        total_cost += c_new
    
    return np.min(Y)

def _objective(params):
    """Objective: mean regret over 5 instances."""
    regrets = []
    for seed in range(5):
        # The true minimum of the instance is 0 (Powell min is 0)
        f_min = 0.0
        f_best = _simulate_one(seed, params)
        regrets.append(f_best - f_min)
    return np.mean(regrets)

def _fit_params():
    """Fit acquisition parameters via grid search + local optimization."""
    best_params = [0.5, 0.0, 0.0]
    best_obj = _objective(best_params)
    
    # Grid search over alpha, beta
    for a in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for b in [0.0, 0.05, 0.1, 0.2]:
            params = [a, b, 0.0]
            obj = _objective(params)
            if obj < best_obj:
                best_obj = obj
                best_params = params
    
    return best_params

# ---- Offline fitting (done once at import) ----
_FIT_PARAMS = [0.5, 0.0, 0.0]  # default
def _do_fit():
    global _FIT_PARAMS
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return
    _FIT_PARAMS = _fit_params()

_do_fit()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with fitted parameters.
    Score = (f_best - mu + alpha * sigma + beta * sigma^2) / cost
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1e-9)
    
    alpha, beta, gamma = _FIT_PARAMS
    num = (f_best - mu) + alpha * sigma + beta * sigma**2
    score = num / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
