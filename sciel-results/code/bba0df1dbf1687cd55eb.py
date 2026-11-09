from scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize_scalar

def compute_prob(support, weights):
    w_sum = np.sum(weights)
    if w_sum < 1e-9:
        return 0.0
    w = weights / w_sum
    n = len(support)
    S = support[:, None] + support[None, :]
    T = 2.0 * support[None, :] - support[:, None]
    
    Q = np.zeros((n, n))
    for k in range(n):
        for l in range(n):
            mask = S < T[k, l]
            Q[k, l] = np.sum(w[:, None] * w[None, :] * mask)
            
    P = np.sum(w[:, None] * w[None, :] * Q)
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    # Approach: Constructive optimization on a 3-point support [0, 1, K]
    # Variables: u (weight of 0 and 1 combined), a_prime (relative weight of 0 in the 0-1 part)
    # Weights: w_0 = a_prime * u, w_1 = (1 - a_prime) * u, w_K = 1 - u
    # Support: [0, 1, K]
    
    def objective(params):
        u = params[0]
        a_prime = params[1]
        if u < 0 or u > 1 or a_prime < 0 or a_prime > 1:
            return 1.0
        K = 10.0
        support = np.array([0.0, 1.0, K])
        w_0 = a_prime * u
        w_1 = (1.0 - a_prime) * u
        w_K = 1.0 - u
        weights = np.array([w_0, w_1, w_K])
        if np.any(weights < -1e-9):
            return 1.0
        p = compute_prob(support, weights)
        return -p

    # Initial guess based on analysis: u ~ 0.6, a_prime ~ 0.35
    x0 = np.array([0.6, 0.35])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
    
    u_opt = res.x[0]
    a_prime_opt = res.x[1]
    
    K = 10.0
    support = np.array([0.0, 1.0, K])
    w_0 = a_prime_opt * u_opt
    w_1 = (1.0 - a_prime_opt) * u_opt
    w_K = 1.0 - u_opt
    weights = np.array([w_0, w_1, w_K])
    
    # Ensure valid
    if np.any(weights < -1e-9) or np.sum(weights) < 1e-9:
        # Fallback to best known 2-point
        return np.array([0.0, 10.0]), np.array([0.35, 0.65])
        
    return support, weights
