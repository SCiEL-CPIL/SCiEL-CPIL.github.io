# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses product-of-marginals model: P[y,x] = R[y] * C[x] / (sum(R)*sum(C))
    This reduces the 100-dim search to 20 dimensions.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Pre-compute grid coordinates
    y_grid, x_grid = np.mgrid[0:n, 0:n]
    x_grid = x_grid.astype(np.float64)
    y_grid = y_grid.astype(np.float64)
    
    def _shannon_entropy(probs):
        """Compute Shannon entropy in bits, ignoring zero probabilities."""
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def compute_entropies_from_marginals(R, C):
        """
        Compute H_{-1}, H_0, H_1, H_2 from row marginal R (length n) and column marginal C (length n).
        P[y,x] = R[y] * C[x] (assumed normalized: sum(R)=1, sum(C)=1)
        """
        # H_0: distribution of X = C (since P[y,x] = R[y]*C[x], marginal of X is C)
        H_0 = _shannon_entropy(C)
        
        # H_{-1}: distribution of X - Y
        # For each value d = x - y, probability = sum_{x-y=d} R[y]*C[x]
        # d ranges from -(n-1) to n-1
        d_values = np.arange(-(n-1), n)  # -9 to 9
        prob_m1 = np.zeros(len(d_values))
        for i, d in enumerate(d_values):
            # x - y = d => x = y + d
            # y ranges from max(0, -d) to min(n-1, n-1-d)
            y_min = max(0, -d)
            y_max = min(n-1, n-1-d)
            if y_min <= y_max:
                ys = np.arange(y_min, y_max+1)
                xs = ys + d
                prob_m1[i] = np.sum(R[ys] * C[xs])
        H_m1 = _shannon_entropy(prob_m1)
        
        # H_1: distribution of X + Y
        # s = x + y, ranges from 0 to 2*(n-1)
        s_values = np.arange(0, 2*n-1)
        prob_1 = np.zeros(len(s_values))
        for i, s in enumerate(s_values):
            # x + y = s => x = s - y
            # y ranges from max(0, s-(n-1)) to min(n-1, s)
            y_min = max(0, s-(n-1))
            y_max = min(n-1, s)
            if y_min <= y_max:
                ys = np.arange(y_min, y_max+1)
                xs = s - ys
                prob_1[i] = np.sum(R[ys] * C[xs])
        H_1 = _shannon_entropy(prob_1)
        
        # H_2: distribution of X + 2Y
        # v = x + 2y, ranges from 0 to (n-1) + 2*(n-1) = 3*(n-1)
        v_max = 3*(n-1)
        prob_2 = np.zeros(v_max+1)
        for y in range(n):
            for x in range(n):
                v = x + 2*y
                prob_2[v] += R[y] * C[x]
        H_2 = _shannon_entropy(prob_2)
        
        return H_m1, H_0, H_1, H_2
    
    def get_ratio_from_marginals(R, C):
        """Compute D/M ratio from marginals."""
        H_m1, H_0, H_1, H_2 = compute_entropies_from_marginals(R, C)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0
        return H_m1 / M
    
    # Strategy: Global optimization on 20-dim parameter space (log R, log C)
    # Represent R and C via softmax of parameters to ensure positivity and sum=1
    
    def softmax(params):
        """Convert params to probability vector via softmax."""
        exp_params = np.exp(params - np.max(params))
        return exp_params / np.sum(exp_params)
    
    def objective_marginals(params):
        """
        Objective: minimize -ratio
        params: [log_R_0, ..., log_R_9, log_C_0, ..., log_C_9]
        """
        if time.time() > deadline - 10:
            return 1000.0
        
        log_R = params[:n]
        log_C = params[n:2*n]
        
        R = softmax(log_R)
        C = softmax(log_C)
        
        ratio = get_ratio_from_marginals(R, C)
        return -ratio
    
    # Bounds for differential evolution
    # Log probabilities can be in [-10, 10] effectively
    bounds = [(-10, 10)] * (2 * n)
    
    best_ratio = 0.0
    best_R = None
    best_C = None
    
    # Baseline: uniform
    R_uniform = np.ones(n) / n
    C_uniform = np.ones(n) / n
    ratio_uniform = get_ratio_from_marginals(R_uniform, C_uniform)
    best_ratio = ratio_uniform
    best_R = R_uniform.copy()
    best_C = C_uniform.copy()
    
    # Run Differential Evolution
    if time.time() < deadline - 30:
        try:
            # Use a fixed seed for reproducibility if needed, but DE is stochastic
            # maxiter=100, popsize=15 (default is 15)
            # tol=1e-5
            res_de = differential_evolution(objective_marginals, bounds, maxiter=100, popsize=15, tol=1e-5, seed=42)
            if res_de.fun < -best_ratio:
                best_params = res_de.x
                R_opt = softmax(best_params[:n])
                C_opt = softmax(best_params[n:])
                ratio_opt = get_ratio_from_marginals(R_opt, C_opt)
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_R = R_opt.copy()
                    best_C = C_opt.copy()
        except Exception:
            pass
    
    # Refine with L-BFGS-B
    if time.time() < deadline - 10:
        best_params = np.concatenate([
            np.log(best_R + 1e-10),
            np.log(best_C + 1e-10)
        ])
        
        try:
            res_lbfgs = minimize(objective_marginals, best_params, method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
            if res_lbfgs.fun < -best_ratio:
                R_opt = softmax(res_lbfgs.x[:n])
                C_opt = softmax(res_lbfgs.x[n:])
                ratio_opt = get_ratio_from_marginals(R_opt, C_opt)
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_R = R_opt.copy()
                    best_C = C_opt.copy()
        except Exception:
            pass
            
    # Construct final P from best marginals
    P = np.outer(best_C, best_R)  # P[y,x] = C[x] * R[y]
    
    # Validate and normalize
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n*n)
    
    return P

# EVOLVE-BLOCK-END
