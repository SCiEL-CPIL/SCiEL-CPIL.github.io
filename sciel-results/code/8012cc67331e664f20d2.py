import numpy as np
import time
import random
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    timeout = 580  # Safety margin for 600s limit

    # Pre-compute indices for projections
    # P is indexed [y, x]
    # y in 0..9, x in 0..9
    ys = np.arange(n).reshape(-1, 1) # 10x1
    xs = np.arange(n).reshape(1, -1) # 1x10
    idx_neg1 = (xs - ys).flatten() + n - 1  # x - y + 9, range 0..18
    idx_0 = xs.flatten()                    # x, range 0..9
    idx_1 = (xs + ys).flatten()             # x + y, range 0..18
    idx_2 = (xs + 2 * ys).flatten()         # x + 2y, range 0..27

    def compute_ratio(P):
        """
        Computes D / M where D = H_{-1} and M = max(H_0, H_1, H_2).
        P is a 10x10 array.
        """
        if not np.all(np.isfinite(P)):
            return 0.0
        P = np.clip(P, 0, None)
        total = np.sum(P)
        if total < 1e-9:
            return 0.0
        P = P / total
        P_flat = P.flatten()

        # Helper to get entropy from projection indices
        def get_entropy(indices):
            counts = np.zeros(np.max(indices) + 1)
            np.add.at(counts, indices, P_flat)
            p_nonzero = counts[counts > 1e-12]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))

        H_neg1 = get_entropy(idx_neg1)
        H_0 = get_entropy(idx_0)
        H_1 = get_entropy(idx_1)
        H_2 = get_entropy(idx_2)

        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return H_neg1 / M

    # Objective function for minimization
    # We want to maximize ratio, so minimize -ratio.
    # We parameterize P using softmax of theta to ensure P >= 0 and sum(P)=1.
    def objective(theta):
        # Softmax
        theta = theta - np.max(theta) # for numerical stability
        exp_theta = np.exp(theta)
        P = exp_theta / np.sum(exp_theta)
        P = P.reshape((n, n))
        ratio = compute_ratio(P)
        if ratio <= 0:
            return 1e6 # Large penalty
        return -ratio

    # Initial best
    P_best = np.ones((n, n)) / (n * n)
    best_score = compute_ratio(P_best)
    
    # Try a few structured initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.zeros(n*n))
    
    # 2. Anti-diagonal (y = 9 - x)
    theta_ad = np.full(n*n, -10.0)
    for x in range(n):
        y = n - 1 - x
        theta_ad[y*n + x] = 0.0
    initializations.append(theta_ad)
    
    # 3. Main diagonal (y = x)
    theta_md = np.full(n*n, -10.0)
    for x in range(n):
        y = x
        theta_md[y*n + x] = 0.0
    initializations.append(theta_md)
        
    # 4. Random sparse
    for _ in range(5):
        theta_r = np.full(n*n, -10.0)
        mask = np.random.rand(n*n) > 0.9
        theta_r[mask] = 0.0
        initializations.append(theta_r)

    # Optimization
    for i, init_theta in enumerate(initializations):
        if time.time() - start_time > timeout:
            break
            
        # Use Nelder-Mead for derivative-free optimization
        # We need to be careful with the number of iterations
        try:
            result = minimize(
                objective, 
                init_theta, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.success or result.fun < -best_score:
                theta_opt = result.x
                theta_opt = theta_opt - np.max(theta_opt)
                P_opt = np.exp(theta_opt) / np.sum(np.exp(theta_opt))
                P_opt = P_opt.reshape((n, n))
                score_opt = compute_ratio(P_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    P_best = P_opt
        except Exception:
            pass

    # If we have time left, do a local polish on the best solution found
    if time.time() - start_time < timeout - 60:
        # Convert P_best back to theta
        P_best_clipped = np.clip(P_best, 1e-12, None)
        P_best_clipped = P_best_clipped / np.sum(P_best_clipped)
        theta_best = np.log(P_best_clipped).flatten()
        
        try:
            # Use Powell for a potentially better local search
            result = minimize(
                objective, 
                theta_best, 
                method='Powell', 
                options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if result.fun < -best_score:
                theta_opt = result.x
                theta_opt = theta_opt - np.max(theta_opt)
                P_opt = np.exp(theta_opt) / np.sum(np.exp(theta_opt))
                P_opt = P_opt.reshape((n, n))
                score_opt = compute_ratio(P_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    P_best = P_opt
        except Exception:
            pass

    # Final cleanup
    P_best = np.clip(P_best, 0, None)
    total = np.sum(P_best)
    if total < 1e-9:
        return np.ones((n, n)) / (n * n)
    P_best = P_best / total
    
    return P_best

# EVOLVE-BLOCK-END
