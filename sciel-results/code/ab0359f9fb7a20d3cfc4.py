# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600 # Safety margin
    
    # Pre-compute grid coordinates
    y, x = np.mgrid[0:n, 0:n]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    
    # Pre-compute projections for slopes
    # H_{-1}: x - y
    proj_m1 = x - y
    # H_0: x
    proj_0 = x
    # H_1: x + y
    proj_1 = x + y
    # H_2: x + 2y
    proj_2 = x + 2*y

    def compute_entropies(P):
        """Compute H_{-1}, H_0, H_1, H_2 for a given distribution P."""
        # Flatten P to iterate over unique projection values
        # It is faster to use np.bincount for the weights
        
        # H_{-1}
        # Values of x-y range from -(n-1) to n-1. Offset by n-1 to make indices non-negative.
        offset = n - 1
        idx_m1 = (proj_m1 + offset).astype(np.int32).ravel()
        weights_m1 = P.ravel()
        counts_m1 = np.bincount(idx_m1, weights=weights_m1, minlength=2*n-1)
        # Filter zeros to avoid log(0)
        probs_m1 = counts_m1[counts_m1 > 0]
        H_m1 = -np.sum(probs_m1 * np.log2(probs_m1)) if len(probs_m1) > 0 else 0.0

        # H_0 (x)
        idx_0 = x.astype(np.int32).ravel()
        counts_0 = np.bincount(idx_0, weights=weights_m1, minlength=n)
        probs_0 = counts_0[counts_0 > 0]
        H_0 = -np.sum(probs_0 * np.log2(probs_0)) if len(probs_0) > 0 else 0.0

        # H_1 (x+y)
        idx_1 = (x + y).astype(np.int32).ravel()
        counts_1 = np.bincount(idx_1, weights=weights_m1, minlength=2*n-1)
        probs_1 = counts_1[counts_1 > 0]
        H_1 = -np.sum(probs_1 * np.log2(probs_1)) if len(probs_1) > 0 else 0.0

        # H_2 (x+2y)
        # Max value is 9 + 2*9 = 27. Min is 0.
        idx_2 = (x + 2*y).astype(np.int32).ravel()
        counts_2 = np.bincount(idx_2, weights=weights_m1, minlength=3*n-1)
        probs_2 = counts_2[counts_2 > 0]
        H_2 = -np.sum(probs_2 * np.log2(probs_2)) if len(probs_2) > 0 else 0.0

        return H_m1, H_0, H_1, H_2

    def objective(params):
        """
        Objective function to minimize: - (H_m1 / max(H_0, H_1, H_2))
        params: [mu_x, mu_y, sigma_x, sigma_y, log_rho]
        Using log_rho to constrain rho in (-1, 1) via tanh.
        """
        if time.time() > deadline - 10:
            return 1000.0 # Penalize if we're running out of time

        mu_x, mu_y, sigma_x, sigma_y, log_rho = params
        
        # Ensure sigmas are positive
        sigma_x = np.abs(sigma_x) + 1e-6
        sigma_y = np.abs(sigma_y) + 1e-6
        
        # Map log_rho to rho in (-1, 1)
        rho = np.tanh(log_rho)
        
        # Construct covariance matrix
        cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                        [rho * sigma_x * sigma_y, sigma_y**2]])
        
        # Inverse covariance
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return 1000.0

        # Compute quadratic form Q(x,y) = (v - mu)^T inv_cov (v - mu)
        # v is column vector [x, y]^T
        dx = x - mu_x
        dy = y - mu_y
        
        # Q = inv_cov[0,0]*dx^2 + 2*inv_cov[0,1]*dx*dy + inv_cov[1,1]*dy^2
        Q = inv_cov[0,0]*dx**2 + 2*inv_cov[0,1]*dx*dy + inv_cov[1,1]*dy**2
        
        # Weights are exp(-0.5 * Q)
        # To avoid underflow/overflow, shift by min Q
        Q_min = np.min(Q)
        weights = np.exp(-0.5 * (Q - Q_min))
        
        # Normalize
        sum_w = np.sum(weights)
        if sum_w == 0 or not np.isfinite(sum_w):
            return 1000.0
        P = weights / sum_w
        
        # Calculate entropies
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 1000.0
            
        ratio = H_m1 / M
        return -ratio # Minimize negative ratio

    # Initial guesses based on "Anisotropic Gaussian with negative correlation"
    # sigma_x approx 2*sigma_y, negative correlation
    # Grid is 0..9, center 4.5
    initial_params = [4.5, 4.5, 2.0, 1.0, np.arctanh(-0.5)] 
    # [mu_x, mu_y, sigma_x, sigma_y, log_rho]
    # log_rho = arctanh(-0.5) ~ -0.549
    
    # Also try uniform as a baseline
    P_uniform = np.ones((n, n)) / (n*n)
    H_m1_u, H_0_u, H_1_u, H_2_u = compute_entropies(P_uniform)
    M_u = max(H_0_u, H_1_u, H_2_u)
    best_ratio = H_m1_u / M_u if M_u > 0 else 0.0
    best_P = P_uniform.copy()

    # Optimization
    # We'll run a few random restarts to avoid local minima
    restarts = 10
    for i in range(restarts):
        if time.time() > deadline - 30:
            break
            
        if i == 0:
            p0 = initial_params
        else:
            # Random perturbations
            p0 = [4.5 + np.random.uniform(-2, 2), 
                  4.5 + np.random.uniform(-2, 2), 
                  np.random.uniform(0.5, 4.0), 
                  np.random.uniform(0.5, 2.0), 
                  np.random.uniform(-1, 1)]
        
        res = minimize(objective, p0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6})
        
        if res.success or res.fun < -best_ratio:
            # Reconstruct P from best params
            mu_x, mu_y, sigma_x, sigma_y, log_rho = res.x
            sigma_x = np.abs(sigma_x) + 1e-6
            sigma_y = np.abs(sigma_y) + 1e-6
            rho = np.tanh(log_rho)
            
            cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                            [rho * sigma_x * sigma_y, sigma_y**2]])
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                continue
                
            dx = x - mu_x
            dy = y - mu_y
            Q = inv_cov[0,0]*dx**2 + 2*inv_cov[0,1]*dx*dy + inv_cov[1,1]*dy**2
            Q_min = np.min(Q)
            weights = np.exp(-0.5 * (Q - Q_min))
            sum_w = np.sum(weights)
            if sum_w > 0 and np.isfinite(sum_w):
                P_candidate = weights / sum_w
                H_m1_c, H_0_c, H_1_c, H_2_c = compute_entropies(P_candidate)
                M_c = max(H_0_c, H_1_c, H_2_c)
                if M_c > 1e-9:
                    ratio_c = H_m1_c / M_c
                    if ratio_c > best_ratio:
                        best_ratio = ratio_c
                        best_P = P_candidate

    # Final validation and cleanup
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = P_uniform

    return best_P

# EVOLVE-BLOCK-END
