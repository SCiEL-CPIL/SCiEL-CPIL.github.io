import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Use a parametric Gaussian ansatz: P(x,y) ∝ exp(-αx² - βy² - γxy).
    2. This reduces the 100-dimensional probability simplex to a 3-dimensional 
       parameter space (α, β, γ), capturing smooth, correlated diagonal structures.
    3. Perform a coarse global grid search over the parameter space to find 
       promising basins.
    4. Refine the best candidates using Nelder-Mead optimization.
    5. Return the best valid distribution found.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 610  # 20s safety buffer

    # Pre-compute grid coordinates
    x_grid, y_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Indices for binning
    # x - y + 9 -> 0..18
    idx_dm1 = (x_flat - y_flat + n - 1).astype(np.int32)
    # x -> 0..9
    idx_x = x_flat.astype(np.int32)
    # x + y -> 0..18
    idx_d1 = (x_flat + y_flat).astype(np.int32)
    # x + 2y -> 0..27
    idx_d2 = (x_flat + 2 * y_flat).astype(np.int32)

    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    def entropy_from_counts(counts):
        """Calculate Shannon entropy (bits) from a count/probability vector."""
        mask = counts > 1e-15
        if not np.any(mask):
            return 0.0
        p = counts[mask]
        p_sum = p.sum()
        if p_sum < 1e-15:
            return 0.0
        p = p / p_sum
        return float(-np.sum(p * np.log2(p)))

    def evaluate_P(P_flat):
        """
        Evaluate the score for a flattened probability vector P_flat.
        Assumes P_flat is non-negative and sums to ~1.
        """
        if not np.all(np.isfinite(P_flat)):
            return -np.inf
        
        # Clip and normalize
        P_norm = np.clip(P_flat, 0, None)
        s = P_norm.sum()
        if s < 1e-9:
            return -np.inf
        P_norm /= s

        c_dm1 = np.bincount(idx_dm1, weights=P_norm, minlength=bin_sizes['dm1'])
        c_x   = np.bincount(idx_x,   weights=P_norm, minlength=bin_sizes['x'])
        c_d1  = np.bincount(idx_d1,  weights=P_norm, minlength=bin_sizes['d1'])
        c_d2  = np.bincount(idx_d2,  weights=P_norm, minlength=bin_sizes['d2'])

        H_dm1 = entropy_from_counts(c_dm1)
        H_x   = entropy_from_counts(c_x)
        H_d1  = entropy_from_counts(c_d1)
        H_d2  = entropy_from_counts(c_d2)

        D = H_dm1
        M = max(H_x, H_d1, H_d2)
        
        if M < 1e-9:
            return -np.inf
            
        return D / M

    def get_P_from_params(params):
        """
        Compute the normalized probability distribution from Gaussian parameters.
        params: [alpha, beta, gamma]
        P(x,y) ∝ exp(-alpha * x^2 - beta * y^2 - gamma * x * y)
        """
        alpha, beta, gamma = params
        
        # Calculate log-probabilities
        log_p = -(alpha * x_flat**2 + beta * y_flat**2 + gamma * x_flat * y_flat)
        
        # Shift for numerical stability
        log_p -= np.max(log_p)
        
        p = np.exp(log_p)
        s = p.sum()
        if s < 1e-15:
            return None
        p /= s
        return p

    def score_from_params(params):
        """
        Objective function for optimization.
        Input: params (alpha, beta, gamma)
        Output: Negative score (for minimization)
        """
        p = get_P_from_params(params)
        if p is None:
            return 1e9
        
        score = evaluate_P(p)
        if not np.isfinite(score):
            return 1e9
        return -score

    # Step 1: Coarse Grid Search
    # Search ranges for alpha, beta, gamma
    # We want to explore different widths and correlations.
    # alpha, beta control the spread in x and y.
    # gamma controls the correlation (diagonal alignment).
    
    # Reasonable ranges:
    # alpha, beta in [0, 2]
    # gamma in [-1.5, 1.5] (note: for positive definite matrix, gamma^2 < 4*alpha*beta, 
    # but we just evaluate the resulting distribution, so we can sample widely and 
    # rely on the normalization to handle it, though very large negative values 
    # might cause underflow/overflow issues if not handled by log-space shift).
    
    # To ensure numerical stability, we should restrict gamma such that the quadratic form 
    # doesn't explode. However, the log-shift handles large negatives. 
    # Large positives in the exponent (if alpha/beta/gamma are negative) would cause overflow.
    # So we constrain alpha, beta >= 0.
    # If gamma is large positive, it penalizes x*y, pushing mass to corners.
    # If gamma is large negative, it rewards x*y, pushing mass to diagonal.
    
    alpha_vals = np.linspace(0.0, 2.0, 11)
    beta_vals = np.linspace(0.0, 2.0, 11)
    gamma_vals = np.linspace(-1.5, 1.5, 15)
    
    best_score = -np.inf
    best_params = None
    best_P = np.ones((n, n)) / N  # Fallback

    # Evaluate grid
    for a in alpha_vals:
        if time.time() > deadline - 30:
            break
        for b in beta_vals:
            for g in gamma_vals:
                params = np.array([a, b, g])
                p = get_P_from_params(params)
                if p is None:
                    continue
                score = evaluate_P(p)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_P = p.reshape((n, n))

    # Step 2: Local Refinement using Nelder-Mead
    # Take the best point from the grid and refine.
    # Also take a few nearby points to ensure we don't get stuck in a local flat area.
    
    if best_params is not None:
        # Define a small neighborhood around the best point for multi-start refinement
        start_points = [best_params]
        
        # Add some perturbed versions
        for _ in range(5):
            perturbation = np.random.normal(0, 0.2, 3)
            start_points.append(best_params + perturbation)
        
        # Also add some random points in the promising region
        for _ in range(5):
            a = np.random.uniform(0.5, 1.5)
            b = np.random.uniform(0.5, 1.5)
            g = np.random.uniform(-1.0, 1.0)
            start_points.append(np.array([a, b, g]))

        for sp in start_points:
            if time.time() > deadline - 10:
                break
            
            try:
                result = minimize(
                    score_from_params,
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if result.fun < 1e8:  # Valid score
                    current_score = -result.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_params = result.x
                        p = get_P_from_params(best_params)
                        if p is not None:
                            best_P = p.reshape((n, n))
            except Exception:
                continue

    # Final safety check
    if best_P is None or not np.all(np.isfinite(best_P)) or best_P.shape != (n, n):
        return np.ones((n, n)) / N
        
    # Ensure it sums to 1 and is valid
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        return np.ones((n, n)) / N
    best_P /= s
        
    return best_P

if __name__ == "__main__":
    pass
