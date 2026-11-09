import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    Strategy:
    1. Generate diverse structured initializations (diagonals, corners, uniforms).
    2. Perform local refinement using L-BFGS-B on log-probabilities from each start.
    3. Return the best valid result found within the time budget.
    This avoids the overhead and potential stagnation of global stochastic search
    by leveraging the smoothness of the entropy landscape in log-space.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 610  # 20s safety buffer

    # Pre-compute indices for entropy calculation
    # P[y, x] -> flat index y*n + x
    ys = np.repeat(np.arange(n), n)
    xs = np.tile(np.arange(n), n)
    
    # Indices for binning
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)       # x - y + 9 -> 0..18
    idx_x   = xs.astype(np.int32)                      # x -> 0..9
    idx_d1  = (xs + ys).astype(np.int32)               # x + y -> 0..18
    idx_d2  = (xs + 2 * ys).astype(np.int32)           # x + 2y -> 0..27
    
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

    def score_from_log_probs(log_p):
        """
        Objective function for optimization.
        Input: log_p (unnormalized log-probabilities)
        Output: Negative score (for minimization)
        """
        # Softmax to get probabilities
        log_p_shifted = log_p - np.max(log_p)
        p = np.exp(log_p_shifted)
        s = p.sum()
        if s < 1e-15:
            return 1e9 # Large penalty
        p = p / s
        
        score = evaluate_P(p)
        if not np.isfinite(score):
            return 1e9
        return -score

    # Generate diverse initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.zeros(N))
    
    # 2. Diagonal (x == y)
    diag = np.zeros(N)
    for i in range(n):
        diag[i*n + i] = 1.0
    initializations.append(np.log(diag + 1e-6))
    
    # 3. Anti-diagonal (x + y == n-1)
    anti_diag = np.zeros(N)
    for i in range(n):
        anti_diag[i*n + (n-1-i)] = 1.0
    initializations.append(np.log(anti_diag + 1e-6))
    
    # 4. Corners
    corners = np.zeros(N)
    corners[0] = 1.0
    corners[n-1] = 1.0
    corners[(n-1)*n] = 1.0
    corners[N-1] = 1.0
    initializations.append(np.log(corners + 1e-6))
    
    # 5. Random sparse patterns
    for _ in range(10):
        p = np.random.rand(N)
        p = np.where(p > 0.5, 1.0, 0.0)
        if p.sum() == 0:
            p = np.ones(N) / N
        initializations.append(np.log(p + 1e-6))
        
    # 6. Structured grids (e.g., every k-th point)
    for k in [2, 3, 5]:
        grid = np.zeros(N)
        for i in range(0, n, k):
            for j in range(0, n, k):
                grid[i*n + j] = 1.0
        if grid.sum() > 0:
            initializations.append(np.log(grid + 1e-6))

    best_log_p = None
    best_score = -np.inf
    best_P = np.ones((n, n)) / N # Fallback

    # Optimize from each start
    for i, init_log_p in enumerate(initializations):
        if time.time() > deadline - 10:
            break
            
        # Add small noise to break symmetry for identical inits
        init_log_p = init_log_p + np.random.normal(0, 0.01, N)
        
        try:
            result = minimize(
                score_from_log_probs, 
                init_log_p, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.fun < 1e8: # Valid score
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_log_p = result.x
                    
                    # Update best_P
                    log_p_shifted = best_log_p - np.max(best_log_p)
                    P_temp = np.exp(log_p_shifted)
                    P_temp /= P_temp.sum()
                    best_P = P_temp.reshape((n, n))
                    
        except Exception:
            continue

    # If optimization failed completely, return uniform
    if best_log_p is None:
        return np.ones((n, n)) / N

    # Final safety check
    if not np.all(np.isfinite(best_P)) or best_P.shape != (n, n):
        return np.ones((n, n)) / N
        
    return best_P

if __name__ == "__main__":
    pass
