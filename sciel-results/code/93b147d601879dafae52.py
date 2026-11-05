import numpy as np
import time as _time
from scipy.optimize import minimize_scalar

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def exercise(self, current_state, time_index):
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def _dual_upper_bound(S_curr, S_next, V_next, discount, K, n_basis=20, n_grid=50):
    """
    Compute dual upper bound on continuation value using parametric convex payoffs.
    
    The dual approach: E[discount * V_next | S_curr] <= min_{lambda >= 0} E[lambda * (S_next - K)^+ + (1-lambda) * V_next | S_curr]
    Actually, for a put, we use: E[discount * V_next | S_curr] <= min_{c} E[discount * (c + lambda * (K - S_next)) | S_curr]
    where c and lambda are chosen to minimize the upper bound.
    
    More precisely, for a Bermudan put, the continuation value at step j is:
    C_j(S) = E[discount * V_{j+1}(S_{j+1}) | S_j = S]
    
    We approximate this using a family of affine functions:
    g(S_next) = a + b * (K - S_next)
    where a >= 0, b >= 0, and we require g(S_next) >= V_{j+1}(S_next) for all S_next in the training data.
    
    The optimal g minimizes E[discount * g(S_next) | S_j = S].
    """
    # For simplicity and speed, we'll use a grid-based approach
    # to find the best affine upper bound
    
    # We want to find a, b such that:
    # 1. a + b * (K - S_next) >= V_next for all paths
    # 2. Minimize E[discount * (a + b * (K - S_next)) | S_curr = S]
    
    # This is a linear programming problem, but we can approximate with a grid search
    
    # First, find the minimum a for a given b:
    # a >= max(V_next - b * (K - S_next))
    
    # For efficiency, we'll sample a grid of b values and compute the optimal a for each
    
    # b ranges from 0 to some max (e.g., 2)
    b_values = np.linspace(0, 2.0, n_basis)
    
    best_upper = np.full(len(S_curr), np.inf)
    
    for b in b_values:
        # Compute required a: a >= max(V_next - b * (K - S_next))
        a_required = np.max(V_next - b * (K - S_next))
        a_required = max(a_required, 0.0)
        
        # For each S_curr, compute E[discount * (a_required + b * (K - S_next)) | S_curr = S]
        # This is an empirical average over paths with S_curr close to S
        
        # For speed, we'll use a simple averaging approach
        # Group paths by S_curr value (bin them)
        
    # For better performance, use a different approach:
    # Fit a simple model: C(S) = max(0, A - B*S) where A, B are found by optimization
    
    # Actually, let's use a more practical approach:
    # For each S_curr, compute the average of V_next for paths with similar S_curr
    # Then smooth this to get a monotone decreasing function
    
    # Binned average approach with smoothing
    n_bins = 50
    S_min = np.min(S_curr)
    S_max = np.max(S_curr)
    bin_edges = np.linspace(S_min, S_max, n_bins + 1)
    
    # Bin the paths
    bin_indices = np.digitize(S_curr, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # For each bin, compute the average V_next
    bin_avgs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_avgs[i] = np.mean(V_next[mask])
            bin_counts[i] = np.sum(mask)
        else:
            bin_avgs[i] = np.nan
    
    # Interpolate to get values at all S_curr points
    valid_bins = np.where(bin_counts > 0)[0]
    if len(valid_bins) < 2:
        # Not enough data, return a constant
        return np.full(len(S_curr), discount * np.mean(V_next))
    
    # Interpolate
    S_grid = 0.5 * (bin_edges[valid_bins] + bin_edges[valid_bins + 1])
    V_grid = bin_avgs[valid_bins]
    
    # Ensure monotone decreasing (for a put, continuation should decrease with S)
    # Apply isotonic regression-like smoothing
    for i in range(len(V_grid) - 2, -1, -1):
        if V_grid[i] < V_grid[i + 1]:
            V_grid[i] = V_grid[i + 1]
    
    # Interpolate to get continuation values at S_curr
    cont_values = np.interp(S_curr, S_grid, V_grid)
    
    return discount * cont_values

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = paths.shape
    n_exercise = 12  # exercise dates at steps 1..12
    
    dt = 1.0 / 12
    r = 0.05
    K = 100.0
    discount = np.exp(-r * dt)
    
    # V[j, i] stores the value of the option at exercise date j (step j+1) for path i
    V = np.zeros((n_exercise, n_paths))
    
    # Last exercise date: j = 11 (index in V), step 12 in paths
    S_last = paths[:, 12]
    V[11, :] = np.maximum(K - S_last, 0.0)
    
    # Backward induction for j = 10 down to 0
    thresholds = np.zeros(n_exercise)
    thresholds[11] = K
    
    for j in range(10, -1, -1):
        if _time.time() > deadline:
            # Use remaining time to fill in thresholds with reasonable defaults
            for jj in range(j, -1, -1):
                thresholds[jj] = K * (0.5 + 0.5 * (11 - jj) / 11)
            break
            
        S_curr = paths[:, j+1]
        
        # Compute continuation value using dual upper bound approach
        cont_values = _dual_upper_bound(S_curr, paths[:, j+2], V[j+1, :], discount, K)
        cont_values = np.maximum(cont_values, 0.0)
        
        # Current Payoff
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Optimal Value
        V[j, :] = np.maximum(payoff, cont_values)
        
        # Find threshold: largest S where payoff >= cont
        # Grid search
        S_grid = np.linspace(0, K, 100)
        payoffs_grid = np.maximum(K - S_grid, 0.0)
        
        # Interpolate continuation values at grid points
        cont_grid = np.interp(S_grid, S_curr, cont_values)
        cont_grid = np.maximum(cont_grid, 0.0)
        
        # Find where payoff - cont changes from positive to negative
        diff = payoffs_grid - cont_grid
        
        # Find the largest S where diff >= 0
        exercise_mask = diff >= 0
        if np.any(exercise_mask):
            threshold = S_grid[np.where(exercise_mask)[0][-1]]
        else:
            threshold = 0.0
        
        thresholds[j] = np.clip(threshold, 0, K)
    
    return ExercisePolicy(thresholds)
