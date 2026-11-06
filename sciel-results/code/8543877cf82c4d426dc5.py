# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import linregress

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using backward induction with Local Linear Regression.
    
    This strategy approximates the continuation value function at each exercise date
    by fitting a local linear model to the in-the-money training paths. This avoids
    the bias of global polynomial fits and the discretization errors of simple grids,
    providing a more accurate estimate of the optimal exercise boundary.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Extract parameters from contract
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Training paths shape: (n_paths, n_cols)
    n_paths, n_cols = training_paths.shape
    
    # Determine exercise data columns
    # Task: "12 monthly exercise dates (no time-0 exercise)"
    # If n_cols > n_dates, assume first col is t0 (S0) and rest are exercise dates.
    # If n_cols == n_dates, assume all cols are exercise dates.
    if n_cols > n_dates:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]
        
    n_paths_eff, n_steps_eff = exercise_data.shape
    if n_steps_eff != n_dates:
        n_dates = n_steps_eff

    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    # Pre-compute log-spots for efficiency
    log_S = np.log(exercise_data)
    
    # Intrinsic payoff at each step
    intrinsic = np.maximum(K - exercise_data, 0.0)
    
    # We store the optimal exercise decision for each path at each step
    # This will be used to define the policy boundaries implicitly, 
    # but for the return object, we need a deterministic rule.
    # Since the state is 1D (Spot), the optimal policy is a threshold B_t for each t.
    # We can estimate B_t by finding the spot level where intrinsic >= continuation.
    
    # Backward induction
    # V_{n-1} = Intrinsic (at maturity, exercise if in the money)
    # Actually, at the last step, we exercise if intrinsic > 0.
    # V_t = max(Intrinsic_t, Discount * E[V_{t+1} | S_t])
    
    # We will estimate the continuation value function V_{t+1} using Local Linear Regression
    # on the set of paths that did NOT exercise at t+1 (or rather, the realized value V_{t+1}).
    
    # Initialize V at the last step
    V_next = intrinsic[:, -1].copy()
    
    # Store the estimated continuation value functions or the resulting thresholds
    # To implement the policy, we need to know for a given S_t, whether to exercise.
    # We can store the fitted local linear model parameters for each time step.
    # However, local linear regression is state-dependent.
    # A simpler and robust approach for 1D state: 
    # For each time step t, we fit a global linear model (or piecewise) to the continuation values 
    # of paths that are in-the-money or near the boundary?
    # Actually, standard LS uses a global basis. Local Linear is better for non-linearities.
    # But for the policy object, we need a fast evaluation.
    
    # Alternative: Estimate the optimal threshold B_t for each t directly.
    # If we assume the continuation value is monotonic decreasing in S (for puts),
    # the exercise region is S < B_t.
    # We can estimate B_t by finding the spot price where the estimated continuation value equals intrinsic.
    
    # Let's use a hybrid: 
    # 1. Compute optimal exercise decisions for training paths via backward induction.
    # 2. For each time step t, determine the boundary B_t by finding the maximum spot price 
    #    among paths that exercised, or by interpolation.
    
    # Actually, a cleaner "different strategy" is to fit the continuation value function 
    # using Local Linear Regression at each step, and then the policy is:
    # Exercise if Intrinsic(S) > Continuation(S).
    
    # To make the policy object fast, we can pre-compute the fitted LLR parameters for a grid of S,
    # or simply store the optimal decisions and infer the boundary.
    
    # Let's stick to the threshold approach but derive it from the LLR-based backward induction.
    
    # V_next is the value at t+1.
    # We need E[V_next | S_t].
    
    # We will store the fitted models for each time step to allow fast evaluation in the policy object.
    # Model: V(S) = alpha + beta * S (local linear)
    # Or simpler: Since the state is 1D, we can just store the estimated continuation value 
    # on a fine grid of S and interpolate.
    
    # Let's use a fine grid of S values for the continuation value estimation.
    # S range: [0.1*K, 2.0*K]
    s_min = 0.1 * K
    s_max = 2.0 * K
    n_grid = 100
    s_grid = np.linspace(s_min, s_max, n_grid)
    
    # Store continuation value estimates on the grid for each time step
    # cont_values[t] is an array of shape (n_grid,)
    cont_values = np.zeros((n_dates, n_grid))
    
    # At the last step, continuation is 0 (or intrinsic if we consider it as terminal value?)
    # V_{n-1} = max(Intrinsic, 0) = Intrinsic.
    # For t = n-2 down to 0:
    # V_t = max(Intrinsic_t, Discount * E[V_{t+1} | S_t])
    
    # We need to estimate E[V_{t+1} | S_t] for S_t in s_grid.
    # We use the training paths (S_{t+1}, V_{t+1}) to fit the model.
    
    # Initialize V for the last step
    V_curr = intrinsic[:, -1].copy()
    
    # We need to store the policy boundaries or the value function.
    # Let's store the estimated continuation value on the grid for each t.
    # cont_grid[t] corresponds to E[V_{t+1} | S_t = s_grid]
    
    # We'll do backward induction.
    # At t = n-1, V = Intrinsic.
    # At t = n-2, we need E[V_{n-1} | S_{n-2}].
    
    # We will compute the optimal exercise mask for training paths to update V.
    # But to define the policy, we just need the boundary.
    
    # Let's compute the continuation value on the grid for each t.
    # cont_grid[t] = Estimated E[V_{t+1} | S_t = s_grid]
    
    # For t = n-1, there is no continuation.
    # For t = n-2 down to 0:
    
    # We need V_{t+1} for the training paths.
    # Let's keep track of V for training paths as we go back.
    V_train = np.zeros((n_dates, n_paths_eff))
    V_train[-1] = intrinsic[:, -1]
    
    # We will store the fitted continuation values on the grid
    # cont_grid[t] is the estimated E[V_{t+1} | S_t] for s in s_grid
    cont_grid = np.zeros((n_dates, n_grid))
    
    # We also need to determine the exercise boundary B_t for the policy.
    # B_t is the largest S such that Intrinsic(S) >= Discount * Cont(S).
    
    boundaries = np.zeros(n_dates)
    
    for t in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            # If we run out of time, break and use what we have
            break
            
        S_t = exercise_data[:, t]
        V_next_train = V_train[:, t+1]
        
        # Fit Local Linear Regression to estimate E[V_next | S_t]
        # We want to evaluate this at s_grid.
        # For each s in s_grid, we fit a local linear model using weights based on distance from S_t to s.
        # This is O(n_grid * n_paths). 100 * 4096 = 400k operations per step. 12 steps -> 5M. Fast.
        
        # We can vectorize the LLR.
        # For each s_j in s_grid:
        # Weights w_i = exp(-(S_ti - s_j)^2 / (2 * h^2))
        # Fit y = a + b*x with weights w.
        # Prediction at s_j is a + b*s_j.
        
        # Bandwidth h. Let's use a fixed bandwidth based on the spread of S.
        # h = 0.1 * K? Or adaptive?
        # Let's use h = 0.1 * K for simplicity.
        h = 0.1 * K
        
        # Pre-compute S_t column vector
        S_t_col = S_t[:, np.newaxis] # (n_paths, 1)
        s_grid_row = s_grid[np.newaxis, :] # (1, n_grid)
        
        # Compute distances
        diffs = S_t_col - s_grid_row # (n_paths, n_grid)
        weights = np.exp(-0.5 * (diffs / h)**2) # (n_paths, n_grid)
        
        # Normalize weights to sum to 1 for each s_j
        w_sums = np.sum(weights, axis=0)
        w_sums[w_sums == 0] = 1.0
        weights_norm = weights / w_sums[np.newaxis, :]
        
        # Weighted Linear Regression: y ~ x
        # We want to solve for a, b such that sum w_i (y_i - (a + b x_i))^2 is minimized.
        # This is a weighted least squares problem.
        # X = [1, S_t]
        # We can use normal equations: (X^T W X)^{-1} X^T W y
        
        # X is (n_paths, 2)
        X = np.column_stack((np.ones(n_paths_eff), S_t))
        
        # For each s_j, we have a weight vector w_j.
        # We need to solve (X^T W_j X) [a, b]^T = X^T W_j y for each j.
        
        # This is a bit slow if done in a loop over n_grid.
        # Let's vectorize the normal equations.
        
        # X^T W_j X is a 2x2 matrix.
        # (X^T W_j X)_{00} = sum w_j_i
        # (X^T W_j X)_{01} = sum w_j_i x_i
        # (X^T W_j X)_{11} = sum w_j_i x_i^2
        
        # X^T W_j y
        # (X^T W_j y)_0 = sum w_j_i y_i
        # (X^T W_j y)_1 = sum w_j_i x_i y_i
        
        # Let's compute these sums for all j at once.
        # w_j_i is weights[i, j].
        # x_i is S_t[i]. y_i is V_next_train[i].
        
        # Sums:
        # sum_w = sum(weights, axis=0) -> already have w_sums
        # sum_wx = sum(weights * S_t[:, np.newaxis], axis=0)
        # sum_wxx = sum(weights * S_t[:, np.newaxis]**2, axis=0)
        # sum_wy = sum(weights * V_next_train[:, np.newaxis], axis=0)
        # sum_wxy = sum(weights * S_t[:, np.newaxis] * V_next_train[:, np.newaxis], axis=0)
        
        sum_wx = np.sum(weights * S_t[:, np.newaxis], axis=0)
        sum_wxx = np.sum(weights * S_t[:, np.newaxis]**2, axis=0)
        sum_wy = np.sum(weights * V_next_train[:, np.newaxis], axis=0)
        sum_wxy = np.sum(weights * S_t[:, np.newaxis] * V_next_train[:, np.newaxis], axis=0)
        
        # Normal equations:
        # [sum_w,   sum_wx] [a]   [sum_wy]
        # [sum_wx,  sum_wxx] [b] = [sum_wxy]
        
        # Determinant
        det = sum_w * sum_wxx - sum_wx**2
        # Avoid division by zero
        det[det == 0] = 1e-9
        
        # Cramer's rule or inverse
        # a = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
        # b = (sum_w * sum_wxy - sum_wx * sum_wy) / det
        
        a = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
        b = (sum_w * sum_wxy - sum_wx * sum_wy) / det
        
        # Continuation value at s_grid: a + b * s_grid
        cont_val_grid = a + b * s_grid
        
        # Store in cont_grid
        cont_grid[t] = cont_val_grid
        
        # Determine boundary B_t
        # Exercise if Intrinsic(S) >= Discount * Cont(S)
        # Intrinsic(S) = max(K - S, 0)
        # Cont(S) is linear: a + b*S
        # We need to find S such that K - S = Discount * (a + b*S) for S < K.
        # K - S = D*a + D*b*S
        # K - D*a = S + D*b*S = S(1 + D*b)
        # S = (K - D*a) / (1 + D*b)
        
        D = discount
        # Check if 1 + D*b is zero
        denom = 1 + D * b
        # b is likely negative for puts (value decreases as S increases)
        # So 1 + D*b should be positive.
        
        B_t = (K - D * a) / denom
        
        # Clip B_t to reasonable range
        if B_t < 0:
            B_t = 0
        if B_t > K:
            # If boundary > K, it means we exercise everywhere? 
            # Intrinsic is 0 for S > K. Cont is positive. So we shouldn't exercise.
            # This formula assumes S < K. If S > K, Intrinsic = 0.
            # We should check if 0 >= D * Cont(S) for S > K.
            # Since Cont(S) > 0, we don't exercise.
            # So B_t should be capped at K.
            B_t = K
            
        boundaries[t] = B_t
        
        # Update V_train for current step t
        # V_t = max(Intrinsic_t, D * Cont(S_t))
        # We need Cont(S_t) for the training paths.
        # We have the model a, b.
        # Cont_train = a + b * S_t
        # Note: a, b are scalars? No, they are arrays of length n_grid.
        # Wait, the LLR fit was done for each s_j in s_grid independently.
        # So a and b are vectors of length n_grid.
        # This means the "model" is actually a function defined on the grid, not a single global linear model.
        # This is correct for LLR.
        # To get Cont(S_t) for training paths, we need to evaluate the LLR at S_t.
        # But we already computed the weights for s_grid.
        # We can reuse the weights if S_t values are in s_grid? No.
        # We need to compute the LLR prediction for S_t values.
        
        # This is expensive to re-do.
        # Alternative: Interpolate cont_val_grid to S_t.
        # Since cont_val_grid is smooth, interpolation is accurate enough.
        
        cont_train = np.interp(S_t, s_grid, cont_val_grid)
        
        V_train[:, t] = np.maximum(intrinsic[:, t], D * cont_train)
        
    # Fill in the last boundary
    # At the last step, exercise if Intrinsic > 0, i.e., S < K.
    boundaries[-1] = K
    
    # If we broke out of the loop early, fill remaining boundaries with K (conservative)
    # Or use the last computed ones?
    # Let's assume we finished. If not, the uncomputed ones are 0.
    # Let's set any 0 boundaries (except if K is 0) to K to be safe?
    # Actually, if t loop didn't finish, boundaries[t] are 0.
    # Let's set boundaries where value is 0 to K (exercise if ITM)
    # But 0 is a valid boundary if K is small? No, K=100.
    # Let's check if we completed all steps.
    # If not, it's better to have a fallback.
    # Let's just use the computed ones. If some are 0, it might be wrong.
    # Let's re-initialize boundaries to K and only overwrite if computed.
    # Wait, I initialized boundaries to 0.
    # Let's reset boundaries to K and re-run? No, too slow.
    # Let's assume the loop completes. 12 steps is fast.
    
    # If any boundary is 0 and t < n_dates-1, it might be an error or true 0.
    # For a put, boundary is usually > 0.
    # Let's set boundaries that are 0 to K to be safe (exercise if ITM).
    boundaries[boundaries == 0] = K
    
    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            Returns: boolean vector
            """
            if time >= len(self.boundaries):
                # Fallback: exercise if in the money
                return (self.K - current_state) > 0
                
            boundary = self.boundaries[time]
            return current_state < boundary
            
    return ExercisePolicy(boundaries, K)

# EVOLVE-BLOCK-END
