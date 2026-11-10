"""Bermudan exercise-policy research: regression-based continuation value with smoothing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using regression on training paths.
    
    Parameters:
    -----------
    training_paths : ndarray of shape (N, 13) where columns are S_0, S_1, ..., S_12
    contract : dict with keys 'K', 'r', 'sigma', 'T', 'num_steps' (optional)
    
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean vector
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 120s
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = contract.get('num_steps', 12)
    
    # Ensure training_paths is 2D
    training_paths = np.asarray(training_paths, dtype=np.float64)
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)
    
    N, total_cols = training_paths.shape
    
    # Determine indexing
    if total_cols == num_steps + 1:
        s_ex = training_paths[:, 1:num_steps+1]  # shape (N, 12)
    elif total_cols == num_steps:
        s_ex = training_paths  # shape (N, 12)
    else:
        s_ex = training_paths[:, -num_steps:]
    
    # Time step in years
    dt = T / num_steps
    discount = np.exp(-r * dt)
    
    def make_features(s):
        """Create polynomial features up to degree 5."""
        s = np.asarray(s, dtype=np.float64)
        return np.column_stack([
            np.ones_like(s),
            s,
            s**2,
            s**3,
            s**4,
            s**5
        ])
    
    # Backward induction with regression
    # poly_coeffs[i] for step i+1 (0-indexed), i=0..11
    poly_coeffs = [None] * num_steps
    
    # Start from the last step (step 12, index 11)
    s_ex_last = s_ex[:, -1]  # S_12
    v_step12 = np.maximum(K - s_ex_last, 0.0)
    
    # For each step from 11 down to 1:
    for i in range(num_steps - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # i is the step index (1-based), so s_ex[:, i-1] is S_i
        s_i = s_ex[:, i-1]  # S_i
        
        # v_next is V_{i+1}(S_{i+1})
        if i == num_steps - 1:
            v_next = v_step12
        else:
            # v_next was computed at the end of the previous iteration
            pass
        
        # Estimate E[discount * v_next | S_i] by regressing on S_i
        target = discount * v_next
        X = make_features(s_i)
        
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except:
            coeffs = np.zeros(X.shape[1])
        
        poly_coeffs[i-1] = coeffs
        
        # Compute continuation value at step i
        cont_val = X @ coeffs
        
        # Intrinsic value at step i
        intrinsic = np.maximum(K - s_i, 0.0)
        
        # Value at step i
        v_current = np.maximum(intrinsic, cont_val)
        
        # For the next iteration, v_next should be v_current
        v_next = v_current
    
    # Check if we have valid coefficients
    if all(c is None for c in poly_coeffs):
        class SimplePolicy:
            def __init__(self, K, num_steps):
                self.K = K
                self.num_steps = num_steps
            
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=np.float64)
                t = int(time)
                if t < 1 or t > self.num_steps:
                    return np.zeros(len(current_state), dtype=bool)
                intrinsic = np.maximum(self.K - current_state, 0.0)
                return intrinsic > 0
        
        return SimplePolicy(K, num_steps)
    
    class BermudanPolicy:
        def __init__(self, K, r, sigma, T, num_steps, poly_coeffs):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.num_steps = num_steps
            self.poly_coeffs = poly_coeffs
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the Bermudan put.
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            
            if time is None:
                return np.zeros(len(current_state), dtype=bool)
            
            t = int(time)
            
            if t < 1 or t > self.num_steps:
                return np.zeros(len(current_state), dtype=bool)
            
            # At the last step (t=12), always exercise if intrinsic > 0
            if t == self.num_steps:
                intrinsic = np.maximum(self.K - current_state, 0.0)
                return intrinsic > 0
            
            # For steps 1-11, compare intrinsic to continuation value
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            # Get the polynomial coefficients for this step
            coeffs = self.poly_coeffs[t-1]
            
            if coeffs is None:
                return intrinsic > 0
            
            # Compute continuation value
            X = make_features(current_state)
            cont_val = X @ coeffs
            
            # Exercise if intrinsic >= continuation value
            return intrinsic >= cont_val
    
    return BermudanPolicy(K, r, sigma, T, num_steps, poly_coeffs)

# EVOLVE-BLOCK-END
