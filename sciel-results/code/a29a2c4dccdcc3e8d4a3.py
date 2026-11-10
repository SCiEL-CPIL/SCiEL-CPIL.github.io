import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using Longstaff-Schwartz Monte Carlo
    with a piecewise linear basis for the continuation value.
    """
    start_wall = time.time()
    deadline = 140.0  # Safety margin before the 150s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    paths = np.asarray(training_paths)
    n_paths, n_steps = paths.shape
    
    # Determine the indices for the 12 exercise dates.
    if n_steps == 12:
        ex_indices = np.arange(12)
    elif n_steps == 13:
        ex_indices = np.arange(1, 13)
    else:
        # Fallback: assume the last 12 columns are the exercise dates
        ex_indices = np.arange(n_steps - 12, n_steps)

    n_ex = len(ex_indices)
    
    # Discount factor per step (monthly)
    dt = T / n_ex
    disc = math.exp(-r * dt)
    
    # Initialize the value at the last exercise date (terminal)
    S_last = paths[:, ex_indices[-1]]
    V = np.maximum(K - S_last, 0.0)
    
    # We will store the regression coefficients and basis nodes for each step
    # coeffs_list[k] for exercise date k (0-indexed in ex_indices)
    coeffs_list = [None] * n_ex 
    nodes_list = [None] * n_ex
    coeffs_list[-1] = None # Terminal doesn't need regression
    nodes_list[-1] = None
    
    # Number of basis functions for the piecewise linear approximation
    n_basis = 100
    
    # Backward induction
    # k goes from n_ex - 2 down to 0
    for k in range(n_ex - 2, -1, -1):
        S_curr = paths[:, ex_indices[k]]
        
        # Continuation value is discounted next value
        V_cont = disc * V
        
        # Determine the range of spot prices for this step
        s_min = np.min(S_curr)
        s_max = np.max(S_curr)
        
        # Add a small buffer to the range to ensure all points are covered
        buffer = 0.05 * (s_max - s_min) if s_max > s_min else 1.0
        s_min -= buffer
        s_max += buffer
        
        if s_min <= 0:
            s_min = 1e-6
            
        # Create a grid of nodes for the piecewise linear basis
        # We use n_basis + 1 nodes to create n_basis intervals
        nodes = np.linspace(s_min, s_max, n_basis + 1)
        
        # Construct the design matrix X for the piecewise linear basis
        # Basis functions are "hat functions" or simple linear segments.
        # A simpler and robust approach for LS is to use the standard linear spline basis:
        # phi_0(S) = 1 - (S - nodes[0])/(nodes[1] - nodes[0]) for S < nodes[1], 0 otherwise
        # phi_j(S) = (S - nodes[j-1])/(nodes[j] - nodes[j-1]) - (S - nodes[j])/(nodes[j+1] - nodes[j]) for nodes[j-1] < S < nodes[j+1]
        # ...
        # However, a very common and efficient basis for this specific problem is just
        # using the spots themselves and their absolute deviations from key points, or
        # simply using a dense set of linear basis functions.
        
        # Let's use a standard linear spline basis.
        # X will have shape (n_paths, n_basis)
        X = np.zeros((n_paths, n_basis))
        
        for j in range(n_basis):
            if j == 0:
                # Left-most basis function
                # phi_0(S) = max(0, (nodes[1] - S) / (nodes[1] - nodes[0]))
                X[:, j] = np.maximum(0, (nodes[1] - S_curr) / (nodes[1] - nodes[0]))
            elif j == n_basis - 1:
                # Right-most basis function
                # phi_last(S) = max(0, (S - nodes[-2]) / (nodes[-1] - nodes[-2]))
                X[:, j] = np.maximum(0, (S_curr - nodes[-2]) / (nodes[-1] - nodes[-2]))
            else:
                # Middle basis functions (Hat functions)
                # phi_j(S) = max(0, 1 - |S - nodes[j]| / (nodes[j+1] - nodes[j-1]) / 2 ... wait, simpler:
                # It's easier to define them as the standard linear B-splines of order 1.
                # phi_j(S) is 0 outside [nodes[j-1], nodes[j+1]]
                # It peaks at 1 at nodes[j].
                # Slope up: 1 / (nodes[j] - nodes[j-1])
                # Slope down: -1 / (nodes[j+1] - nodes[j])
                
                # Since nodes are equally spaced, let h = nodes[1] - nodes[0]
                h = nodes[1] - nodes[0]
                # phi_j(S) = max(0, (S - nodes[j-1])/h) - max(0, (S - nodes[j])/h) ... No, that's not right.
                
                # Let's use the explicit hat function formula:
                # If S < nodes[j-1] or S > nodes[j+1], phi_j(S) = 0
                # If nodes[j-1] <= S <= nodes[j], phi_j(S) = (S - nodes[j-1]) / (nodes[j] - nodes[j-1])
                # If nodes[j] <= S <= nodes[j+1], phi_j(S) = (nodes[j+1] - S) / (nodes[j+1] - nodes[j])
                
                # Vectorized implementation:
                left_part = np.maximum(0, (S_curr - nodes[j-1]) / (nodes[j] - nodes[j-1]))
                right_part = np.maximum(0, (nodes[j+1] - S_curr) / (nodes[j+1] - nodes[j]))
                
                # Actually, the standard hat function is:
                # phi_j(S) = (S - nodes[j-1])/h for S in [nodes[j-1], nodes[j]]
                # phi_j(S) = (nodes[j+1] - S)/h for S in [nodes[j], nodes[j+1]]
                # phi_j(S) = 0 otherwise
                
                # My previous left_part/right_part logic was slightly off for the peak.
                # Let's just compute it directly.
                mask_left = (S_curr >= nodes[j-1]) & (S_curr <= nodes[j])
                mask_right = (S_curr >= nodes[j]) & (S_curr <= nodes[j+1])
                
                val = np.zeros_like(S_curr)
                val[mask_left] = (S_curr[mask_left] - nodes[j-1]) / h
                val[mask_right] = (nodes[j+1] - S_curr[mask_right]) / h
                X[:, j] = val
                
        # Solve least squares: X @ c = V_cont
        try:
            c, residuals, rank, s = np.linalg.lstsq(X, V_cont, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to simple mean if lstsq fails
            c = np.zeros(n_basis)
            c[0] = np.mean(V_cont) # Assign to first basis? Better to just use a constant approximation
            # Actually, if it fails, let's just use the previous coefficients or a simple heuristic
            if k + 1 < n_ex and coeffs_list[k+1] is not None:
                c = coeffs_list[k+1]
            else:
                c = np.zeros(n_basis)
                
        coeffs_list[k] = c
        nodes_list[k] = nodes
        
        # Calculate regression value for all paths at this step
        V_reg = X @ c
        
        # Exercise decision: Exercise if intrinsic > continuation (regression)
        intrinsic = np.maximum(K - S_curr, 0.0)
        exercise_mask = intrinsic > V_reg
        
        # Update V for next iteration (going backwards)
        # V is the value at step k, which is max(intrinsic, V_reg)
        V = np.where(exercise_mask, intrinsic, V_reg)
        
        # Deadline guard
        if time.time() > deadline:
            break
            
    # Fill remaining coeffs if deadline hit early
    for k in range(n_ex - 1, -1, -1):
        if coeffs_list[k] is None and k < n_ex - 1:
            # Fallback: use the next available coeffs or simple heuristic
            if k + 1 < n_ex and coeffs_list[k+1] is not None:
                coeffs_list[k] = coeffs_list[k+1]
                nodes_list[k] = nodes_list[k+1]
            else:
                # Simple heuristic: use a single constant basis or linear
                # Let's just create a dummy basis and coefficients
                coeffs_list[k] = np.array([np.mean(V)])
                nodes_list[k] = np.array([0.0, 1.0]) # Dummy nodes

    class ExercisePolicy:
        def __init__(self, coeffs, nodes, n_ex, K_val):
            self.coeffs = coeffs
            self.nodes = nodes
            self.n_ex = n_ex
            self.K = K_val
            
        def exercise(self, current_state, time_val):
            # time_val is in [0, T]
            # Map time_val to step index
            # Exercise dates: 1/12, 2/12, ..., 12/12
            step_float = time_val * self.n_ex
            step_idx = int(round(step_float)) - 1 # 0-indexed
            
            if step_idx < 0:
                return np.zeros_like(current_state, dtype=bool)
            if step_idx >= self.n_ex:
                step_idx = self.n_ex - 1
                
            c = self.coeffs[step_idx]
            nodes = self.nodes[step_idx]
            
            if c is None or nodes is None:
                return np.zeros_like(current_state, dtype=bool)
                
            # Evaluate the piecewise linear basis for the current states
            n_paths_eval = len(current_state)
            n_basis = len(c)
            X_eval = np.zeros((n_paths_eval, n_basis))
            
            # Check if it's the fallback case (single coefficient)
            if len(nodes) == 2 and n_basis == 1:
                # Constant approximation
                V_reg = np.full(n_paths_eval, c[0])
            else:
                h = nodes[1] - nodes[0]
                for j in range(n_basis):
                    if j == 0:
                        X_eval[:, j] = np.maximum(0, (nodes[1] - current_state) / (nodes[1] - nodes[0]))
                    elif j == n_basis - 1:
                        X_eval[:, j] = np.maximum(0, (current_state - nodes[-2]) / (nodes[-1] - nodes[-2]))
                    else:
                        mask_left = (current_state >= nodes[j-1]) & (current_state <= nodes[j])
                        mask_right = (current_state >= nodes[j]) & (current_state <= nodes[j+1])
                        val = np.zeros_like(current_state)
                        val[mask_left] = (current_state[mask_left] - nodes[j-1]) / h
                        val[mask_right] = (nodes[j+1] - current_state[mask_right]) / h
                        X_eval[:, j] = val
                V_reg = X_eval @ c
                
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            return intrinsic > V_reg

    return ExercisePolicy(coeffs_list, nodes_list, n_ex, K)
