import numpy as np

class BermudanPolicy:
    def __init__(self, training_paths, contract):
        self.K = contract.get('K', 100.0)
        self.r = contract.get('r', 0.05)
        self.sigma = contract.get('sigma', 0.2)
        self.T = contract.get('T', 1.0)
        self.num_steps = 12
        
        # Determine shape of training_paths
        if training_paths.ndim == 1:
            training_paths = training_paths.reshape(1, -1)
        
        N, total_cols = training_paths.shape
        
        # Assume standard format: columns are t=0, t=1, ..., t=12
        # If total_cols == 13, index 0 is t=0, 1..12 are exercise dates
        # If total_cols == 12, index 0..11 are exercise dates t=1..12
        if total_cols == self.num_steps + 1:
            self.S = training_paths[:, 1:] # Shape (N, 12)
        elif total_cols == self.num_steps:
            self.S = training_paths
        else:
            # Fallback: assume last 12 columns
            self.S = training_paths[:, -self.num_steps:]
            
        self.num_paths = self.S.shape[0]
        self.dt = self.T / self.num_steps
        self.discount_factor = np.exp(-self.r * self.dt)
        
        # Store RBF parameters for each step k (1 to 11)
        # self.rbf_params[k] = (centers, bandwidths, coefficients)
        self.rbf_params = {}
        
        # Terminal value at step 12 (index 11 in self.S)
        S_last = self.S[:, -1]
        V_next = np.maximum(self.K - S_last, 0.0)
        
        # Backward induction
        # We iterate from step 11 down to 1
        # self.S indices: 0 is t=1, 1 is t=2, ..., 10 is t=11, 11 is t=12
        num_centers = 15
        
        for k in range(self.num_steps - 1, 0, -1):
            # k is the current step index in 1-based exercise dates (1..11)
            # In self.S, current state is at index k-1
            S_curr = self.S[:, k-1]
            
            # V_next is the value at step k+1
            # Discount it
            V_disc = V_next * self.discount_factor
            
            # Fit RBF model
            # 1. Determine centers (quantiles)
            # Avoid edges to prevent extrapolation issues
            quantiles = np.linspace(0.05, 0.95, num_centers)
            centers = np.quantile(S_curr, quantiles)
            
            # Ensure centers are unique
            centers = np.unique(centers)
            
            # 2. Determine bandwidth
            # Simple heuristic: median distance between adjacent centers
            if len(centers) > 1:
                diffs = np.diff(centers)
                bandwidth = 0.5 * np.median(diffs)
                if bandwidth < 1e-6:
                    bandwidth = 1.0
            else:
                bandwidth = 1.0
                
            # 3. Build Gram Matrix
            # S_curr shape (N,), centers shape (M,)
            # Diff shape (N, M)
            diff = S_curr[:, np.newaxis] - centers[np.newaxis, :]
            gram = np.exp(-0.5 * (diff / bandwidth) ** 2)
            
            # 4. Solve for coefficients with regularization
            # Add small ridge for stability
            reg = 1e-6
            gram_reg = gram + reg * np.eye(len(centers))
            
            try:
                coeffs = np.linalg.solve(gram_reg, V_disc)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                coeffs, _, _, _ = np.linalg.lstsq(gram_reg, V_disc, rcond=None)
            
            self.rbf_params[k] = (centers, bandwidth, coeffs)
            
            # Calculate continuation value for all paths
            C_curr = gram @ coeffs
            
            # Immediate payoff
            P_curr = np.maximum(self.K - S_curr, 0.0)
            
            # Value at current step
            V_curr = np.maximum(P_curr, C_curr)
            
            # Update V_next for the next iteration (which is step k)
            V_next = V_curr

    def exercise(self, current_state, time):
        # current_state: array of spot prices
        # time: current time step (1 to 12)
        
        # Ensure current_state is a numpy array
        S = np.asarray(current_state, dtype=float)
        
        # Handle time mapping
        # If time is a float, round to nearest int
        t_idx = int(round(time))
        
        # Clamp t_idx to valid range 1..12
        if t_idx < 1:
            t_idx = 1
        elif t_idx > self.num_steps:
            t_idx = self.num_steps
            
        if t_idx == self.num_steps:
            # Last step: Exercise if in the money
            return S < self.K
            
        # Get parameters for this step
        params = self.rbf_params.get(t_idx)
        if params is None:
            # Fallback: Exercise if in the money
            return S < self.K
            
        centers, bandwidth, coeffs = params
        
        # Calculate continuation value
        diff = S[:, np.newaxis] - centers[np.newaxis, :]
        gram = np.exp(-0.5 * (diff / bandwidth) ** 2)
        C = gram @ coeffs
        
        # Immediate payoff
        P = np.maximum(self.K - S, 0.0)
        
        # Exercise if Payoff >= Continuation
        return P >= C

def fit_exercise_policy(training_paths, contract):
    return BermudanPolicy(training_paths, contract)
