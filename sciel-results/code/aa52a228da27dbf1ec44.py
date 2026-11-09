"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.stats as stats

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz Monte Carlo regression.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_dates)
        Simulated spot price paths. n_dates = 12 (monthly, no time-0).
    
    contract : object or dict
        Contains contract parameters.
    
    Returns:
    --------
    An object with an exercise(current_state, time) method returning a boolean vector.
    """
    # Extract parameters from contract or use defaults from task description
    K = getattr(contract, 'K', 100.0)
    r = getattr(contract, 'r', 0.05)
    sigma = getattr(contract, 'sigma', 0.2)
    T = getattr(contract, 'T', 1.0)
    n_dates = getattr(contract, 'n_dates', 12)
    
    # Ensure training_paths is a numpy array
    S = np.asarray(training_paths, dtype=np.float64)
    if S.ndim == 1:
        S = S.reshape(1, -1)
    
    n_paths, n_steps = S.shape
    
    # Discount factors for each step
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Payoff at each date: put payoff = max(K - S, 0)
    payoffs = np.maximum(K - S, 0.0)
    
    # Longstaff-Schwartz backward induction
    # V[i] = continuation value at step i, estimated via regression
    V = np.zeros((n_paths, n_steps), dtype=np.float64)
    
    # At the last date, continuation value = payoff
    V[:, -1] = payoffs[:, -1]
    
    # Basis functions: Use Chebyshev polynomials for better numerical stability
    # Map S to [-1, 1] range for Chebyshev?
    # S is around 90-110. Let's normalize S to [0, 1] or similar for stability.
    # Or just use a fixed range based on expected GBM behavior.
    # With sigma=0.2, T=1, S0=100. S is likely in [50, 150].
    # Let's use a simple normalization: s_norm = (S - 50) / 100
    # Or just use raw S with higher order polynomials if condition number is okay.
    # Given 4096 paths, raw polynomials up to order 4 or 5 are usually fine.
    # Let's try order 4: 1, S, S^2, S^3, S^4
    
    n_basis = 10
    def basis_functions(s):
        """Compute Chebyshev basis functions for regression."""
        s = np.asarray(s)
        # Normalize s to [-1, 1] range for Chebyshev stability
        # S is roughly in [50, 150] for these parameters
        s_norm = (s - 100.0) / 50.0
        # Clip to [-1, 1] to avoid extrapolation issues if S goes out of range
        s_norm = np.clip(s_norm, -1.0, 1.0)
        
        # Compute Chebyshev polynomials T_k(x)
        # T_0(x) = 1
        # T_1(x) = x
        # T_k(x) = 2*x*T_{k-1}(x) - T_{k-2}(x)
        cols = [np.ones_like(s_norm)]
        if n_basis > 1:
            cols.append(s_norm.copy())
            for k in range(2, n_basis):
                t_k = 2.0 * s_norm * cols[-1] - cols[-2]
                cols.append(t_k)
        return np.column_stack(cols)
    
    # Store thetas for each time step
    # thetas[t] will be the coefficients for step t
    thetas = [None] * n_steps
    
    # Backward induction
    for t in range(n_steps - 2, -1, -1):
        # Target: discounted V at next step
        target = discount * V[:, t + 1]
        
        # Features at current step
        X = basis_functions(S[:, t])
        
        # Regularized least squares (Ridge Regression)
        # Minimize ||X @ theta - target||^2 + lambda * ||theta||^2
        lambda_reg = 1e-3
        X_t_X = X.T @ X
        X_t_y = X.T @ target
        # Add regularization
        X_t_X_reg = X_t_X + lambda_reg * np.eye(n_basis)
        
        try:
            theta = np.linalg.solve(X_t_X_reg, X_t_y)
        except np.linalg.LinAlgError:
            # Fallback: use mean
            theta = np.zeros(n_basis)
            theta[0] = np.mean(target)
        
        thetas[t] = theta
        
        # Estimated continuation value at time t
        V_cont = X @ theta
        
        # Value at time t
        V[:, t] = np.maximum(payoffs[:, t], V_cont)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thetas, K, n_dates):
            self.thetas = thetas
            self.K = K
            self.n_dates = n_dates
            self.n_basis = n_basis
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            """
            s = np.asarray(current_state, dtype=np.float64)
            if s.ndim == 0:
                s = s.reshape(1)
                
            # Determine which exercise date this corresponds to
            # time is likely an index or float time.
            # Task says "12 monthly exercise dates".
            # If time is int 1..12, map to 0..11.
            # If time is float 0..1, map to index.
            
            if isinstance(time, (int, np.integer)):
                if 1 <= time <= self.n_dates:
                    t_idx = time - 1
                elif 0 <= time < self.n_dates:
                    t_idx = time
                else:
                    t_idx = 0
            else:
                # Assume time is in years [0, T]
                # Exercise dates at k/12
                t_idx = int(round(time * 12)) - 1
                t_idx = max(0, min(self.n_dates - 1, t_idx))
            
            if t_idx >= len(self.thetas) or self.thetas[t_idx] is None:
                return np.zeros(len(s), dtype=bool)
            
            theta = self.thetas[t_idx]
            X = basis_functions(s)
            V_cont = X @ theta
            payoff = np.maximum(self.K - s, 0.0)
            
            # Exercise if payoff is greater than or equal to continuation value
            # Using >= handles the tie case where holding offers no benefit over exercising
            return payoff >= V_cont
    
    return ExercisePolicy(thetas, K, n_dates)
# EVOLVE-BLOCK-END
