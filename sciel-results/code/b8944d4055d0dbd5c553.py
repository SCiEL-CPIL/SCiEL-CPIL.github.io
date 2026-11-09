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
        The contract specifies: Spot 90/100/110, K=100, r=0.05, sigma=0.2, T=1.
        12 monthly exercise dates.
    
    contract : object or dict
        Contains contract parameters. We assume it has fields or can be accessed
        to get K, r, sigma, T, n_dates, etc. Based on the task description:
        K=100, r=0.05, sigma=0.2, T=1.0, 12 monthly dates.
    
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
    # We store the estimated continuation value for each path at each step
    V = np.zeros((n_paths, n_steps), dtype=np.float64)
    
    # At the last date, continuation value = payoff (must exercise or let expire, but since it's the last, V = payoff)
    V[:, -1] = payoffs[:, -1]
    
    # Basis functions for regression: polynomial in S
    # Use up to cubic: 1, S, S^2, S^3
    def basis_functions(s):
        """Compute basis functions for regression."""
        s = np.asarray(s)
        return np.column_stack([
            np.ones_like(s),
            s,
            s**2,
            s**3
        ])
    
    # Backward induction from last date to first
    for t in range(n_steps - 2, -1, -1):
        # Continue value at time t: discounted expected value at t+1
        # For each path, the "target" for regression is the discounted continuation value at t+1
        # But we need to estimate E[continuation | S_t]
        
        # Target: discounted V at next step
        target = discount * V[:, t + 1]
        
        # Features at current step
        X = basis_functions(S[:, t])
        
        # Least squares regression: target = X @ theta
        # Solve for theta
        try:
            theta, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback: use mean
            theta = np.array([np.mean(target), 0, 0, 0])
        
        # Estimated continuation value at time t
        V_cont = X @ theta
        
        # Exercise value: max(payoff, continuation)
        # But for the policy, we need to know whether to exercise
        # V[:, t] = max(payoff[:, t], V_cont)
        V[:, t] = np.maximum(payoffs[:, t], V_cont)
    
    # Now we have the full value function V
    # The exercise policy: exercise at time t if payoff > continuation value
    # We need to store the continuation values to make the decision
    
    # Re-compute continuation values for the policy
    # Store the regression coefficients for each time step
    thetas = []
    V_cont_store = np.zeros((n_paths, n_steps), dtype=np.float64)
    
    V[:, -1] = payoffs[:, -1]
    V_cont_store[:, -1] = payoffs[:, -1]
    
    for t in range(n_steps - 2, -1, -1):
        target = discount * V[:, t + 1]
        X = basis_functions(S[:, t])
        try:
            theta, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            theta = np.array([np.mean(target), 0, 0, 0])
        thetas.append(theta)
        V_cont = X @ theta
        V_cont_store[:, t] = V_cont
        V[:, t] = np.maximum(payoffs[:, t], V_cont)
    
    # thetas is in reverse order (from t=n_steps-2 down to t=0)
    # Reverse to get thetas[0] for t=0, thetas[1] for t=1, etc.
    thetas = thetas[::-1]
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thetas, K, n_dates):
            self.thetas = thetas
            self.K = K
            self.n_dates = n_dates
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            ----------
            current_state : array-like, shape (n_eval_paths,)
                Current spot prices.
            time : float or int
                The current time index. If float, we assume it's in [0, T].
                We need to map to the exercise date index.
                
            Returns:
            --------
            Boolean array of shape (n_eval_paths,)
            """
            s = np.asarray(current_state, dtype=np.float64)
            if s.ndim == 0:
                s = s.reshape(1)
                
            # Determine which exercise date this corresponds to
            # time is likely an index from 0 to n_dates-1, or a fraction of T
            # Based on the task: "12 monthly exercise dates (no time-0 exercise)"
            # So exercise dates are at t=1/12, 2/12, ..., 12/12
            # The 'time' parameter could be the date index (1-12) or the actual time
            # Let's assume time is the date index (1-based or 0-based)
            
            # Handle time as index
            if isinstance(time, (int, np.integer)):
                # If time is 1-based (1 to 12), convert to 0-based
                if time >= 1 and time <= self.n_dates:
                    t_idx = time - 1
                elif time >= 0 and time < self.n_dates:
                    t_idx = time
                else:
                    t_idx = 0
            else:
                # If time is a float (actual time in years), map to index
                # Exercise dates are at k/12 for k=1,...,12
                # Find the closest exercise date index
                t_idx = int(round(time * 12)) - 1
                t_idx = max(0, min(self.n_dates - 1, t_idx))
            
            if t_idx >= len(self.thetas):
                return np.zeros(len(s), dtype=bool)
            
            theta = self.thetas[t_idx]
            X = basis_functions(s)
            V_cont = X @ theta
            payoff = np.maximum(self.K - s, 0.0)
            
            return payoff > V_cont
    
    return ExercisePolicy(thetas, K, n_dates)
# EVOLVE-BLOCK-END
