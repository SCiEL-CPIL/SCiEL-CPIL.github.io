# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using global parametric regression (quadratic in log-price)
    for continuation values, derived from training paths.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Extract parameters
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Handle input shape
    n_paths, n_cols = training_paths.shape
    if n_cols == n_dates:
        exercise_data = training_paths
    elif n_cols == n_dates + 1:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]
        
    # Precompute log-spots for regression features
    # X features: [1, ln(S), (ln(S))^2]
    # We will fit V_{t+1}(S_{t+1}) as a function of S_t? 
    # No, standard approach: 
    # 1. Compute terminal values V_{n-1}(S_{n-1}) = Payoff(S_{n-1})
    # 2. For t = n-2 down to 0:
    #    a. Interpolate V_{t+1} to get values on the next step's states? 
    #       Actually, we have the paths. 
    #    b. We want to estimate E[ V_{t+1}(S_{t+1}) | S_t ].
    #    c. We can regress V_{t+1}(S_{t+1}) on features of S_t.
    #       Since S_{t+1} is correlated with S_t, and V is smooth, 
    #       V_{t+1}(S_{t+1}) can be approximated by a function of S_t.
    #       A quadratic in ln(S_t) is a good basis for GBM options.
    
    # Let's define the basis function
    def get_features(S):
        log_S = np.log(S)
        return np.column_stack([np.ones_like(log_S), log_S, log_S**2])
    
    # Initialize V for the last date (index n_dates - 1)
    # V is the value at time t given S_t.
    # At the last date, V = Payoff.
    # We don't need to store V as a grid, but as a function.
    # However, for the regression at t-1, we need the values V_t(S_t) for the paths.
    
    # Let's store the coefficients for the regression of V_t on features of S_t.
    # Coeffs[t] are [c0, c1, c2] such that V_t(S) ≈ c0 + c1*ln(S) + c2*(ln(S))^2
    # Wait, this is an approximation of the *value function*.
    # Is it better to regress the *continuation value* or the *total value*?
    # Total Value V_t(S) = max(Intrinsic, Continuation).
    # If we regress V_{t+1}(S_{t+1}) on features of S_t, we are estimating the conditional expectation directly.
    # Let's do that.
    
    # V_coeffs[t] represents the function V_t(S).
    # For t = n_dates - 1, V is simply the payoff. It's not a quadratic, but we can represent it.
    # However, for the regression step, we need the *values* of V_{t+1} at the specific path states S_{t+1}.
    # We can just keep the values V_vals[t] for the training paths at time t.
    
    V_vals = [None] * n_dates
    S_vals = [exercise_data[:, t] for t in range(n_dates)]
    
    # Last date
    V_vals[-1] = np.maximum(K - S_vals[-1], 0.0)
    
    # We need to determine the exercise decision at each t.
    # Exercise if Intrinsic(S_t) >= E[ V_{t+1}(S_{t+1}) | S_t ] * discount
    # Let's estimate the continuation value function C_t(S_t) by regressing V_{t+1}(S_{t+1}) on features of S_t.
    
    # Store the coefficients for the continuation value function C_t(S)
    # C_t(S) ≈ d0 + d1*ln(S) + d2*(ln(S))^2
    cont_coeffs = [None] * n_dates
    
    for t in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_t = S_vals[t]
        V_next_vals = V_vals[t+1]
        
        # Features for S_t
        X = get_features(S_t)
        
        # Regress V_next_vals on X
        # V_next_vals = X @ coeff + error
        # We want to minimize sum of squares.
        # Use np.linalg.lstsq
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, V_next_vals, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to simple mean if regression fails
            coeffs = np.array([np.mean(V_next_vals), 0.0, 0.0])
            
        cont_coeffs[t] = coeffs
        
        # Calculate the estimated continuation value for each path at time t
        # Note: This is E[V_{t+1}(S_{t+1}) | S_t]
        cont_vals_paths = X @ coeffs
        
        # Discount
        dt = T / n_dates
        discount = np.exp(-r * dt)
        cont_vals_discounted = cont_vals_paths * discount
        
        # Intrinsic value
        intrinsic_vals = np.maximum(K - S_t, 0.0)
        
        # Total Value
        V_vals[t] = np.maximum(intrinsic_vals, cont_vals_discounted)
        
    # Now we have V_vals for all t, and cont_coeffs for t=0..n-2.
    # We need to construct the policy object.
    # The policy exercise(S, t) should return True if Intrinsic(S) >= Continuation(S).
    # Continuation(S) is given by the regression model: C_t(S) = d0 + d1*ln(S) + d2*(ln(S))^2.
    # Then discount it.
    # Compare with Intrinsic(S).
    
    class ExercisePolicy:
        def __init__(self, K, cont_coeffs, n_dates, r, T):
            self.K = K
            self.cont_coeffs = cont_coeffs
            self.n_dates = n_dates
            self.r = r
            self.T = T
            self.dt = T / n_dates
            self.discount = np.exp(-self.r * self.dt)
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            """
            # If time is the last date, always exercise if intrinsic > 0
            if time >= self.n_dates - 1:
                return (self.K - current_state) > 0
                
            if time < 0 or time >= len(self.cont_coeffs) or self.cont_coeffs[time] is None:
                # Fallback
                return (self.K - current_state) > 0
                
            coeffs = self.cont_coeffs[time]
            # Calculate continuation value for current states
            log_S = np.log(current_state)
            # C(S) = c0 + c1*ln(S) + c2*(ln(S))^2
            cont_val = coeffs[0] + coeffs[1]*log_S + coeffs[2]*log_S**2
            cont_val_discounted = cont_val * self.discount
            
            intrinsic_val = np.maximum(self.K - current_state, 0.0)
            
            return intrinsic_val >= cont_val_discounted
            
    return ExercisePolicy(K, cont_coeffs, n_dates, r, T)

# EVOLVE-BLOCK-END
