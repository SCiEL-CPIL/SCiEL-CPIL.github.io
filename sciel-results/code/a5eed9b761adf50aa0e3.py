"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using Monte Carlo Backward Induction (DP).
    
    Strategy:
    Instead of optimizing a parametric boundary (like the current approach), we use the
    training paths to compute the optimal exercise decision via backward induction.
    
    1. At the final exercise date (j=11), we always exercise if the intrinsic value is positive.
       (Actually, for a Bermudan, we can choose not to exercise, but since there are no future
       dates, exercising is optimal if intrinsic > 0).
       
    2. For j = 10 down to 0:
       We need to estimate the continuation value V_j(S_j) for each path.
       Since we don't have a grid, we use the training paths themselves.
       For each path i at time j, we look at the "value" of continuing, which is the
       discounted expected value of the optimal policy from j+1 onwards.
       
       However, a direct path-wise DP is tricky because the state space is continuous.
       A common approximation is to use the training paths to estimate the continuation value
       via regression or simply by using the "average" of the future payoffs for paths that
       are in a similar state.
       
       A simpler and more robust approach for this specific setup (12 steps, 4096 paths) is:
       Use the training paths to compute the "optimal" exercise decision for each path at each
       step by assuming that the continuation value is the average of the discounted future
       payoffs for all paths that are still "active" and have a state close to S_j.
       
       But "close to" is ambiguous. Let's use a different structural change:
       **Longstaff-Schwarz (LSM) with Polynomial Regression**.
       
       LSM is the standard method for Bermudan options. It uses regression to estimate the
       continuation value at each step.
       
       Algorithm:
       1. Start from the last exercise date j=11.
          V_11(S) = max(K - S, 0)  (since no future steps, always exercise if in the money)
          Actually, for Bermudan, at the last step, we exercise if intrinsic > 0.
          So, continuation value at step 11 is 0 (if we don't exercise, we get 0).
          Wait, the value at step 11 is max(Intrinsic, 0) if we can exercise.
          Let's define V_j(S_j) as the value of the option at step j, given we haven't exercised yet.
          V_11(S_11) = max(K - S_11, 0)
          
       2. For j = 10 down to 0:
          a. Compute the discounted expected continuation value:
             E_j[V_{j+1}(S_{j+1}) | S_j]
          b. We estimate this using regression on the training paths.
             We fit a model (e.g., polynomial) to predict V_{j+1}(S_{j+1}) from S_j.
             But V_{j+1} is known for each path from the previous step.
             So, for each path i, we have (S_j[i], V_{j+1}[i]).
             We fit a regression model m_j(S) such that m_j(S_j[i]) ≈ V_{j+1}[i].
          c. The continuation value is m_j(S_j[i]).
          d. The optimal decision is: Exercise if Intrinsic_j(S_j[i]) > m_j(S_j[i]).
          e. The value V_j(S_j[i]) = max(Intrinsic_j(S_j[i]), m_j(S_j[i])).
          
       3. The policy is: Exercise at step j if S_j < Boundary_j, where Boundary_j is
          implicitly defined by the regression. To get a simple threshold policy, we can
          find the boundary by solving Intrinsic_j(B) = m_j(B).
          Or, we can just use the regression to make the decision directly.
          But the interface requires a function exercise(current_state, time) that returns a boolean.
          So we need a deterministic rule.
          
          We can use the regression model m_j to compute the continuation value for any S,
          and then exercise if (K - S) > m_j(S).
          
          This is a fundamentally different approach: it uses regression to estimate the
          continuation value, rather than optimizing a parametric boundary.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    if training_paths is None or len(training_paths) == 0:
        class FallbackPolicy:
            def exercise(self, current_state, time):
                return current_state < K
        return FallbackPolicy()

    paths = np.asarray(training_paths, dtype=float)
    n_paths, n_cols = paths.shape
    
    if n_cols >= 13:
        start_idx = 1
        end_idx = 12
    else:
        start_idx = 0
        end_idx = 11
        
    S_ex = paths[:, start_idx:end_idx+1]  # Shape (n_paths, 12)
    
    dt = T / n_exercises
    disc = np.exp(-r * dt)
    
    # V[j] will store the value of the option at step j for each path
    # We start from the last step
    V = np.zeros((n_paths, n_exercises))
    
    # At the last step (j=11), the value is the intrinsic value (since we can exercise)
    V[:, 11] = np.maximum(K - S_ex[:, 11], 0.0)
    
    # Regression models for each step j=0..10
    # We'll use polynomial regression of degree 3
    # Features: [1, S, S^2, S^3]
    degree = 3
    
    # For each step j from 10 down to 0
    for j in range(10, -1, -1):
        # Target: discounted V[j+1]
        target = disc * V[:, j+1]
        
        # Features for regression: polynomial in S_j
        S_j = S_ex[:, j]
        X = np.column_stack([
            np.ones_like(S_j),
            S_j,
            S_j ** 2,
            S_j ** 3
        ])
        
        # Fit linear regression: X @ beta = target
        # Use least squares
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback: use the average target as a constant model
            beta = np.array([np.mean(target), 0, 0, 0])
        
        # Continuation value for each path at step j
        cont_val = X @ beta
        
        # Intrinsic value at step j
        intrinsic = np.maximum(K - S_j, 0.0)
        
        # Optimal value at step j
        V[:, j] = np.maximum(intrinsic, cont_val)
        
        # Store the regression coefficients for this step
        # We'll use them to create the exercise policy
        # The exercise decision is: Exercise if Intrinsic > ContVal
        # i.e., (K - S) > cont_val(S)
        # This is not a simple threshold, but we can approximate it by finding the boundary
        # Or, we can just use the regression model in the policy.
        
    # Now, we need to create a policy that can be evaluated on new states.
    # The policy is: Exercise at step j if (K - S) > m_j(S), where m_j is the regression model.
    # We'll store the regression coefficients for each step.
    
    # Re-fit the regressions to store them properly
    reg_coefs = []
    for j in range(10, -1, -1):
        target = disc * V[:, j+1]
        S_j = S_ex[:, j]
        X = np.column_stack([
            np.ones_like(S_j),
            S_j,
            S_j ** 2,
            S_j ** 3
        ])
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.array([np.mean(target), 0, 0, 0])
        reg_coefs.append(beta)
    
    # Reverse the list so that reg_coefs[j] corresponds to step j
    reg_coefs = reg_coefs[::-1]
    
    class ExercisePolicy:
        def __init__(self, reg_coefs, n_exercises, K):
            self.reg_coefs = reg_coefs
            self.n_exercises = n_exercises
            self.K = K
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices.
            time: step index 0..11.
            """
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            if time == self.n_exercises - 1:
                # At the last step, exercise if in the money
                return current_state < self.K
            
            # For other steps, use the regression model
            S = np.asarray(current_state, dtype=float)
            beta = self.reg_coefs[time]
            
            # Continuation value: m_j(S)
            cont_val = beta[0] + beta[1]*S + beta[2]*S**2 + beta[3]*S**3
            
            # Intrinsic value
            intrinsic = np.maximum(self.K - S, 0.0)
            
            # Exercise if intrinsic > continuation
            return intrinsic > cont_val
    
    return ExercisePolicy(reg_coefs, n_exercises, K)
# EVOLVE-BLOCK-END
