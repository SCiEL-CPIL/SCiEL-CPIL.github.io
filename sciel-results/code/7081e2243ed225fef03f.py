"""Bermudan put exercise policy: smooth regression + high-precision threshold root-finding."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq
from scipy.special import expit  # sigmoid

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt

    # training_paths: shape (n_paths, n_time_steps) or similar
    # We need to figure out the structure. Typically for Bermudan:
    # training_paths[i, j] is the spot price at time j for path i
    # j goes from 0 to n_exercise (or n_exercise+1 if including terminal)
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = paths.shape[0]
    # Assume paths has n_exercise+1 columns (time 0 to T) or n_exercise columns (exercise dates only)
    # Given "12 monthly exercise dates (no time-0 exercise)", the training paths likely include
    # the state at each exercise date. Let's assume shape (n_paths, n_exercise) or (n_paths, n_exercise+1).
    # If the first column is time 0, we ignore it for exercise decisions but might use it for features.
    # Standard Bermudan setup: we observe S at t_1, ..., t_n.
    
    # Let's assume paths is (n_paths, n_exercise) representing S at t_1..t_n
    # Or (n_paths, n_exercise+1) with S_0 included.
    # The prompt says "candidate receives current states only during scoring".
    # For training, we have the full path.
    
    # Let's determine n_steps from paths
    n_steps = paths.shape[1]
    
    # If n_steps == n_exercise, indices 0..n_exercise-1 correspond to t_1..t_n
    # If n_steps == n_exercise+1, indices 0..n_exercise correspond to t_0..t_n
    # We will handle both. The exercise decision at step k (1-indexed in time)
    # depends on S at that time and the continuation value.
    
    # For a Bermudan put, the payoff if exercised at time t_k is (K - S_k)^+.
    # The continuation value is the expected discounted value of optimal future exercise.
    
    # Longstaff-Schwartz method:
    # 1. Simulate paths (we have them in training_paths).
    # 2. At each exercise date t_k from last to first:
    #    a. Compute the "exercise payoff" E_k = (K - S_k)^+.
    #    b. Compute the "continuation value" C_k = E[exp(-r*dt) * V_{k+1} | S_k].
    #       This is estimated by regressing the discounted future values V_{k+1} on basis functions of S_k.
    #    c. V_k = max(E_k, C_k).
    #    d. Store the threshold or the regression coefficients.
    
    # Basis functions: We'll use a set of basis functions of S.
    # To improve accuracy, we can use a richer basis or normalize S.
    # Let's use basis: [1, S, S^2, S^3, exp(-S), exp(S), (S-K), (S-K)^2, ...]
    # Or better: normalize S by K to keep values O(1).
    # Let x = S/K.
    # Basis: [1, x, x^2, x^3, exp(-x), exp(-2x), (x-1), (x-1)^2, (x-1)^3]
    
    def make_basis(S_vals):
        """Create basis matrix for regression. S_vals is 1D array of spot prices."""
        x = S_vals / K
        # Use a mix of polynomial and exponential basis for flexibility
        # x is typically around 1.0
        B = np.column_stack([
            np.ones_like(x),
            x,
            x**2,
            x**3,
            np.exp(-x),
            np.exp(-2*x),
            (x - 1.0),
            (x - 1.0)**2,
            (x - 1.0)**3,
            np.exp(-(x - 1.0)**2),  # Gaussian bump
        ])
        return B
    
    # We need to compute the values V_k for each path at each time step.
    # V_{n+1} = 0 (no value after final exercise date, or terminal payoff if it were American)
    # For Bermudan, V_n = (K - S_n)^+ since no future exercise.
    
    # Let's store V for each step
    # V[k] is an array of shape (n_paths,) representing the value at time t_k for each path.
    # We compute backwards from k = n_exercise to 1.
    
    # First, determine the mapping from paths columns to time steps.
    # If n_steps == n_exercise:
    #   col 0 -> t_1, col 1 -> t_2, ..., col n_exercise-1 -> t_n
    # If n_steps == n_exercise + 1:
    #   col 0 -> t_0, col 1 -> t_1, ..., col n_exercise -> t_n
    #   We use cols 1..n_exercise for exercise decisions.
    
    if n_steps == n_exercise:
        # cols 0..n_exercise-1 correspond to t_1..t_n
        s_cols = list(range(n_exercise))
    else:
        # Assume n_steps == n_exercise + 1
        # cols 1..n_exercise correspond to t_1..t_n
        s_cols = list(range(1, n_exercise + 1))
    
    # V at final exercise date t_n: V_n = (K - S_n)^+
    S_n = paths[:, s_cols[-1]]
    V_next = np.maximum(K - S_n, 0.0)
    
    # We will store the continuation value regression for each step k < n.
    # For step k (1-indexed, t_k), we regress V_{k+1} (discounted) on S_k.
    # Then C_k = B(S_k) @ coeffs.
    # V_k = max(E_k, C_k).
    
    # We need to store the basis coefficients or the resulting continuation values for evaluation.
    # For evaluation, we only have current state S_t. We need to decide whether to exercise.
    # The decision is: Exercise if (K - S_t)^+ > C_t(S_t).
    # So we need C_t(S_t) as a function of S_t.
    # We can store the regression coefficients for each step and compute C_t(S) on the fly.
    
    # However, the basis functions depend on K, which is fixed.
    # Let's store the coefficients for each step k (1 to n-1).
    # coeffs[k] is the coefficient vector for step k.
    
    coeffs_list = [None] * n_exercise  # coeffs_list[k] for step k+1 (0-indexed)
    
    # Iterate backwards from k = n-1 down to 1 (1-indexed)
    # In 0-indexed for s_cols: from len(s_cols)-2 down to 0
    for k_idx in range(len(s_cols) - 2, -1, -1):
        # Current time step is s_cols[k_idx]
        # Next time step is s_cols[k_idx + 1]
        
        S_k = paths[:, s_cols[k_idx]]
        S_next = paths[:, s_cols[k_idx + 1]]
        
        # V_next is the value at S_next (already computed from previous iteration)
        # Discounted future value: exp(-r*dt) * V_next
        discounted_V_next = np.exp(-r * dt) * V_next
        
        # Regress discounted_V_next on basis(S_k)
        B_k = make_basis(S_k)
        
        # Solve least squares
        # Use np.linalg.lstsq for stability
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(B_k, discounted_V_next, rcond=None)
        except Exception:
            coeffs = np.zeros(B_k.shape[1])
        
        coeffs_list[k_idx] = coeffs
        
        # Continuation value at S_k
        C_k = B_k @ coeffs
        
        # Exercise payoff at S_k
        E_k = np.maximum(K - S_k, 0.0)
        
        # Value at S_k
        V_k = np.maximum(E_k, C_k)
        
        # Update V_next for next iteration
        V_next = V_k
    
    # Now we have coeffs_list for steps 1 to n-1.
    # For step n, the decision is always to exercise if (K - S_n)^+ > 0, which is always true if S_n < K.
    # But wait, at the final exercise date, there is no continuation value, so we always exercise if in-the-money.
    # So for k = n, exercise if S_n < K.
    
    # For k < n, exercise if (K - S_k)^+ > C_k(S_k).
    # C_k(S) = B(S) @ coeffs_list[k_idx].
    
    # To make the policy smooth and robust, we can find the threshold S_k* where (K - S) = C_k(S).
    # For S < S_k*, (K - S) > C_k(S) (assuming C_k is increasing in S? Not necessarily).
    # Actually, for a put, the continuation value is typically increasing in S.
    # The exercise region is S < threshold.
    # Let's find the threshold for each step k by solving (K - S) = C_k(S) for S.
    # If no solution, or if the inequality holds for all S < K, then threshold is K.
    # If the inequality never holds, threshold is 0.
    
    # Let's compute the threshold for each step k (1 to n-1).
    thresholds = [0.0] * n_exercise  # thresholds[k_idx] for step k_idx+1
    
    for k_idx in range(len(s_cols) - 1):  # steps 1 to n-1
        coeffs = coeffs_list[k_idx]
        
        # Define the difference function: D(S) = (K - S) - C(S)
        # We want to find S such that D(S) = 0, and D(S) > 0 for S < threshold.
        
        def C_of_S(S_val):
            if S_val <= 0:
                S_val = 1e-6
            B_val = make_basis(np.array([S_val]))
            return float(B_val @ coeffs)
        
        def D(S_val):
            return (K - S_val) - C_of_S(S_val)
        
        # Find the root of D(S) = 0 in the interval [0, K].
        # D(K) = -C(K). C(K) is typically positive, so D(K) < 0.
        # D(0) = K - C(0). C(0) is typically small, so D(0) > 0.
        # So there should be a root in (0, K).
        
        threshold = K  # Default: exercise if S < K
        
        try:
            # Check if D(0) > 0 and D(K) < 0
            D_0 = D(1e-6)
            D_K = D(K)
            
            if D_0 > 0 and D_K < 0:
                # Use Brent's method for high precision
                threshold = brentq(D, 1e-6, K, xtol=1e-10, rtol=1e-12)
            elif D_0 > 0 and D_K >= 0:
                # D is positive everywhere in [0, K], so always exercise
                threshold = K
            elif D_0 <= 0 and D_K < 0:
                # D is negative everywhere, so never exercise (unless S=0)
                threshold = 0.0
            else:
                # D(0) <= 0 and D(K) >= 0, unusual. Try to find root.
                try:
                    threshold = brentq(D, 1e-6, K, xtol=1e-10, rtol=1e-12)
                except Exception:
                    threshold = K
        except Exception:
            threshold = K
        
        thresholds[k_idx] = threshold
    
    # For the final step (k = n), threshold is K.
    thresholds[-1] = K
    
    # Now create the ExercisePolicy object.
    
    class ExercisePolicy:
        def __init__(self, thresholds, n_exercise):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices for each path.
            time: current time step (1-indexed, 1 to n_exercise).
            Returns: boolean vector, True if exercise.
            """
            # time is 1-indexed.
            # If time is not in [1, n_exercise], return False.
            if time < 1 or time > self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            # Map time to threshold index.
            # time 1 -> thresholds[0], time 2 -> thresholds[1], ..., time n -> thresholds[n-1]
            idx = time - 1
            thresh = self.thresholds[idx]
            
            # Exercise if S < threshold
            return current_state < thresh
    
    return ExercisePolicy(thresholds, n_exercise)
# EVOLVE-BLOCK-END
