"""Bermudan exercise-policy research using Least Squares Monte Carlo (LSMC)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option.
    
    Args:
        training_paths: np.array of shape (n_paths, n_steps+1) or similar. 
                        Based on task description: 4096 paths, 12 monthly dates (no t=0).
                        Usually paths include t=0. Let's assume shape (N, 13) where col 0 is S0, 
                        cols 1-12 are S at t=1/12, ..., 12/12.
                        Or shape (N, 12) if t=0 is excluded? 
                        Task says "12 monthly exercise dates (no time-0 exercise)".
                        Spot 90/100/110. 
                        Let's inspect typical structures. Usually, paths are generated from S0.
                        If training_paths has shape (N, 13), index 0 is S0, indices 1..12 are exercise dates.
                        If shape (N, 12), indices 0..11 are exercise dates.
                        Let's assume the standard GBM path generation includes the initial spot.
                        However, the prompt says "Candidate receives current states only during scoring".
                        The input `training_paths` is used to fit the policy.
                        
                        Let's assume `training_paths` is an array of stock prices.
                        If it's GBM, we need to know the parameters or infer them.
                        The contract likely contains parameters or the paths are pre-generated.
                        Given "Spot 90/100/110", it implies multiple starting spots or just a description of the scenario.
                        Actually, "Spot 90/100/110" might refer to different scenarios or just the range.
                        But the task says "Bermudan put on risk-neutral GBM... Spot 90/100/110".
                        This phrasing is slightly ambiguous. Does it mean the initial spot is random? 
                        Or are there 3 distinct contracts? 
                        "4096 training paths ... per contract".
                        "Spot 90/100/110" might mean the initial spot S0 is one of these?
                        Or maybe the paths are generated for S0=100?
                        Let's look at the baseline import: `stopping_baseline`.
                        
                        Let's assume the standard case: S0 is fixed (likely 100) and `training_paths` contains the simulated prices.
                        If S0 varies, the policy must be state-dependent.
                        
                        Let's assume `training_paths` shape is (N, T+1) where T=12.
                        Columns 1..12 correspond to exercise dates t_1...t_12.
                        Column 0 is S0.
                        
                        Parameters:
                        K = 100
                        r = 0.05
                        sigma = 0.2
                        T = 1.0
                        dt = 1/12
                        n_steps = 12
                        """
    
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Extract parameters from contract if available, otherwise use defaults from task description
    # The contract object might have attributes or be a dict.
    # Let's try to access attributes safely.
    try:
        K = getattr(contract, 'K', 100.0)
        r = getattr(contract, 'r', 0.05)
        sigma = getattr(contract, 'sigma', 0.2)
        T = getattr(contract, 'T', 1.0)
    except Exception:
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0
        
    # Assume training_paths is (N, 13) where col 0 is S0, cols 1-12 are exercise times.
    # If shape is (N, 12), we assume cols 0-11 are exercise times and S0 is not included or is constant.
    # Given "Spot 90/100/110" in description, it might imply the paths cover a range or S0 is fixed.
    # Let's check the shape.
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        # Fallback to simple threshold if shape is unexpected
        class ExercisePolicy:
            def __init__(self):
                pass
            def exercise(self, current_state, time):
                # time is likely index 0..11 or 1..12
                # Assume current_state is array of spots
                return current_state <= K
        return ExercisePolicy()
        
    N, M = paths.shape
    
    # Determine exercise indices
    # If M == 13, exercise indices are 1..12
    # If M == 12, exercise indices are 0..11
    if M == 13:
        ex_indices = range(1, 13)
        S0_idx = 0
    elif M == 12:
        ex_indices = range(0, 12)
        S0_idx = None # S0 not explicitly in array, or paths start at t=1
    else:
        # Assume last 12 columns are exercise dates
        ex_indices = range(M-12, M)
        S0_idx = 0 if M > 12 else None

    # Number of exercise dates
    n_ex = len(ex_indices)
    dt = T / n_ex
    disc = np.exp(-r * dt)
    
    # Payoff at each exercise date
    # Payoff is max(K - S, 0)
    payoffs = np.zeros((N, n_ex))
    for i, idx in enumerate(ex_indices):
        payoffs[:, i] = np.maximum(K - paths[:, idx], 0.0)
    
    # Backward induction LSMC
    # V[i] is the value at exercise date i
    # V[n_ex-1] is just the payoff at the last date
    V = np.zeros((N, n_ex))
    V[:, n_ex-1] = payoffs[:, n_ex-1]
    
    # Thresholds or Regression Coefficients for each step
    # We will store the continuation values C[i] for each path at step i
    # and the decision: exercise if Payoff[i] >= C[i]
    
    # Features: We use log(S) and its powers to approximate the continuation function
    # F(S) = a0 + a1*log(S) + a2*log(S)^2 ...
    # To keep it fast and robust, we use a small number of features.
    # Let's use features: [1, log(S), log(S)^2, log(S)^3]
    
    max_features = 4
    reg_coeffs = [None] * n_ex
    continuation_vals = np.zeros((N, n_ex))
    
    # Iterate backwards from n_ex-2 down to 0
    for i in range(n_ex - 2, -1, -1):
        if time.time() > deadline:
            # If time is running out, break and use simple heuristic for remaining
            # For remaining steps, assume exercise if deep ITM
            break
            
        # Future value discounted to current time
        # The value at step i+1 is V[:, i+1]
        # Discounted to step i: disc * V[:, i+1]
        y = disc * V[:, i+1]
        
        # Features from current state S_i
        S_i = paths[:, ex_indices[i]]
        
        # Avoid log(0)
        S_safe = np.maximum(S_i, 1e-6)
        log_S = np.log(S_safe)
        
        # Construct feature matrix
        # X = [1, log_S, log_S^2, log_S^3]
        X = np.column_stack([
            np.ones(N),
            log_S,
            log_S**2,
            log_S**3
        ])
        
        # Solve least squares: X @ coeff = y
        # Use np.linalg.lstsq for stability
        try:
            coeff, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback: use mean
            coeff = np.zeros(max_features)
            coeff[0] = np.mean(y)
            
        reg_coeffs[i] = coeff
        
        # Calculate continuation value C_i for all paths
        C_i = X @ coeff
        continuation_vals[:, i] = C_i
        
        # Update V_i: Value is max(Payoff_i, C_i)
        # But for the next step's regression target, we need the *realized* value of the optimal policy.
        # In LSMC, we typically use the max of payoff and continuation value as the target for the previous step.
        V[:, i] = np.maximum(payoffs[:, i], C_i)
        
    # Define the Exercise Policy object
    class ExercisePolicy:
        def __init__(self, K, reg_coeffs, ex_indices, paths_shape, S0_idx):
            self.K = K
            self.reg_coeffs = reg_coeffs
            self.ex_indices = ex_indices
            self.paths_shape = paths_shape
            self.S0_idx = S0_idx
            self.n_ex = len(ex_indices)
            
            # Precompute nothing else, we compute features on the fly
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices at time 'time'
            time: index of the exercise date (0-based or 1-based? 
                  Task says "12 monthly exercise dates (no time-0 exercise)".
                  Usually, `time` passed to exercise is the step index 0..11.
                  Let's assume `time` is 0-based index of the exercise date.
            """
            # Ensure current_state is an array
            S = np.asarray(current_state)
            
            # Determine which step this is
            # If `time` is an integer index 0..11
            if isinstance(time, (int, np.integer)):
                step_idx = int(time)
            else:
                # If time is a float (e.g., 1/12), we need to map it to index
                # But usually in these interfaces, `time` is the step index.
                # Let's assume it's an index.
                step_idx = int(time)
                
            if step_idx < 0 or step_idx >= self.n_ex:
                # Out of bounds, default to no exercise or exercise if deep ITM
                return S <= self.K
                
            # Get coefficients for this step
            coeff = self.reg_coeffs[step_idx]
            if coeff is None:
                # Fallback if optimization didn't finish
                return S <= self.K
                
            # Calculate features
            S_safe = np.maximum(S, 1e-6)
            log_S = np.log(S_safe)
            
            # X = [1, log_S, log_S^2, log_S^3]
            # Since we only need the value, we can compute directly
            C = (coeff[0] + 
                 coeff[1] * log_S + 
                 coeff[2] * log_S**2 + 
                 coeff[3] * log_S**3)
                 
            # Payoff
            P = np.maximum(self.K - S, 0.0)
            
            # Exercise if Payoff >= Continuation Value
            return P >= C
            
    return ExercisePolicy(K, reg_coeffs, ex_indices, paths.shape, S0_idx)

# EVOLVE-BLOCK-END
