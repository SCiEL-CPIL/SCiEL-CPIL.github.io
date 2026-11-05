# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using dense binomial tree backward induction.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 100.0  # Safety margin before 120s kill

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Use 1200 steps to ensure perfect alignment with 12 exercise dates (100 steps per period)
    # This matches the "calendar-matched 1200-step binomial-tree reference" mentioned in the task.
    n_tree_steps = 1200
    dt_tree = T / n_tree_steps
    
    # CRR parameters
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = np.exp(-sigma * np.sqrt(dt_tree))
    p = (np.exp(r * dt_tree) - d) / (u - d)
    discount_tree = np.exp(-r * dt_tree)
    
    # Determine S0
    S0 = np.median(training_paths[:, 0])
    
    # For efficiency, work with log-price grid
    log_S0 = np.log(S0)
    log_u = np.log(u)
    log_d = np.log(d)
    
    # Exercise dates in tree steps
    # Since n_tree_steps is a multiple of n_exercise (1200 % 12 == 0),
    # the exercise dates fall exactly on tree nodes.
    steps_per_exercise = n_tree_steps // n_exercise
    exercise_tree_steps = [i * steps_per_exercise for i in range(1, n_exercise + 1)]
    
    # Initialize the value at maturity
    j_max = n_tree_steps
    j_vals = np.arange(j_max + 1)
    log_S_final = log_S0 + n_tree_steps * log_d + j_vals * (log_u - log_d)
    S_final = np.exp(log_S_final)
    
    # Value at maturity: put payoff
    V = np.maximum(K - S_final, 0.0)
    
    # We need to track the exercise boundary at each exercise date
    boundaries = [None] * n_exercise
    
    # Map from tree step to exercise index
    step_to_ex_idx = {}
    for k, ts in enumerate(exercise_tree_steps):
        step_to_ex_idx[ts] = k
    
    # Backward induction
    for i in range(n_tree_steps - 1, -1, -1):
        # Continuation value
        V_cont = discount_tree * (p * V[1:] + (1.0 - p) * V[:-1])
        
        # Stock prices at current step
        j_curr = np.arange(i + 1)
        log_S_curr = log_S0 + i * log_d + j_curr * (log_u - log_d)
        S_curr = np.exp(log_S_curr)
        
        # Payoff at current step
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Exercise if payoff > continuation
        exercise_mask = payoff > V_cont
        
        # Check if this is an exercise date
        if i in step_to_ex_idx:
            ex_idx = step_to_ex_idx[i]
            if np.any(exercise_mask):
                # Boundary is the maximum S where exercise is optimal
                S_ex = S_curr[exercise_mask]
                boundaries[ex_idx] = float(np.max(S_ex))
            else:
                boundaries[ex_idx] = 0.0
        
        # Update V
        V = np.maximum(payoff, V_cont)
        
        # Deadline check
        if time.time() > deadline:
            break
    
    # Fill any remaining None boundaries with reasonable defaults
    # For a Bermudan put, earlier exercise dates have lower thresholds
    # Use a simple heuristic: interpolate from what we have
    for k in range(n_exercise):
        if boundaries[k] is None:
            # Default: slightly below K, decreasing for earlier dates
            boundaries[k] = K * 0.95 * (1.0 - 0.01 * k)
    
    # Ensure boundaries are monotonically non-decreasing (earlier dates should have lower or equal thresholds)
    # Actually for a put, the exercise region expands as maturity approaches, so thresholds should increase
    # Let's enforce monotonicity: boundaries[k] <= boundaries[k+1]
    for k in range(n_exercise - 1):
        if boundaries[k] > boundaries[k + 1]:
            boundaries[k] = boundaries[k + 1]
    
    # Also ensure no boundary exceeds K (no point exercising OTM)
    for k in range(n_exercise):
        if boundaries[k] > K:
            boundaries[k] = K
    
    # Create the exercise policy
    class ExercisePolicy:
        def __init__(self, K, boundaries, n_exercise, T):
            self.K = K
            self.boundaries = boundaries
            self.n_exercise = n_exercise
            self.T = T
        
        def exercise(self, current_state, time):
            """
            current_state: array of S values (shape: (n_paths,))
            time: current time t
            Returns boolean array: True if should exercise at this path
            """
            n_paths = len(current_state)
            result = np.zeros(n_paths, dtype=bool)
            
            # Determine which exercise date we're at based on time
            # time is assumed to be one of the exercise dates t_i = i / n_exercise
            # Calculate the index directly
            ex_idx_float = time * self.n_exercise
            ex_idx = int(round(ex_idx_float))
            
            # Clamp index to valid range [1, n_exercise]
            if ex_idx < 1:
                ex_idx = 1
            if ex_idx > self.n_exercise:
                ex_idx = self.n_exercise
            
            # boundaries is 0-indexed, so exercise date i (1-indexed) is at boundaries[i-1]
            if 0 <= ex_idx - 1 < len(self.boundaries):
                threshold = self.boundaries[ex_idx - 1]
                # Exercise if S <= threshold (and implicitly in the money since threshold <= K)
                result = (current_state <= threshold)
            
            return result
    
    policy = ExercisePolicy(K, boundaries, n_exercise, T)
    return policy

# EVOLVE-BLOCK-END
