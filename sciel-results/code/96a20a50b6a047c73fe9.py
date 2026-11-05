"""Bermudan put exercise policy via dense binomial tree backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a dense binomial tree.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 105.0  # Safety margin before 120s kill

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # S0 from training paths (first time point, median)
    S0 = float(np.median(training_paths[:, 0]))
    
    # Build a dense binomial tree
    # Use many steps for accuracy, but check time budget
    # We'll use a fixed number of steps that's fast enough
    n_steps = 2000  # Should be fast enough with numpy vectorization
    
    dt_tree = T / n_steps
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = 1.0 / u
    p = (np.exp(r * dt_tree) - d) / (u - d)
    
    # Exercise dates in tree steps: i * n_steps / n_exercise for i=1,...,n_exercise
    exercise_steps = [int(round(i * n_steps / n_exercise)) for i in range(1, n_exercise + 1)]
    
    # Backward induction on the tree
    # Value at each node
    # At maturity (step n_steps), value = max(K - S, 0)
    # We work backward, and at exercise dates we take max(payoff, continuation)
    
    # Number of nodes at step m is m+1
    # We'll store values at each step
    
    # Start from maturity
    # S at step n_steps, node j: S0 * u^j * d^(n_steps-j) = S0 * (u/d)^j * d^n_steps
    # = S0 * u^(2j - n_steps) * d^n_steps... let's just compute directly
    # S_j = S0 * u^j * d^(n_steps - j)
    
    j_arr = np.arange(n_steps + 1)
    S_maturity = S0 * (u ** j_arr) * (d ** (n_steps - j_arr))
    V = np.maximum(K - S_maturity, 0.0)
    
    # Map exercise steps to their index in the backward loop
    # We'll iterate from step n_steps-1 down to 0
    # At each exercise step, apply exercise decision
    
    # Create a set of exercise steps for O(1) lookup
    exercise_step_set = set(exercise_steps)
    
    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        # Number of nodes at this step: step + 1
        # Values at next step (step+1): V has length step+2
        # Up move: node j -> node j+1 at next step
        # Down move: node j -> node j at next step
        
        V_up = V[1:]  # length step+1
        V_down = V[:-1]  # length step+1
        
        V_cont = np.exp(-r * dt_tree) * (p * V_up + (1 - p) * V_down)
        
        # Check if this is an exercise step
        if step in exercise_step_set:
            # Compute spot prices at this step
            j_arr_curr = np.arange(step + 1)
            S_curr = S0 * (u ** j_arr_curr) * (d ** (step - j_arr_curr))
            payoff = np.maximum(K - S_curr, 0.0)
            V = np.maximum(payoff, V_cont)
        else:
            V = V_cont
        
        # Deadline check
        if time.time() > deadline:
            break
    
    # Now extract exercise boundaries at each exercise date
    # For each exercise date, find the threshold S where exercise becomes optimal
    # We need to map from tree step to actual time and spot
    
    boundaries = [None] * n_exercise
    
    # For each exercise date, we need to know the continuation value vs exercise
    # Let's redo backward induction but record the decision at each exercise step
    # Actually, let's re-run and record the boundary at each exercise step
    
    # Re-run backward induction, recording boundaries
    # At maturity
    j_arr = np.arange(n_steps + 1)
    S_maturity = S0 * (u ** j_arr) * (d ** (n_steps - j_arr))
    V = np.maximum(K - S_maturity, 0.0)
    
    # Map exercise steps to exercise index (0-based)
    # exercise_steps[i] corresponds to exercise date i+1, which is boundaries[i]
    ex_step_to_idx = {exercise_steps[i]: i for i in range(n_exercise)}
    
    for step in range(n_steps - 1, -1, -1):
        V_up = V[1:]
        V_down = V[:-1]
        V_cont = np.exp(-r * dt_tree) * (p * V_up + (1 - p) * V_down)
        
        if step in ex_step_to_idx:
            ex_idx = ex_step_to_idx[step]
            j_arr_curr = np.arange(step + 1)
            S_curr = S0 * (u ** j_arr_curr) * (d ** (step - j_arr_curr))
            payoff = np.maximum(K - S_curr, 0.0)
            
            # Exercise where payoff > continuation
            exercise_mask = payoff > V_cont
            
            if np.any(exercise_mask):
                # Find the maximum S where exercise is optimal
                S_exercise = S_curr[exercise_mask]
                boundaries[ex_idx] = float(np.max(S_exercise))
            else:
                boundaries[ex_idx] = 0.0
            
            V = np.maximum(payoff, V_cont)
        else:
            V = V_cont
        
        if time.time() > deadline:
            # Fill remaining boundaries with default
            for k in range(n_exercise):
                if boundaries[k] is None:
                    boundaries[k] = K * 0.7
            break
    
    # Fill any remaining None boundaries
    for k in range(n_exercise):
        if boundaries[k] is None:
            boundaries[k] = K * 0.7
    
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
            for i in range(1, self.n_exercise + 1):
                t_ex = i / self.n_exercise
                if abs(time - t_ex) < 1e-4:
                    ex_idx = i - 1  # 0-indexed
                    if ex_idx < len(self.boundaries) and self.boundaries[ex_idx] is not None:
                        threshold = self.boundaries[ex_idx]
                        # Exercise if S <= threshold and in the money
                        result = (current_state <= threshold) & (current_state < self.K)
                    break
            
            return result
    
    policy = ExercisePolicy(K, boundaries, n_exercise, T)
    return policy

# EVOLVE-BLOCK-END
