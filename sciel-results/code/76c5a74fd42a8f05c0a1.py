import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option.
    Uses a high-resolution binomial tree with linear interpolation of the
    exercise boundary for sub-node precision.
    """
    start_wall = time.time()
    deadline = 145.0  # safety margin from 150 s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Use a 1200-step binomial tree (as the task mentions as a reference).
    N_steps = 1200
    dt = T / N_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # We will compute the critical price S_crit at each of the 12 exercise dates.
    # Exercise dates correspond to tree steps j = k * (N_steps / n_exercises) = k * 100.
    # We perform backward induction.

    # Tree parameters:
    # S(i, j) = S0 * u**(2*i - j)  (CRR with ud=1)
    # We use S0 = 100 for tree construction; the critical price is a level.
    S0 = 100.0
    log_S0 = math.log(S0)
    log_u = math.log(u)

    # We need to record the boundary at steps j = 100, 200, ..., 1200.
    # For each such step, we find the node index i where the switch from
    # "exercise" to "continue" occurs, and interpolate for sub-node precision.

    # Terminal step N:
    i_terminal = np.arange(N_steps + 1)
    exponents_terminal = 2 * i_terminal - N_steps
    S_terminal = S0 * np.exp(exponents_terminal * log_u)
    V_next = np.maximum(K - S_terminal, 0.0)

    # We will store for each exercise step k the interpolated critical price.
    s_crit_values = [0.0] * (n_exercises + 1)  # 1-indexed

    # Backward induction
    for j in range(N_steps - 1, -1, -1):
        # Current step j: nodes i = 0..j
        i_curr = np.arange(j + 1)
        exponents_curr = 2 * i_curr - j
        S_curr = S0 * np.exp(exponents_curr * log_u)

        V_int = np.maximum(K - S_curr, 0.0)
        V_cont = disc * (p * V_next[1:] + (1 - p) * V_next[:-1])

        # Check if this is an exercise date
        if j % 100 == 0 and j > 0:
            k = j // 100
            diff = V_int - V_cont
            exercise_mask = diff >= 0
            if np.any(exercise_mask):
                # Find the last index where exercise is optimal
                last_exercise_idx = np.where(exercise_mask)[0][-1]
                # The next index (if it exists) is the first "continue" node
                first_continue_idx = last_exercise_idx + 1
                if first_continue_idx < j + 1:
                    # Interpolate between last_exercise_idx and first_continue_idx
                    # Using linear interpolation of the diff:
                    d_left = diff[last_exercise_idx]   # >= 0
                    d_right = diff[first_continue_idx] # < 0
                    if d_right < 0:
                        # Fraction where diff = 0
                        frac = d_left / (d_left - d_right)
                        # S values at the two nodes
                        S_left = S_curr[last_exercise_idx]
                        S_right = S_curr[first_continue_idx]
                        # Linear interpolation in S
                        s_crit_values[k] = S_left + frac * (S_right - S_left)
                    else:
                        # Should not happen, but fallback
                        s_crit_values[k] = S_curr[last_exercise_idx]
                else:
                    # All nodes exercise
                    s_crit_values[k] = S_curr[-1]
            else:
                # No exercise at this step
                s_crit_values[k] = 0.0

        # Update V for this step
        V_curr = np.where(V_int > V_cont, V_int, V_cont)
        V_next = V_curr

        # Deadline guard
        if time.time() > deadline:
            # Fallback: use the last computed boundaries; fill remaining with 0
            break

    # Fill any uncomputed boundaries with 0 (never exercise)
    for k in range(1, n_exercises + 1):
        if s_crit_values[k] == 0.0 and k <= 12:
            # If we didn't compute it, leave as 0
            pass

    class ExercisePolicy:
        def __init__(self, s_crits, n_exercises):
            self.s_crits = s_crits
            self.n_exercises = n_exercises

        def exercise(self, current_state, time_val):
            k_float = time_val * self.n_exercises
            k = int(round(k_float))

            if k < 1:
                return np.zeros_like(current_state, dtype=bool)
            if k > self.n_exercises:
                k = self.n_exercises

            crit = self.s_crits[k]
            if crit == 0.0:
                return np.zeros_like(current_state, dtype=bool)

            return current_state <= crit

    return ExercisePolicy(s_crit_values, n_exercises)
