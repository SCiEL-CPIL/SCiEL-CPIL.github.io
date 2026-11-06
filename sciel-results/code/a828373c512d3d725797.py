"""Bermudan exercise policy using high-resolution binomial tree backward induction."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12

    # Use a high-resolution tree: multiple steps per exercise period
    steps_per_period = 50
    n_steps = n_exercise * steps_per_period  # 600 total steps
    dt = T / n_steps

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    S0 = 100.0
    log_S0 = np.log(S0)

    # Exercise dates in terms of tree steps
    # Exercise at tree steps: steps_per_period, 2*steps_per_period, ..., n_steps
    exercise_steps = [steps_per_period * k for k in range(1, n_exercise + 1)]
    # Map exercise date index (1..12) to tree step
    ex_date_to_step = {k: steps_per_period * k for k in range(1, n_exercise + 1)}

    # Backward induction
    # At final step (n_steps), value = payoff
    # We only need values at exercise dates, so we can store them there

    # Start at maturity
    k_final = np.arange(n_steps + 1)
    S_final = S0 * u ** (2 * k_final - n_steps)
    values = np.maximum(K - S_final, 0.0)

    # Store value arrays at each exercise date (for the policy)
    # value_at_ex[date_idx] = array of values at that exercise date's tree nodes
    value_at_ex = {}
    # Also store the spot prices at each exercise date for reference
    S_at_ex = {}

    # We need to go backward from step n_steps to step 1
    # At each step, if it's an exercise date, we apply max(exercise, continue)
    # and store the result

    # To save memory, we only store values at exercise dates
    # But we need the full values array for backward induction

    # Go backward
    for step in range(n_steps, 0, -1):
        # current 'values' is for step+1 (length step+2)
        # Compute continuation value for step (length step+1)
        up_val = values[1:step + 2]
        down_val = values[0:step + 1]
        cont_val = discount * (p * up_val + (1 - p) * down_val)

        # Check if this is an exercise date
        is_ex_date = False
        ex_idx = None
        for k in range(1, n_exercise + 1):
            if step == steps_per_period * k:
                is_ex_date = True
                ex_idx = k
                break

        if is_ex_date:
            k_nodes = np.arange(step + 1)
            S_nodes = S0 * u ** (2 * k_nodes - step)
            ex_val = np.maximum(K - S_nodes, 0.0)
            new_values = np.maximum(ex_val, cont_val)
            values = new_values
            value_at_ex[ex_idx] = new_values.copy()
            S_at_ex[ex_idx] = S_nodes.copy()
        else:
            values = cont_val

    # Check time
    if time.time() > DEADLINE - 5:
        # Build a simple fallback policy using critical prices from the tree
        critical_prices = {}
        for k in range(1, n_exercise + 1):
            vals = value_at_ex.get(k, None)
            if vals is not None:
                S_nodes = S_at_ex[k]
                # Find the highest S where exercise is optimal
                ex_val = np.maximum(K - S_nodes, 0.0)
                # Continuation value can be approximated by the stored values
                # where exercise was NOT optimal
                # Find critical price: highest S where ex_val >= cont_val
                # Since we stored max(ex, cont), we need to identify where ex > cont
                # Approximate: find where value equals ex_val (exercise was chosen)
                ex_mask = np.abs(vals - ex_val) < 1e-10
                exercising = np.where(ex_mask)[0]
                if len(exercising) > 0:
                    max_k = exercising[-1]
                    if max_k < len(S_nodes) - 1:
                        # Interpolate
                        S_ex = S_nodes[max_k]
                        S_no_ex = S_nodes[max_k + 1]
                        # Simple: use midpoint or the higher one
                        cp = (S_ex + S_no_ex) / 2
                    else:
                        cp = S_nodes[max_k]
                else:
                    cp = 0.0
                critical_prices[k] = cp
            else:
                critical_prices[k] = 0.0

        class ExercisePolicyFallback:
            def __init__(self):
                self.critical_prices = critical_prices
                self.n_exercise = n_exercise
                self.T = T
                self.K = K
                self.r = r

            def exercise(self, current_state, time):
                if time <= 0 or time > self.T:
                    return np.zeros(len(current_state), dtype=bool)
                k_float = time * self.n_exercise / self.T
                k = int(round(k_float))
                k = max(1, min(k, self.n_exercise))
                t_ex = k * self.T / self.n_exercise
                if abs(time - t_ex) > 0.01 * self.T:
                    return np.zeros(len(current_state), dtype=bool)
                cp = self.critical_prices.get(k, 0.0)
                return current_state < cp

        return ExercisePolicyFallback()

    # Build the policy using the full value functions at exercise dates
    # For each exercise date, we have:
    #   S_at_ex[k]: array of spot prices at the tree nodes
    #   value_at_ex[k]: array of optimal values at those nodes
    # To evaluate the policy for a given state S:
    #   Find the two tree nodes bracketing S
    #   Interpolate the value
    #   Compare with immediate exercise payoff
    #   Exercise if payoff >= interpolated value

    class ExercisePolicy:
        def __init__(self):
            self.value_at_ex = value_at_ex
            self.S_at_ex = S_at_ex
            self.n_exercise = n_exercise
            self.T = T
            self.K = K
            self.r = r
            self.S0 = S0
            self.u = u
            self.steps_per_period = steps_per_period
            # Precompute log of S for faster interpolation
            self.log_S_at_ex = {}
            for k in range(1, n_exercise + 1):
                self.log_S_at_ex[k] = np.log(self.S_at_ex[k])

        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time
            Returns: boolean array, True if exercise
            """
            if time <= 0 or time > self.T:
                return np.zeros(len(current_state), dtype=bool)

            # Determine exercise date index
            k_float = time * self.n_exercise / self.T
            k = int(round(k_float))
            k = max(1, min(k, self.n_exercise))

            # Check if close to an exercise date
            t_ex = k * self.T / self.n_exercise
            if abs(time - t_ex) > 0.01 * self.T:
                return np.zeros(len(current_state), dtype=bool)

            S = np.asarray(current_state, dtype=np.float64)
            n = len(S)

            # Get the tree data for this exercise date
            S_nodes = self.S_at_ex[k]
            vals = self.value_at_ex[k]
            log_S_nodes = self.log_S_at_ex[k]

            # Compute immediate exercise payoff
            ex_payoff = np.maximum(self.K - S, 0.0)

            # Interpolate continuation value from the tree
            # The tree nodes are in increasing order of S
            # For each S, find the two bracketing nodes and linearly interpolate
            # Use log-space for better interpolation

            log_S = np.log(S)
            # Clip to the range of the tree
            log_S_clipped = np.clip(log_S, log_S_nodes[0], log_S_nodes[-1])

            # Find the index where log_S falls in the tree
            # np.searchsorted returns the insertion point
            idx = np.searchsorted(log_S_nodes, log_S_clipped, side='right')
            # Clamp index to valid range
            idx = np.clip(idx, 1, len(log_S_nodes) - 1)

            # Linear interpolation in log-space
            log_S_left = log_S_nodes[idx - 1]
            log_S_right = log_S_nodes[idx]
            val_left = vals[idx - 1]
            val_right = vals[idx]

            # Interpolation weight
            denom = log_S_right - log_S_left
            # Avoid division by zero (shouldn't happen but just in case)
            safe_denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
            w = (log_S_clipped - log_S_left) / safe_denom
            w = np.clip(w, 0.0, 1.0)

            cont_val = w * val_right + (1 - w) * val_left

            # Exercise if payoff >= continuation value
            # Use a small tolerance to handle floating point issues
            return ex_payoff >= cont_val - 1e-10

    return ExercisePolicy()

# EVOLVE-BLOCK-END
