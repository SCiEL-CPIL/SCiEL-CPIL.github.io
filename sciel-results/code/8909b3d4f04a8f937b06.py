"""Bermudan exercise-policy research: parametric smooth boundary via sigmoid of polynomial."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin before 120s kill

    # Fixed parameters from task contract
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12  # 12 monthly exercise dates
    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Process training paths
    # training_paths shape: (n_paths, n_steps + 1) i.e., S_0 to S_12
    # Exercise decisions at t=1..12 (indices 1..12 in the path array)
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    n_paths, n_cols = paths.shape

    # We expect n_cols = n_steps + 1 = 13 (S_0 through S_12)
    # Exercise at steps 1..12 corresponds to columns 1..12
    if n_cols < n_steps + 1:
        # Fallback: assume paths are just the exercise-date states
        n_cols = paths.shape[1]
        n_steps_actual = n_cols - 1
    else:
        n_steps_actual = n_steps

    # Extract states at exercise dates: columns 1..n_steps_actual
    states = paths[:, 1:n_steps_actual + 1]  # shape (n_paths, n_steps_actual)
    n_exercise = n_steps_actual

    # Remaining time fractions: at exercise step i (1-indexed), remaining steps = n_exercise - i + 1
    # But for the boundary, we use the fraction of time elapsed or remaining
    # Let's define: at step i (0-indexed in states, i=0..n_exercise-1), 
    # the exercise date is t_i = (i+1)*dt, remaining time = T - t_i = (n_exercise - i)*dt
    # remaining_fraction = (n_exercise - i) / n_exercise
    step_indices = np.arange(n_exercise)  # 0-indexed
    remaining_frac = (n_exercise - step_indices) / n_exercise  # shape (n_exercise,)

    # European put threshold for reference: S_E = K * exp(-r * tau) where tau is time to expiry
    # Actually for a put, the European value threshold isn't a simple boundary.
    # Let's use a simple parametric boundary: exercise if S < B(tau)
    # where B(tau) = K * sigmoid(a * (1 - tau) + b) or similar
    # 
    # Parametric form: B(tau) = K * (1 / (1 + exp(-a*(tau - c)))) 
    # At tau=1 (start), B should be low (deep ITM put, exercise early if S very low)
    # At tau=0 (expiry), B should be K (exercise if S < K)
    # Actually for a put: exercise if S < B(tau)
    # At expiry tau=0: exercise if S < K, so B(0) = K
    # At tau=T: exercise only if S is very low, so B(T) is small
    #
    # Let's use: B(tau) = K * exp(-alpha * tau) * (1 + beta * tau)
    # Or simpler: B(tau) = K * (1 - gamma * tau)
    # At tau=0: B=K, at tau=T: B=K*(1-gamma*T)
    #
    # Better: use a sigmoid-based smooth boundary
    # B(tau) = K * sigmoid(logit_K - a * tau)
    # where sigmoid(x) = 1/(1+exp(-x))
    # At tau=0: B = K * sigmoid(logit_K) ≈ K if logit_K large
    # At tau=T: B = K * sigmoid(logit_K - a*T)
    #
    # Let's parameterize: B(tau) = K * s(a * (1 - tau) + b)
    # where s is sigmoid. We want B(0) ≈ K, so s(a + b) ≈ 1, meaning a+b large
    # B(T) = K * s(b), we want this to be small, so b is negative
    #
    # Alternative: B(tau) = K * exp(-c * tau^d)
    # At tau=0: B=K, at tau=T: B=K*exp(-c*T^d)
    #
    # Let's go with a polynomial in tau with sigmoid:
    # B(tau) = K * sigmoid(sum_{k=0}^{p} w_k * tau^k)
    # We want B(0) ≈ K => sigmoid(w_0) ≈ 1 => w_0 > 0
    # B(1) should be smaller
    #
    # Simpler approach: B(tau) = K * (1 - sum_{k=1}^{p} w_k * tau^k) with B > 0
    # Let's use: B(tau) = K * exp(-sum_{k=1}^{p} w_k * tau^k)
    # At tau=0: B=K
    # At tau=1: B=K*exp(-sum w_k)
    #
    # Let's use degree-3 polynomial in the exponent:
    # B(tau) = K * exp(-(w1*tau + w2*tau^2 + w3*tau^3))
    # Parameters: w1, w2, w3 >= 0
    #
    # Exercise policy: at step i, exercise if S_i < B(tau_i)
    # where tau_i = (n_exercise - i) * dt / T = (n_exercise - i) / n_exercise

    def compute_value(params):
        """Compute the mean discounted payoff for a given boundary parameterization."""
        # params = [w1, w2, w3]
        w1, w2, w3 = params
        # Ensure non-negative via softplus-like transformation or just clip
        w1 = max(0, w1)
        w2 = max(0, w2)
        w3 = max(0, w3)
        
        # Compute boundary at each exercise step
        # tau_i for step i (0-indexed in states): remaining time fraction
        taus = remaining_frac  # shape (n_exercise,)
        
        # B(tau) = K * exp(-(w1*tau + w2*tau^2 + w3*tau^3))
        exponents = w1 * taus + w2 * taus**2 + w3 * taus**3
        boundaries = K * np.exp(-exponents)  # shape (n_exercise,)
        
        # For each path, determine exercise decisions via backward induction
        # Actually, for a put, the optimal policy is: exercise at the first time
        # where S_i < B(tau_i) AND the continuation value is less than the exercise payoff
        # But with a parametric boundary, we approximate: exercise if S_i < B(tau_i)
        # This is a greedy/approximate policy. For better accuracy, we could do
        # backward induction using the boundary as a heuristic.
        #
        # Let's do proper backward induction with the boundary as a refinement:
        # 1. Start from the last step: exercise if S < K (always optimal at expiry for put)
        # 2. For earlier steps: exercise if S < B(tau) AND S < continuation_value
        #    where continuation_value = discount * E[max(K - S_next, 0) | S_i]
        #    But we don't have conditional expectations easily.
        #
        # Simpler: just use the boundary directly. Exercise if S_i < B(tau_i).
        # This is a one-step look-ahead approximation.
        
        # Exercise decisions: (n_paths, n_exercise)
        exercise = states < boundaries[np.newaxis, :]  # broadcasting
        
        # Compute discounted payoff
        # If exercised at step i (0-indexed), payoff = K - S_i, discounted by discount^(i+1)
        # If never exercised, payoff = max(K - S_final, 0) at expiry, discounted by discount^n_exercise
        
        # Find first exercise step for each path
        # exercise[:, i] is True if exercised at step i
        # We want the first i where exercise[:, i] is True
        
        # Create a matrix of discounted payoffs if exercised at step i
        payoffs = np.zeros((n_paths, n_exercise))
        for i in range(n_exercise):
            payoff_i = np.maximum(K - states[:, i], 0)
            payoffs[:, i] = payoff_i * (discount ** (i + 1))
        
        # For each path, find the first exercise decision
        # If no exercise, use the final payoff
        first_exercise = np.argmax(exercise, axis=1)  # index of first True, or n_exercise-1 if all False
        # But argmax returns 0 if all False, so we need to check
        any_exercise = exercise.any(axis=1)
        
        # Compute value
        values = np.zeros(n_paths)
        for i in range(n_exercise):
            mask = (first_exercise == i) & any_exercise
            values[mask] = payoffs[mask, i]
        
        # Paths that never exercised
        no_exercise = ~any_exercise
        if np.any(no_exercise):
            final_payoff = np.maximum(K - states[:, -1], 0)
            values[no_exercise] = final_payoff[no_exercise] * (discount ** n_exercise)
        
        return np.mean(values)

    def negative_value(params):
        return -compute_value(params)

    # Initial guess: w1=0.5, w2=0.1, w3=0.0
    x0 = np.array([0.5, 0.1, 0.0])
    
    # Try a few initial points
    best_params = x0
    best_val = compute_value(x0)
    
    initial_guesses = [
        np.array([0.3, 0.2, 0.0]),
        np.array([0.5, 0.1, 0.0]),
        np.array([0.8, 0.0, 0.0]),
        np.array([0.2, 0.3, 0.1]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
    ]
    
    for guess in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        val = compute_value(guess)
        if val > best_val:
            best_val = val
            best_params = guess.copy()
    
    # Refine with L-BFGS-B
    if time.time() - start_time < TIME_LIMIT - 20:
        try:
            result = minimize(
                negative_value,
                best_params,
                method='L-BFGS-B',
                bounds=[(0, 5), (0, 5), (0, 5)],
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if result.fun < -best_val:
                best_params = result.x
                best_val = -result.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            result = minimize(
                negative_value,
                best_params,
                method='L-BFGS-B',
                bounds=[(0, 5), (0, 5), (0, 5)],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if result.fun < -best_val:
                best_params = result.x
                best_val = -result.fun
        except Exception:
            pass
    
    # Create the exercise policy object
    w1, w2, w3 = best_params
    w1 = max(0, w1)
    w2 = max(0, w2)
    w3 = max(0, w3)
    
    class ExercisePolicy:
        def __init__(self, w1, w2, w3, K, r, sigma, T, n_steps):
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            self.dt = T / n_steps
            self.discount = np.exp(-r * self.dt)
        
        def exercise(self, current_state, time):
            """
            current_state: array of current stock prices, shape (n_paths,)
            time: current time or step index
            Returns: boolean vector indicating whether to exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            
            # Determine the remaining time fraction
            # time could be the step index (0-indexed) or the actual time
            # Let's assume time is the step index (0-indexed) for the exercise date
            if isinstance(time, (int, np.integer)):
                step_idx = int(time)
            elif isinstance(time, float):
                # Convert time to step index
                step_idx = int(round(time / self.dt)) - 1  # time is actual time, convert to step
                step_idx = max(0, min(step_idx, self.n_steps - 1))
            else:
                step_idx = 0
            
            step_idx = max(0, min(step_idx, self.n_steps - 1))
            
            # Remaining time fraction
            tau = (self.n_steps - step_idx) / self.n_steps
            
            # Boundary
            exponent = self.w1 * tau + self.w2 * tau**2 + self.w3 * tau**3
            boundary = self.K * np.exp(-exponent)
            
            # Exercise if S < boundary
            return current_state < boundary
    
    policy = ExercisePolicy(w1, w2, w3, K, r, sigma, T, n_steps)
    return policy
# EVOLVE-BLOCK-END
