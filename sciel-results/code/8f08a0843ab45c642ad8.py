"""Bermudan put exercise policy via analytic backward induction on a log-grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt

    # Build a log-space grid for spot prices
    # Put payoff is (K - S)+, so we care about S in [0, ~2*K]
    # Use log-grid for better resolution at low prices
    n_grid = 2000
    log_S_min = np.log(0.5)  # S_min ~ 0.5
    log_S_max = np.log(250.0)  # S_max ~ 250
    log_S_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_S_grid)

    # GBM transition: S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # In log-space: log(S_{t+dt}) = log(S_t) + (r - sigma^2/2)*dt + sigma*sqrt(dt)*Z
    # So log(S_{t+dt}) ~ Normal(log(S_t) + drift, sigma^2*dt)
    # We can compute the risk-neutral expectation using the fact that
    # E[exp(a * log(S_{t+dt})) | log(S_t)] = exp(a * (log(S_t) + drift) + 0.5 * a^2 * sigma^2 * dt)
    # But for the put payoff max(K - S, 0), we need E[max(K - S_{t+dt}, 0) | S_t]

    # Actually, let's use a discretized approach: for each grid point S_i,
    # the next state is S_i * exp(drift + sigma*sqrt(dt)*Z)
    # We can compute the expectation by quadrature or by mapping to the grid.
    
    # Better approach: use the closed-form for the one-step continuation value.
    # For GBM, E[max(K - S_{t+dt}, 0) | S_t = S] has a closed form.
    # Let S_{t+dt} = S * exp(X) where X ~ N(mu, sigma^2*dt), mu = (r - sigma^2/2)*dt
    # E[max(K - S*exp(X), 0)] = K*Phi(-d2) - S*exp(mu)*Phi(-d1)
    # where d1 = (log(S/K) + mu + sigma^2*dt) / (sigma*sqrt(dt))
    #       d2 = d1 - sigma*sqrt(dt)
    # Wait, let me re-derive. For a European put with payoff max(K - S_T, 0):
    # E[max(K - S_T, 0)] = K*Phi(-d2) - S_0*exp(r*T)*Phi(-d1)
    # where d1 = (ln(S_0/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T))
    #       d2 = d1 - sigma*sqrt(T)
    
    # For our case, T = dt, S_0 = S, and we're computing the undiscounted expectation.
    # Then we discount by exp(-r*dt).
    
    # So continuation_value(S) = exp(-r*dt) * E[max(K - S_{t+dt}, 0) | S_t = S]
    #                          = exp(-r*dt) * [K*Phi(-d2) - S*exp(r*dt)*Phi(-d1)]
    #                          = K*exp(-r*dt)*Phi(-d2) - S*Phi(-d1)
    
    # where d1 = (ln(S/K) + (r + sigma^2/2)*dt) / (sigma*sqrt(dt))
    #       d2 = d1 - sigma*sqrt(dt)
    
    from scipy.stats import norm
    
    def continuation_value(S):
        """Compute the discounted continuation value for a one-step Bermudan put."""
        d1 = (np.log(S / K) + (r + sigma**2 / 2.0) * dt) / (sigma * np.sqrt(dt))
        d2 = d1 - sigma * np.sqrt(dt)
        # Discounted expectation
        cv = K * np.exp(-r * dt) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return np.maximum(cv, 0.0)  # Continuation value is non-negative
    
    # Backward induction
    # V[n_exercise](S) = max(K - S, 0)  # Terminal payoff
    # V[i](S) = max(K - S, 0, continuation_value(V[i+1](S)))  # for i = n_exercise-1, ..., 1
    
    # Store the exercise decision at each step
    # exercise_decision[i][j] = True if we exercise at step i when S = S_grid[j]
    
    # We'll store the value function at each step
    # V[i] for i = 1, ..., n_exercise (1-indexed, i=n_exercise is terminal)
    
    # Terminal step
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store exercise decisions
    # exercise_decisions[i] for i = 1, ..., n_exercise-1 (steps where we can choose)
    # exercise_decisions[i][j] = True if exercise at step i, grid point j
    exercise_decisions = [None] * n_exercise  # index 0 unused, 1..n_exercise-1 used
    
    # Backward induction from step n_exercise-1 down to 1
    for i in range(n_exercise - 1, 0, -1):
        # Check time
        if time.time() > deadline - 5.0:
            # Not enough time, use a simple threshold
            break
        
        # Immediate exercise value
        exercise_val = np.maximum(K - S_grid, 0.0)
        
        # Continuation value: discount the expected next-step value
        # We need E[V_next(S_{t+dt}) | S_t = S]
        # This is not simply continuation_value(S) because V_next is not a simple put payoff.
        # We need to compute the expectation of V_next under the GBM transition.
        
        # Use numerical integration via the grid.
        # For each S_j, S_{t+dt} = S_j * exp(mu + sigma*sqrt(dt)*Z)
        # We can use Gauss-Hermite quadrature or just evaluate on a fine grid.
        
        # Let's use a quadrature approach. For each S_j, we need:
        # E[V_next(S_j * exp(mu + sigma*sqrt(dt)*Z))]
        # where mu = (r - sigma^2/2)*dt
        
        # Use Gauss-Hermite quadrature with ~50 points
        from scipy.special import roots_hermite
        n_quad = 60
        x_quad, w_quad = roots_hermite(n_quad)
        
        # Transform: Z = sqrt(2)*x, so exp(mu + sigma*sqrt(dt)*Z) = exp(mu + sigma*sqrt(2*dt)*x)
        # The expectation becomes: (1/sqrt(pi)) * sum(w_quad * V_next(S_j * exp(mu + sigma*sqrt(2*dt)*x_quad)))
        
        mu = (r - sigma**2 / 2.0) * dt
        sd = sigma * np.sqrt(2.0 * dt)
        
        # For each grid point S_j, compute the expectation
        # S_next = S_j * exp(mu + sd * x_quad)  -> shape (n_grid, n_quad)
        # This might be memory-heavy, so do it in chunks
        
        n_chunk = 200
        V_cont = np.zeros(n_grid)
        
        for chunk_start in range(0, n_grid, n_chunk):
            chunk_end = min(chunk_start + n_chunk, n_grid)
            S_chunk = S_grid[chunk_start:chunk_end]  # shape (chunk_size,)
            # S_next = S_chunk[:, None] * exp(mu + sd * x_quad[None, :])
            log_S_next = np.log(S_chunk)[:, None] + mu + sd * x_quad[None, :]
            S_next = np.exp(log_S_next)  # shape (chunk_size, n_quad)
            
            # Evaluate V_next at S_next via interpolation
            # V_next is defined on S_grid, so interpolate
            V_next_interp = np.interp(S_next, S_grid, V_next, left=V_next[0], right=0.0)
            
            # Quadrature: (1/sqrt(pi)) * sum(w_quad * V_next_interp)
            # Note: the standard Gauss-Hermite integral is (1/sqrt(pi)) * sum(w_i * f(sqrt(2)*x_i))
            # Here our integral is E[f(Z)] where Z~N(0,1) = (1/sqrt(2*pi)) * integral exp(-x^2/2) f(x) dx
            # = (1/sqrt(pi)) * integral exp(-u^2) f(sqrt(2)*u) du
            # So with u = x_quad, the weight is (1/sqrt(pi)) * w_quad
            V_cont[chunk_start:chunk_end] = np.sum(w_quad * V_next_interp, axis=1) / np.sqrt(np.pi)
        
        # Discount
        V_cont = np.exp(-r * dt) * V_cont
        
        # Optimal value: max(exercise, continuation)
        V_opt = np.maximum(exercise_val, V_cont)
        
        # Exercise decision: exercise if exercise_val >= V_cont
        exercise_decisions[i] = exercise_val >= V_cont
        
        V_next = V_opt
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, S_grid, exercise_decisions, n_exercise, exercise_times, dt):
            self.S_grid = S_grid
            self.exercise_decisions = exercise_decisions
            self.n_exercise = n_exercise
            self.exercise_times = exercise_times
            self.dt = dt
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices (1D)
            time: current time (scalar)
            Returns: boolean array of same shape as current_state
            """
            # Determine which exercise step we're at
            # time is in [0, T], exercise at times exercise_times[0], ..., exercise_times[n_exercise-1]
            # Step i (1-indexed) corresponds to time exercise_times[i-1]
            
            # Find the step index
            # If time is close to exercise_times[i-1], we're at step i
            # Use the closest exercise time
            if time <= 0:
                return np.zeros(len(current_state), dtype=bool)
            
            # Find which exercise step this corresponds to
            # exercise_times is 1-indexed in our decisions: exercise_decisions[1] is at time exercise_times[0]
            step_idx = None
            min_dist = float('inf')
            for i in range(1, self.n_exercise):
                t_i = self.exercise_times[i - 1]
                dist = abs(time - t_i)
                if dist < min_dist:
                    min_dist = dist
                    step_idx = i
            
            if step_idx is None or step_idx >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            dec = self.exercise_decisions[step_idx]
            if dec is None:
                # Fallback: simple threshold
                return current_state < K
            
            # Interpolate the decision: for each current_state, find the nearest grid point
            # Actually, the decision is a boolean per grid point. We can use the grid value directly.
            # For a given S, find the index in S_grid and use that decision.
            # Since the decision is piecewise constant on the grid, we can use searchsorted.
            indices = np.searchsorted(self.S_grid, current_state)
            # Clip to valid range
            indices = np.clip(indices, 0, len(self.S_grid) - 1)
            # For better accuracy, use the nearest grid point
            # Actually, searchsorted gives the insertion point. Let's use the nearest.
            # Simpler: just use the grid decision at the nearest point
            return dec[indices]
    
    return ExercisePolicy(S_grid, exercise_decisions, n_exercise, exercise_times, dt)
# EVOLVE-BLOCK-END
