"""Bermudan exercise-policy research: backward induction with grid-based boundaries."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction.
    
    Args:
        training_paths: array of shape (n_paths, n_times) with simulated asset prices
        contract: dict with contract parameters
        
    Returns:
        An object with exercise(current_state, time) method returning boolean vector.
    """
    start_time = time.time()
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise times: no time-0 exercise, so at T/n_exercise, 2T/n_exercise, ..., T
    dt = T / n_exercise
    exercise_times = np.array([(i + 1) * dt for i in range(n_exercise)])
    
    # Build a fine grid for backward induction
    # For a put with K=100, relevant S range is roughly [0, 1.5*K] (puts are worthless above K)
    S_max = 1.5 * K
    n_grid = 4000  # fine grid for accuracy
    
    S_grid = np.linspace(0.0, S_max, n_grid)
    
    # Backward induction: compute continuation value at each exercise date
    # V[i] = value at exercise date i (i=0 is last exercise at T, i=n_exercise-1 is first)
    # We go backward from maturity
    
    # At maturity (last exercise date), the put payoff is max(K - S, 0)
    # But wait - the last exercise is at T, so at T the value is the payoff
    # Actually, for Bermudan, the last exercise date IS at T, so V at T is max(K-S, 0)
    
    # Let's define: V_j(S) = optimal value at exercise date j (j=0,...,n_exercise-1)
    # where j=0 is the first exercise date (t=dt) and j=n_exercise-1 is the last (t=T)
    
    # Backward induction from j=n_exercise-1 (last, at T) to j=0 (first, at dt)
    
    # At the last exercise date (T), value = payoff = max(K - S, 0)
    V = np.maximum(K - S_grid, 0.0)
    
    # For j from n_exercise-2 down to 0:
    # V_j(S) = max(K - S, exp(-r*dt) * E[V_{j+1}(S_{next}) | S])
    
    # Under risk-neutral GBM:
    # S_next = S * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    # E[V_{j+1}(S_next) | S] needs to be computed
    
    # For efficiency, precompute the transition
    sqrt_dt = np.sqrt(dt)
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * sqrt_dt
    
    # For each S in grid, compute E[V_{j+1}(S_next)]
    # S_next = S * exp(drift + vol * Z)
    # We can use a Gaussian quadrature or Monte Carlo
    
    # Using Gauss-Hermite quadrature for speed and accuracy
    # E[f(Z)] where Z ~ N(0,1) = (1/sqrt(pi)) * sum w_i f(sqrt(2)*x_i)
    from scipy.special import roots_hermite
    
    # Precompute quadrature points and weights - use more points for accuracy
    n_quad = 80
    x_quad, w_quad = roots_hermite(n_quad)
    # Transform: Z = sqrt(2)*x, so E[f(Z)] = (1/sqrt(pi)) * sum w_i f(sqrt(2)*x_i)
    Z_quad = np.sqrt(2.0) * x_quad
    w_quad_norm = w_quad / np.sqrt(np.pi)
    
    # Precompute exp(drift + vol * Z_quad)
    exp_factor = np.exp(drift + vol * Z_quad)  # shape (n_quad,)
    
    # Now do backward induction, storing exercise decisions directly
    V = np.maximum(K - S_grid, 0.0)  # Start with maturity payoff
    exercise_decisions = []  # List of boolean arrays, one per exercise date (in reverse order)
    
    for j in range(n_exercise - 2, -1, -1):
        chunk_size = 400
        exercise_here = np.zeros(n_grid, dtype=bool)
        
        for chunk_start in range(0, n_grid, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_grid)
            S_chunk = S_grid[chunk_start:chunk_end]  # (chunk_size,)
            
            # S_next for each S in chunk and each quadrature point
            S_next = S_chunk[:, None] * exp_factor[None, :]  # (chunk_size, n_quad)
            
            # Clip to grid range
            S_next_clipped = np.clip(S_next, 0.0, S_max)
            
            # Interpolate V at S_next_clipped
            V_interp = np.interp(S_next_clipped.ravel(), S_grid, V).reshape(chunk_size, n_quad)
            
            # Expected continuation value
            expected_cont = V_interp @ w_quad_norm  # (chunk_size,)
            
            # Discounted continuation value
            disc_cont = np.exp(-r * dt) * expected_cont
            
            # Exercise value
            exercise_val = np.maximum(K - S_chunk, 0.0)
            
            # Exercise if exercise_val >= disc_cont
            ex_here = exercise_val >= disc_cont
            exercise_here[chunk_start:chunk_end] = ex_here
            
            V[chunk_start:chunk_end] = np.maximum(exercise_val, disc_cont)
        
        exercise_decisions.append(exercise_here)
        
        # Check time with safety margin
        if time.time() - start_time > 130:
            break
    
    # exercise_decisions[j] corresponds to exercise date j (in reverse: last appended is j=0)
    # Actually, we appended in order j=n_exercise-2, n_exercise-3, ..., 0
    # So exercise_decisions[0] is for j=n_exercise-2, exercise_decisions[-1] is for j=0
    # Let's reverse: exercise_decisions_reversed[i] is for exercise date i (0-indexed from first)
    exercise_decisions = exercise_decisions[::-1]
    
    # Now create the policy object
    class ExercisePolicy:
        def __init__(self, S_grid, exercise_decisions, K, r, dt, n_exercise):
            self.S_grid = S_grid
            self.exercise_decisions = exercise_decisions
            self.K = K
            self.r = r
            self.dt = dt
            self.n_exercise = n_exercise
            self.T = n_exercise * dt
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time and state.
            
            Args:
                current_state: array of asset prices
                time: current time (float)
                
            Returns:
                Boolean array: True if should exercise
            """
            # Determine which exercise date this corresponds to
            # Exercise dates are at dt, 2*dt, ..., n_exercise*dt
            # time should be close to one of these
            
            # Find the exercise index
            # t_i = (i+1) * dt for i=0,...,n_exercise-1
            # So i = round(time/dt) - 1
            
            if time <= 0:
                # No exercise at time 0
                return np.zeros(len(current_state), dtype=bool)
            
            idx = int(round(time / self.dt)) - 1
            
            if idx < 0 or idx >= self.n_exercise:
                # Invalid time, don't exercise
                return np.zeros(len(current_state), dtype=bool)
            
            # Get exercise decisions for this date
            if idx >= len(self.exercise_decisions):
                # Not enough backward induction done, use simple rule
                # Exercise if in-the-money and near boundary
                return current_state <= self.K * 0.85
            
            decisions = self.exercise_decisions[idx]
            
            # For each state, find the corresponding grid index and check decision
            # Use interpolation for sub-grid precision on the boundary
            
            if not np.any(decisions):
                return np.zeros(len(current_state), dtype=bool)
            
            if np.all(decisions):
                return np.ones(len(current_state), dtype=bool)
            
            # Find the boundary with sub-grid precision
            # The exercise region is S <= B where B is the boundary
            true_indices = np.where(decisions)[0]
            false_indices = np.where(~decisions)[0]
            
            if len(true_indices) == 0:
                return np.zeros(len(current_state), dtype=bool)
            if len(false_indices) == 0:
                return np.ones(len(current_state), dtype=bool)
            
            # Find the transition point: last True and first False
            last_true = true_indices[-1]
            first_false = false_indices[0]
            
            # If they're adjacent, interpolate between them
            if first_false == last_true + 1:
                S_true = self.S_grid[last_true]
                S_false = self.S_grid[first_false]
                # Linear interpolation: boundary is where exercise_val = disc_cont
                # We can estimate it as the midpoint or use linear interpolation
                S_boundary = (S_true + S_false) / 2.0
            else:
                # There's a gap, use the last True
                S_boundary = self.S_grid[last_true]
            
            # Exercise if current_state <= S_boundary
            return current_state <= S_boundary
    
    return ExercisePolicy(S_grid, exercise_decisions, K, r, dt, n_exercise)
# EVOLVE-BLOCK-END
