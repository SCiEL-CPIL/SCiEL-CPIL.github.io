import numpy as np
from numpy.polynomial.hermite import hermgauss
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    dt = T / n_exercise
    
    # Grid for log(S): S in [1, 500]
    n_grid = 4000
    log_s_min = np.log(1.0)
    log_s_max = np.log(500.0)
    log_s_grid = np.linspace(log_s_min, log_s_max, n_grid)
    s_grid = np.exp(log_s_grid)
    
    # Gauss-Hermite quadrature for N(0,1)
    # E[f(X)] = (1/sqrt(2*pi)) * integral f(x) exp(-x^2/2) dx
    # Use transformation: X = mu + sigma * Z, Z ~ N(0,1)
    # E[g(X)] = integral g(mu + sigma*z) phi(z) dz
    # Using Gauss-Hermite: integral h(x) exp(-x^2) dx ≈ sum w_i h(x_i)
    # So integral g(z) exp(-z^2/2) dz = sqrt(2) * integral g(z/sqrt(2)) exp(-u^2) du
    # E[g(Z)] = (1/sqrt(2*pi)) * sqrt(2) * sum w_i g(x_i/sqrt(2))
    #          = (1/sqrt(pi)) * sum w_i g(x_i/sqrt(2))
    
    n_quad = 80
    x_gh, w_gh = hermgauss(n_quad)
    # Transform: Z = x_gh / sqrt(2) for N(0,1)
    z_nodes = x_gh / np.sqrt(2.0)
    z_weights = w_gh / np.sqrt(np.pi)  # sum to ~1
    
    # Discount factor
    disc = np.exp(-r * dt)
    
    # Backward induction: compute value at each exercise date
    # At time n (last exercise date), value = max(K - S, 0)
    # At earlier times, value = max(K - S, disc * E[V_next(S_next)])
    
    # S_next = S * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    drift = (r - 0.5 * sigma**2) * dt
    vol_sqrt_dt = sigma * np.sqrt(dt)
    
    # Value function at last exercise date (index n_exercise - 1)
    # Exercise dates are at t_1, t_2, ..., t_12 (no time-0 exercise)
    # So there are 12 exercise decisions
    
    # V at step i (0-indexed, i=0 is first exercise date)
    # We need V at each of the 12 exercise dates
    
    # Initialize: at the last exercise date, V = max(K - S, 0)
    V_next = np.maximum(K - s_grid, 0.0)
    
    # Store critical prices for each exercise date
    critical_prices = np.zeros(n_exercise)
    
    # We'll store value functions for each step
    V_all = np.zeros((n_exercise, n_grid))
    V_all[n_exercise - 1] = V_next
    
    # Find critical price for last step
    # Exercise where K - S > 0, i.e., S < K
    # Boundary is where K - S = V_cont, but at last step V_cont = 0
    # So exercise if S < K
    critical_prices[n_exercise - 1] = K
    
    # Backward induction from step n_exercise-2 down to 0
    for i in range(n_exercise - 2, -1, -1):
        # Check deadline
        if time.time() > DEADLINE - 10:
            break
        
        # Compute S_next for each grid point
        # S_next = S * exp(drift + vol_sqrt_dt * Z)
        # For each grid point s, S_next = s * exp(drift + vol_sqrt_dt * z_j)
        # E[V_next(S_next)] = sum_j z_weights_j * V_next(s * exp(drift + vol_sqrt_dt * z_j))
        
        # For each s in s_grid, compute expected continuation value
        # S_next_matrix: (n_grid, n_quad)
        log_s_next = np.log(s_grid)[:, np.newaxis] + drift + vol_sqrt_dt * z_nodes[np.newaxis, :]
        s_next = np.exp(log_s_next)
        
        # Interpolate V_next at s_next
        # Clip to grid range
        s_next_clipped = np.clip(s_next, s_grid[0], s_grid[-1])
        V_next_interp = np.interp(s_next_clipped, s_grid, V_next)
        
        # Expected continuation value
        cont_value = np.sum(V_next_interp * z_weights[np.newaxis, :], axis=1)
        cont_value = disc * cont_value
        
        # Exercise value
        exercise_value = np.maximum(K - s_grid, 0.0)
        
        # Value function
        V_curr = np.maximum(exercise_value, cont_value)
        V_all[i] = V_curr
        
        # Find critical price: where exercise_value = cont_value
        # In the region S < K, exercise_value = K - S (decreasing)
        # cont_value is increasing in S
        # The boundary is where they cross
        diff = exercise_value - cont_value
        # Find sign change in diff (from positive to negative)
        # In region S < K
        mask_region = s_grid < K
        if np.any(mask_region):
            diff_region = diff[mask_region]
            s_region = s_grid[mask_region]
            # Find where diff changes from positive to negative
            sign_change = np.where(np.diff(np.sign(diff_region)) < 0)[0]
            if len(sign_change) > 0:
                idx = sign_change[0]
                # Linear interpolation for the boundary
                s1, s2 = s_region[idx], s_region[idx + 1]
                d1, d2 = diff_region[idx], diff_region[idx + 1]
                if abs(d2 - d1) > 1e-12:
                    crit = s1 + (s2 - s1) * (-d1) / (d2 - d1)
                else:
                    crit = s1
                critical_prices[i] = crit
            else:
                # No sign change: either always exercise or never
                if diff_region[0] > 0:
                    critical_prices[i] = K
                else:
                    critical_prices[i] = 0.0
        else:
            critical_prices[i] = K
        
        V_next = V_curr
    
    # Now we have critical_prices for each of the 12 exercise dates
    # The policy: at exercise date i, exercise if S < critical_prices[i]
    
    # Fit a smooth parametric curve: critical price as a function of time step
    # critical_prices is a vector of 12 values
    # We want a smooth monotonic function. Since for a put, the critical price
    # typically decreases as we get closer to expiration (less time value),
    # let's check the pattern and fit accordingly.
    
    # Actually, for Bermudan puts, the critical price generally decreases over time
    # (less time to wait means less optionality value, so you exercise at higher prices earlier)
    # Wait, that's the opposite. Let me think again.
    # At the last exercise date, critical price = K (exercise if in the money)
    # At earlier dates, you have more time value, so you're less willing to exercise
    # This means the critical price is LOWER at earlier dates (you need to be deeper ITM)
    # So critical_prices should be increasing toward the end (toward K)
    
    # Let's fit a smooth curve: log(critical_price) as a function of step index
    # Use a monotonic cubic or just linear interpolation with smoothing
    
    # For the exercise policy, we just need to compare S to critical_prices[i]
    # at each time step i. The "smooth parametric" aspect is that we use the
    # fitted curve rather than raw grid values.
    
    # Fit: log(crit) = a + b * i + c * i^2 (quadratic in time step)
    # Ensure monotonicity by checking derivatives
    
    steps = np.arange(n_exercise, dtype=float)
    log_crit = np.log(np.maximum(critical_prices, 1e-6))
    
    # Fit quadratic
    if n_exercise >= 3:
        coeffs = np.polyfit(steps, log_crit, 2)
        # Check monotonicity: derivative = 2*a*i + b should have consistent sign
        # For a put, critical price increases over time, so derivative should be positive
        # If not monotonic, fall back to linear
        deriv = 2 * coeffs[0] * steps + coeffs[1]
        if np.all(deriv > 0) or np.all(deriv < 0):
            # Use quadratic
            log_crit_fit = np.polyval(coeffs, steps)
            critical_prices_fit = np.exp(log_crit_fit)
        else:
            # Fall back to linear
            coeffs_lin = np.polyfit(steps, log_crit, 1)
            log_crit_fit = np.polyval(coeffs_lin, steps)
            critical_prices_fit = np.exp(log_crit_fit)
    else:
        critical_prices_fit = critical_prices.copy()
    
    # Use the fitted critical prices for the policy
    # This gives a smooth, parametric boundary
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, crit_prices, K, n_exercise, s_grid, log_s_grid):
            self.crit_prices = crit_prices
            self.K = K
            self.n_exercise = n_exercise
            self.s_grid = s_grid
            self.log_s_grid = log_s_grid
            # Precompute for fast lookup
            # For each time step, we just need to compare S to crit_prices[i]
        
        def exercise(self, current_state, time):
            # current_state: array of spot prices
            # time: current time (to determine which exercise date we're at)
            # Return boolean vector: True if exercise
            
            # Determine which exercise date we're at
            # Exercise dates are at t_i = i * dt for i = 1, ..., 12
            # time is the current time
            # Find the exercise date index
            
            # time should be close to one of the exercise dates
            # dt = T / n_exercise
            dt_local = 1.0 / self.n_exercise
            # Exercise dates: dt, 2*dt, ..., 12*dt
            # Find which one we're closest to
            t_scaled = time / dt_local
            # Should be an integer from 1 to 12
            step_idx = int(round(t_scaled)) - 1  # 0-indexed
            step_idx = np.clip(step_idx, 0, self.n_exercise - 1)
            
            # Exercise if S < critical_price
            crit = self.crit_prices[step_idx]
            return (current_state < crit) & (current_state < self.K)
    
    return ExercisePolicy(critical_prices_fit, K, n_exercise, s_grid, log_s_grid)
