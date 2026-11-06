"""Bermudan exercise-policy: Exact backward induction on a discretized spot grid for GBM Bermudan put."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import quad

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    n_steps = n_dates

    # Exercise times: T/12, 2T/12, ..., T
    dt = T / n_steps
    exercise_times = np.linspace(dt, T, n_steps)

    # Discretize spot price grid
    # Grid should cover reasonable range for a put with K=100
    n_grid = 800
    S_min = 0.01 * K
    S_max = 5.0 * K
    S_grid = np.linspace(S_min, S_max, n_grid)
    dS = S_grid[1] - S_grid[0]

    # Precompute log of grid for efficiency
    log_S_grid = np.log(S_grid)

    # Backward induction
    # V[i] = max(K - S, exp(-r*dt) * E[V[i+1](S') | S])
    # At final step: V[n-1] = max(K - S, 0)
    
    # Start from maturity
    V_next = np.maximum(K - S_grid, 0.0)

    # Store thresholds for each exercise date
    # threshold[i] = spot price below which we exercise at date i
    thresholds = np.zeros(n_steps)

    # For the final date, exercise if K - S > 0, i.e., S < K
    # But we also need to compare with continuation (which is 0 at maturity)
    # So at maturity, exercise iff S < K
    thresholds[-1] = K

    # Backward induction from step n-2 down to 0
    for i in range(n_steps - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 10.0:
            # Fallback: use previous threshold
            thresholds[i] = thresholds[i + 1] if i + 1 < n_steps else K
            continue

        # For each spot price S_j, compute E[V_next(S') | S_j]
        # Under risk-neutral GBM: S' = S_j * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        # where Z ~ N(0,1)
        # E[V_next(S') | S_j] = integral V_next(S_j * exp((r-0.5*sig^2)*dt + sig*sqrt(dt)*z)) * phi(z) dz
        
        # Use numerical integration. Since V_next is piecewise linear (from max), 
        # we can use quadrature on the grid.
        
        # For efficiency, use the fact that S' is lognormal given S_j
        # E[V_next(S') | S_j] = integral V_next(S') f(S'|S_j) dS'
        # where f(S'|S_j) is the lognormal density
        
        # Alternative: use Gauss-Hermite quadrature on the lognormal
        # Let Y = log(S'), then Y ~ N(log(S_j) + (r-0.5*sig^2)*dt, sig^2*dt)
        # E[V_next(S') | S_j] = E[V_next(exp(Y)) | S_j]
        
        # Use 64-point Gauss-Hermite quadrature
        # We need to be careful with the integration range
        
        mu_j = log_S_grid[i] if False else None  # placeholder
        
        # Actually, let's vectorize: for each S_j in the grid, compute the expectation
        # This is O(n_grid * n_quadrature) which is 800 * 64 = 51200 evaluations, fast
        
        # Gauss-Hermite nodes and weights
        # np.polynomial.hermite.hermgauss(n) returns nodes x_i and weights w_i
        # for integral exp(-x^2) f(x) dx
        # We need integral exp(-(z - mu/sigma)^2 / 2) f(z) dz which is different
        
        # Better: use scipy.integrate.quad_vec or just do it on the grid with trapezoidal rule
        
        # Since S_grid is fine enough (800 points over [0.1K, 5K]), we can use 
        # the grid to approximate the expectation via the lognormal CDF
        
        # For each S_j, S' = S_j * exp(m + s*Z) where m=(r-0.5*sig^2)*dt, s=sig*sqrt(dt)
        # E[V_next(S') | S_j] = integral V_next(S_j * exp(m + s*z)) * phi(z) dz
        
        # Use 128-point Gauss-Hermite on standard normal
        # Transform: z = (x + mu_z)/sigma_z where mu_z=0, sigma_z=1 for standard normal
        # Actually for standard normal: integral f(z) phi(z) dz = (1/sqrt(2pi)) integral f(z) exp(-z^2/2) dz
        # Use Gauss-Hermite: integral exp(-x^2) g(x) dx = sum w_i g(x_i)
        # Let x = z/sqrt(2), so z = x*sqrt(2), dz = sqrt(2) dx
        # integral exp(-z^2/2) f(z) dz = integral exp(-x^2) f(x*sqrt(2)) * sqrt(2) dx
        # So E[f(Z)] = (1/sqrt(2pi)) * sqrt(2) * sum w_i f(x_i * sqrt(2)) = (1/sqrt(pi)) * sum w_i f(x_i * sqrt(2))
        
        n_quad = 64
        nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
        # nodes are in [-inf, inf] for exp(-x^2) integral
        # z = node * sqrt(2)
        z_nodes = nodes * np.sqrt(2)
        z_weights = weights / np.sqrt(np.pi)  # normalized so sum(z_weights) = 1
        
        m = (r - 0.5 * sigma**2) * dt
        s = sigma * np.sqrt(dt)
        
        # For each S_j, compute S' = S_j * exp(m + s * z_node)
        # V_next is defined on S_grid, so we need to interpolate
        
        # Create interpolation function for V_next
        # V_next is piecewise linear on S_grid
        def interp_V(S_eval):
            """Interpolate V_next at arbitrary S values using linear interpolation on S_grid."""
            # Clip to grid range
            S_eval = np.clip(S_eval, S_min, S_max)
            # Find indices
            idx = np.searchsorted(S_grid, S_eval)
            idx = np.clip(idx, 1, n_grid - 1)
            S0_idx = S_grid[idx - 1]
            S1_idx = S_grid[idx]
            V0 = V_next[idx - 1]
            V1 = V_next[idx]
            frac = (S_eval - S0_idx) / (S1_idx - S0_idx + 1e-30)
            return V0 + frac * (V1 - V0)
        
        # Compute E[V_next(S') | S_j] for all S_j in the grid
        # Vectorized: S_grid[:, None] * exp(m + s * z_nodes[None, :])
        S_prime = S_grid[:, None] * np.exp(m + s * z_nodes[None, :])
        V_prime = interp_V(S_prime)
        E_cont = np.sum(V_prime * z_weights[None, :], axis=1)
        
        # Discounted continuation value
        disc = np.exp(-r * dt)
        cont_value = disc * E_cont
        
        # Immediate exercise payoff
        ex_payoff = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V_curr = np.maximum(ex_payoff, cont_value)
        
        # Determine threshold: the largest S where ex_payoff >= cont_value
        # For a put, ex_payoff is decreasing in S, cont_value is increasing in S
        # So the threshold is where they cross
        diff = ex_payoff - cont_value
        # Find where diff changes from positive to negative
        # diff should be positive for small S, negative for large S
        positive_mask = diff >= 0
        if np.any(positive_mask):
            last_positive = np.where(positive_mask)[0][-1]
            # Interpolate to find exact crossing
            if last_positive < n_grid - 1:
                # Linear interpolation between last_positive and last_positive+1
                d0 = diff[last_positive]
                d1 = diff[last_positive + 1]
                if d0 != d1:
                    frac = d0 / (d0 - d1)
                    thresholds[i] = S_grid[last_positive] + frac * (S_grid[last_positive + 1] - S_grid[last_positive])
                else:
                    thresholds[i] = S_grid[last_positive]
            else:
                thresholds[i] = S_grid[last_positive]
        else:
            # Never exercise (shouldn't happen for a put)
            thresholds[i] = S_min
        
        V_next = V_curr

    # Create the exercise policy class
    class BermudanExercisePolicy:
        def __init__(self, thresholds, n_steps, dt, K):
            self.thresholds = thresholds
            self.n_steps = n_steps
            self.dt = dt
            self.K = K

        def exercise(self, current_state, time):
            """
            current_state: array of stock prices
            time: current time
            Returns: boolean array, True if exercise
            """
            if time is None or time <= 0:
                return np.zeros(len(current_state), dtype=bool)

            # Determine step index
            step_idx = int(round(time / self.dt)) - 1
            step_idx = max(0, min(step_idx, self.n_steps - 1))

            threshold = self.thresholds[step_idx]
            # Exercise if current spot is below the threshold
            return current_state < threshold

    policy = BermudanExercisePolicy(thresholds, n_steps, dt, K)
    return policy
# EVOLVE-BLOCK-END
