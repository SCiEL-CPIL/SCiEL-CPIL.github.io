"""Bermudan put exercise policy: backward induction on log-grid with smooth exercise boundary."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class ExercisePolicy:
    def __init__(self, cont_values, log_grid, step, K, n_grid, smoothness=0.0):
        self.cont_values = cont_values  # List of arrays, one per exercise date (dates 1..N)
        self.log_grid = log_grid
        self.step = step
        self.K = K
        self.n_grid = n_grid
        self.smoothness = smoothness  # If > 0, use soft decision

    def exercise(self, current_state, time_idx):
        """
        Determine exercise decision for given states at a specific time index.
        
        Parameters:
        -----------
        current_state : array-like
            Spot prices for a batch of paths.
        time_idx : int
            The exercise date index (1-based, 1..12). If 0 or >N, never exercise.
            
        Returns:
        --------
        boolean array
            True if exercise is optimal for that state/time.
        """
        if time_idx <= 0 or time_idx > len(self.cont_values):
            return np.zeros(len(current_state), dtype=bool)
        
        states = np.asarray(current_state, dtype=np.float64)
        log_states = np.log(np.maximum(states, 1e-15))
        
        # Interpolate continuation value
        cont = self.cont_values[time_idx - 1]
        
        idx_float = (log_states - self.log_grid[0]) / self.step
        idx_float = np.clip(idx_float, 0, self.n_grid - 1)
        
        idx0 = idx_float.astype(np.int64)
        idx1 = np.minimum(idx0 + 1, self.n_grid - 1)
        idx0 = np.clip(idx0, 0, self.n_grid - 1)
        
        t = idx_float - idx0
        
        c0 = cont[idx0]
        c1 = cont[idx1]
        
        continuation = c0 * (1.0 - t) + c1 * t
        intrinsic = np.maximum(self.K - states, 0.0)
        
        if self.smoothness > 0:
            # Soft decision: use smooth approximation
            # The difference d = intrinsic - continuation
            # Soft max: we want True when d > 0
            # Use: sigmoid(d / smoothness) > 0.5, which is equivalent to d > 0
            # But for a smoother boundary that might help MC estimation:
            # Actually, the boolean output is what matters. Let's just use the hard decision
            # but with a tiny hysteresis to avoid boundary noise
            d = intrinsic - continuation
            # Add tiny epsilon to break ties toward not exercising (conservative)
            return d > 1e-10
        else:
            return intrinsic > continuation


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a log-grid.
    
    Parameters:
    -----------
    training_paths : array, shape (n_paths, n_timepoints)
        Training price paths (not used for fitting; we solve analytically).
    contract : dict
        Contract parameters: K, r, sigma, T, n_steps, etc.
        
    Returns:
    --------
    ExercisePolicy object with .exercise(current_state, time_idx) method.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Extract contract parameters
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_steps = int(contract.get('n_steps', 12))

    dt = T / n_steps
    disc = np.exp(-r * dt)
    mu = r - 0.5 * sigma * sigma
    vol_step = sigma * np.sqrt(dt)

    # Grid construction: log-uniform grid
    min_s = 0.3
    max_s = 800.0
    
    n_grid = 4096
    log_min = np.log(min_s)
    log_max = np.log(max_s)
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)
    step = log_grid[1] - log_grid[0]

    # Quadrature: Gauss-Hermite with 64 points
    n_quad = 64
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    z_quad = np.sqrt(2.0) * x_gh
    w_quad = w_gh / np.sqrt(np.pi)

    # Backward induction
    cont_values = [None] * n_steps
    
    # At the last exercise date (date n_steps), continuation = 0
    cont_values[n_steps - 1] = np.zeros(n_grid)
    
    # Value at the last date is just the payoff
    V_next = np.maximum(K - grid, 0.0)
    
    shift = mu * dt
    
    for k in range(n_steps - 1, 0, -1):
        if time.time() > deadline:
            for rem_k in range(k - 1, 0, -1):
                cont_values[rem_k - 1] = np.zeros(n_grid)
            break
        
        log_next = log_grid[:, np.newaxis] + shift + vol_step * z_quad[np.newaxis, :]
        log_next_flat = log_next.ravel()
        
        idx_float = (log_next_flat - log_grid[0]) / step
        idx_float = np.clip(idx_float, 0.0, float(n_grid - 1))
        
        idx0 = idx_float.astype(np.int64)
        idx1 = np.minimum(idx0 + 1, n_grid - 1)
        idx0 = np.clip(idx0, 0, n_grid - 1)
        
        t = idx_float - idx0
        
        y0 = V_next[idx0]
        y1 = V_next[idx1]
        
        interp_vals = y0 * (1.0 - t) + y1 * t
        interp_vals = interp_vals.reshape(n_grid, n_quad)
        
        E_val = np.dot(interp_vals, w_quad)
        Cont = disc * E_val
        
        cont_values[k - 1] = Cont
        
        Payoff = np.maximum(K - grid, 0.0)
        V_next = np.maximum(Payoff, Cont)

    # Ensure all cont_values are filled
    for i in range(n_steps):
        if cont_values[i] is None:
            cont_values[i] = np.zeros(n_grid)

    # Refinement pass: recompute on a denser sub-grid in the critical low-S region
    # The exercise boundary for a put is typically around S/K ~ 0.5-0.8
    # Refine the region from min_s to 2*K with a denser grid
    if time.time() < deadline - 10.0:
        refine_min = 0.2
        refine_max = 2.0 * K
        n_refine = 4096  # Same resolution but focused
        log_ref_min = np.log(refine_min)
        log_ref_max = np.log(refine_max)
        log_grid_ref = np.linspace(log_ref_min, log_ref_max, n_refine)
        grid_ref = np.exp(log_grid_ref)
        step_ref = log_grid_ref[1] - log_grid_ref[0]
        
        # Re-run backward induction on the refined grid
        V_next_ref = np.maximum(K - grid_ref, 0.0)
        cont_values_ref = [None] * n_steps
        cont_values_ref[n_steps - 1] = np.zeros(n_refine)
        
        for k in range(n_steps - 1, 0, -1):
            if time.time() > deadline:
                break
            
            log_next_ref = log_grid_ref[:, np.newaxis] + shift + vol_step * z_quad[np.newaxis, :]
            log_next_ref_flat = log_next_ref.ravel()
            
            idx_float_ref = (log_next_ref_flat - log_grid_ref[0]) / step_ref
            idx_float_ref = np.clip(idx_float_ref, 0.0, float(n_refine - 1))
            
            idx0_ref = idx_float_ref.astype(np.int64)
            idx1_ref = np.minimum(idx0_ref + 1, n_refine - 1)
            idx0_ref = np.clip(idx0_ref, 0, n_refine - 1)
            
            t_ref = idx_float_ref - idx0_ref
            
            y0_ref = V_next_ref[idx0_ref]
            y1_ref = V_next_ref[idx1_ref]
            
            interp_vals_ref = y0_ref * (1.0 - t_ref) + y1_ref * t_ref
            interp_vals_ref = interp_vals_ref.reshape(n_refine, n_quad)
            
            E_val_ref = np.dot(interp_vals_ref, w_quad)
            Cont_ref = disc * E_val_ref
            
            cont_values_ref[k - 1] = Cont_ref
            
            Payoff_ref = np.maximum(K - grid_ref, 0.0)
            V_next_ref = np.maximum(Payoff_ref, Cont_ref)
        
        # Use the refined continuation values if all were computed
        if all(c is not None for c in cont_values_ref):
            cont_values = cont_values_ref
            log_grid = log_grid_ref
            step = step_ref
            n_grid = n_refine

    return ExercisePolicy(cont_values, log_grid, step, K, n_grid, smoothness=0.0)

# EVOLVE-BLOCK-END
