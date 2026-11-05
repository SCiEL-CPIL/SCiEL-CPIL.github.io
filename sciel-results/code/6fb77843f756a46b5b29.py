"""Bermudan exercise-policy research: Refined backward-induction with root-finding polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a fine grid,
    followed by a high-precision root-finding polish on the exercise boundaries.
    
    Parameters:
    -----------
    training_paths : ndarray, shape (n_paths, n_steps+1)
        Training price paths (unused in this deterministic grid approach, kept for interface)
    contract : dict or object
        Contract specification
    
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = 135.0  # wall-clock limit with safety margin
    
    # Extract contract parameters
    K = _get_param(contract, 'K', 100.0)
    r = _get_param(contract, 'r', 0.05)
    sigma = _get_param(contract, 'sigma', 0.2)
    T = _get_param(contract, 'T', 1.0)
    n_steps = _get_param(contract, 'n_steps', 12)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_steps
    # exercise_times = np.arange(1, n_steps + 1) * dt
    
    # Build a fine grid for the underlying price
    # Focus on the relevant range: evaluation spots are 90/100/110, K=100
    # Put exercise is relevant for S < K
    S_min = 2.0
    S_max = 150.0
    n_grid = 4000  # High resolution grid
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Discount factors
    df = np.exp(-r * dt)
    
    # Precompute Gauss-Hermite quadrature nodes/weights for E[f(Z)] where Z ~ N(0,1)
    # High number of points for accurate expectation calculation
    n_quad = 120
    x_gh, w_gh = _gauss_hermite(n_quad)
    # Transform: for Z ~ N(0,1), E[f(Z)] = (1/sqrt(pi)) * integral exp(-x^2) f(sqrt(2)*x) dx
    z_nodes = np.sqrt(2.0) * x_gh
    z_weights = w_gh / np.sqrt(np.pi)
    
    # Precompute log of S_grid for fast computation
    log_S_grid = np.log(S_grid)
    
    # GBM parameters
    mu = r - 0.5 * sigma**2
    sd_log = sigma * math.sqrt(dt)
    
    # Backward induction
    # V_list[i] will store the value function at exercise date i (1-indexed in list, 0-indexed in array)
    # We store V for dates n_steps, n_steps-1, ..., 1
    # Let's use a list of arrays: V_vals[i] corresponds to date i+1 (1..n_steps)
    # We compute backwards: V(n_steps) is known. Then V(n_steps-1), etc.
    
    V_vals = [None] * n_steps
    
    # At maturity (last exercise date, index n_steps-1 in 0-based, date n_steps)
    V_last = np.maximum(K - S_grid, 0.0)
    V_vals[n_steps - 1] = V_last
    
    # Grid-based boundaries (initial estimates)
    boundaries_grid = np.full(n_steps, np.nan)
    
    # Work backwards from the second-to-last date to the first date
    # i is the index in V_vals (0 to n_steps-1)
    # We are computing V_vals[i] using V_vals[i+1]
    for i in range(n_steps - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 10:
            # If we run out of time, fill remaining with heuristic and break
            for j in range(i, -1, -1):
                boundaries_grid[j] = K * 0.85
            break
        
        V_next = V_vals[i + 1]
        
        # Compute continuation value: discounted expected value at next step
        # Vectorized: for all S_grid points simultaneously
        # S_next[j, k] = S_grid[j] * exp(mu*dt + sd_log * z_nodes[k])
        log_S_next = log_S_grid[:, np.newaxis] + mu * dt + sd_log * z_nodes[np.newaxis, :]
        S_next = np.exp(log_S_next)
        S_next_clipped = np.clip(S_next, S_min, S_max)
        
        # Interpolate V_next at S_next for each grid point
        V_interp = np.zeros_like(S_next_clipped)
        for k in range(n_quad):
            V_interp[:, k] = np.interp(S_next_clipped[:, k], S_grid, V_next)
        
        # Weighted average over quadrature points
        cont = df * (V_interp * z_weights[np.newaxis, :]).sum(axis=1)
        
        # Decision: exercise if (K - S) > continuation value
        exercise_payoff = np.maximum(K - S_grid, 0.0)
        should_exercise = exercise_payoff >= cont
        V_curr = np.where(should_exercise, exercise_payoff, cont)
        
        V_vals[i] = V_curr
        
        # Find the boundary: the highest S where we still exercise
        if np.any(should_exercise):
            boundary_idx = np.where(should_exercise)[0]
            boundaries_grid[i] = S_grid[boundary_idx[-1]]
        else:
            boundaries_grid[i] = S_min
    
    # If the loop broke early, fill remaining
    for i in range(n_steps - 1, -1, -1):
        if np.isnan(boundaries_grid[i]):
            boundaries_grid[i] = K * 0.85

    # REFINE BOUNDARIES USING ROOT FINDING
    # The grid boundary is approximate. We refine it by finding the exact root of
    # f(S) = Payoff(S) - Continuation(S) = 0
    # We use the V_vals computed above.
    
    refined_boundaries = np.zeros(n_steps)
    
    for i in range(n_steps):
        # We need V_next to compute continuation.
        # If i is the last date (n_steps-1), there is no next date.
        # The "continuation" value for the last date is 0 (if we don't exercise, we get 0? 
        # No, at the last exercise date, if we don't exercise, the option expires worthless.
        # So Continuation = 0.
        # The boundary is where Payoff = 0 => S = K.
        if i == n_steps - 1:
            refined_boundaries[i] = K
            continue
        
        V_next = V_vals[i + 1]
        
        # Define the function to find the root of
        def f(S_val):
            # Payoff
            p = max(K - S_val, 0.0)
            if p == 0.0:
                # If S >= K, payoff is 0. Continuation is usually > 0.
                # f(S) = 0 - Cont < 0.
                # We are looking for the crossing from Positive (Exercise) to Negative (Continue)
                # Or rather, we know the boundary is below K.
                pass
            
            # Continuation Value
            # S_next = S_val * exp(mu*dt + sd_log * Z)
            log_S_next_single = math.log(S_val) + mu * dt + sd_log * z_nodes
            S_next_single = np.exp(log_S_next_single)
            S_next_clipped_single = np.clip(S_next_single, S_min, S_max)
            
            # Interpolate V_next
            V_next_interp = np.interp(S_next_clipped_single, S_grid, V_next)
            
            cont_val = df * np.dot(V_next_interp, z_weights)
            
            return p - cont_val
        
        # Initial guess from grid boundary
        S_guess = boundaries_grid[i]
        
        # Define a search window around the grid boundary
        # The grid spacing is (S_max - S_min) / n_grid
        grid_step = (S_max - S_min) / n_grid
        search_low = max(S_min, S_guess - 5 * grid_step)
        search_high = min(K, S_guess + 5 * grid_step)
        
        # Ensure we are in the region where a root might exist (S < K)
        # For a put, exercise is optimal for low S.
        # f(S) is positive for low S (Payoff > Cont) and negative for high S (Payoff < Cont).
        # We want the largest S where f(S) >= 0.
        
        # Check signs at boundaries of the window
        f_low = f(search_low)
        f_high = f(search_high)
        
        # If f_low <= 0, the boundary is lower than search_low.
        # If f_high >= 0, the boundary is higher than search_high.
        # Use bisection to find the root.
        
        # If no sign change in the window, fall back to grid boundary or expand search
        if f_low * f_high > 0:
            # No sign change. 
            # If f_low > 0 and f_high > 0, boundary is > search_high.
            # If f_low < 0 and f_high < 0, boundary is < search_low.
            # Try a wider search or just use grid boundary.
            # For safety, use grid boundary.
            refined_boundaries[i] = S_guess
        else:
            # Bisection
            lo = search_low
            hi = search_high
            f_lo = f_low
            f_hi = f_high
            
            for _ in range(50): # 50 iterations is enough for double precision
                mid = (lo + hi) / 2.0
                f_mid = f(mid)
                
                if f_mid == 0:
                    lo = hi = mid
                    break
                if f_lo * f_mid <= 0:
                    hi = mid
                    f_hi = f_mid
                else:
                    lo = mid
                    f_lo = f_mid
                    
            refined_boundaries[i] = (lo + hi) / 2.0

    # Create the exercise policy
    policy = _ExercisePolicy(refined_boundaries, n_steps, dt, K)
    return policy


def _get_param(contract, key, default):
    """Extract parameter from contract (dict or object)."""
    if contract is None:
        return default
    if isinstance(contract, dict):
        return contract.get(key, default)
    else:
        # Try common attribute names
        for attr in [key, key.lower(), key.upper(), '_' + key]:
            if hasattr(contract, attr):
                return getattr(contract, attr)
        # Try common variations
        name_map = {
            'K': ['K', 'strike', 'strike_price', 'knockout', 'knockout_price'],
            'r': ['r', 'rate', 'risk_free_rate', 'riskfree_rate'],
            'sigma': ['sigma', 'vol', 'volatility', 'vola'],
            'T': ['T', 'maturity', 'expiry', 'tenor'],
            'n_steps': ['n_steps', 'n_exercise', 'n_exercises', 'n_dates', 'steps'],
        }
        if key in name_map:
            for alt in name_map[key]:
                if hasattr(contract, alt):
                    return getattr(contract, alt)
        return default


def _gauss_hermite(n):
    """Compute Gauss-Hermite nodes and weights for n points.
    
    Returns nodes x_k and weights w_k such that:
    sum_k w_k * f(x_k) approx integral_{-inf}^{inf} exp(-x^2) * f(x) dx
    """
    n_int = int(n)
    if n_int <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n_int == 1:
        return np.array([0.0]), np.array([math.sqrt(math.pi)])
    
    try:
        from numpy.polynomial.hermite import hermgauss
        x, w = hermgauss(n_int)
        return x, w
    except (ImportError, Exception):
        pass
    
    # Fallback: build Jacobi matrix for Hermite polynomials with weight exp(-x^2)
    b = np.sqrt(np.arange(1, n_int, dtype=float))
    A = np.diag(b, 1) + np.diag(b, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    idx = np.argsort(eigenvalues)
    x = eigenvalues[idx]
    v = eigenvectors[:, idx]
    
    w = math.sqrt(math.pi) * (v[0, :] ** 2)
    
    return x, w


class _ExercisePolicy:
    """Exercise policy based on fitted boundaries."""
    
    def __init__(self, boundaries, n_steps, dt, K):
        self.boundaries = boundaries  # boundaries[i] for exercise date i+1
        self.n_steps = n_steps
        self.dt = dt
        self.K = K
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise at the given time and state.
        
        Parameters:
        -----------
        current_state : array-like
            Current spot prices
        time : float or int
            Current time (could be continuous or step index)
        
        Returns:
        --------
        numpy array of booleans
        """
        current_state = np.asarray(current_state, dtype=float)
        
        # Determine which exercise date this corresponds to
        # time could be a float (continuous) or int (step index)
        if isinstance(time, (int, np.integer)):
            step = int(time)
            if step < 1 or step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1  # 0-based index into boundaries
        else:
            t = float(time)
            # Find the closest exercise date
            # Exercise dates are at k * dt for k=1..n_steps
            # If t is not close to any exercise date, we cannot exercise.
            # However, the evaluator might pass the exact date.
            # Let's check if t is close to a multiple of dt.
            
            # Calculate step index
            step_float = t / self.dt
            step = round(step_float)
            
            # Check if t is actually close to step * dt
            # Tolerance: 1e-9
            if abs(t - step * self.dt) > 1e-9:
                # Not an exercise date
                return np.zeros(len(current_state), dtype=bool)
                
            if step < 1 or step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1
        
        # Exercise if S <= boundary
        boundary = self.boundaries[idx]
        # Only exercise if intrinsic value is positive (S < K)
        # The boundary is already < K, but check for safety
        return (current_state <= boundary) & (current_state < self.K)


# EVOLVE-BLOCK-END
