"""Bermudan put exercise policy via deterministic backward induction on a price grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm

_DEADLINE = 145.0

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using deterministic backward induction.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_time_steps) with GBM paths
    - contract: dict with keys like 'K', 'r', 'sigma', 'T', 'exercise_dates'
    
    Returns:
    - An object with .exercise(current_state, time) method returning boolean array
    """
    start_time = _time.time()
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_exercise_dates', 12)
    
    # Exercise dates: no exercise at t=0, 12 monthly dates
    exercise_times = np.linspace(0, T, n_dates + 1)[1:]  # t_1, t_2, ..., t_12
    
    # Create a fine grid of spot prices (log-spaced)
    S_min = 1.0
    S_max = 300.0
    n_grid = 10000
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # Precompute quadrature points for the 1D integral
    # We need to compute E[V(S * exp(a + b*Z))] where Z ~ N(0,1)
    # Use Gauss-Hermite or simple trapezoidal on a range of Z
    # Z range: [-5, 5] should be sufficient
    n_quad = 200
    z_quad = np.linspace(-5, 5, n_quad)
    z_weights = norm.pdf(z_quad) * (z_quad[1] - z_quad[0])  # trapezoidal weights (approx)
    # Better: use proper quadrature
    # Actually, let's use the fact that we can compute the expectation via
    # change of variables. Let Y = a + b*Z, then Y ~ N(a, b^2)
    # E[V(S*exp(Y))] = ∫ V(S*exp(y)) φ((y-a)/b)/b dy
    # But simpler to just do the integral over z.
    
    # For each step, Δt = T/n_dates
    dt = T / n_dates
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    # Backward induction
    # V_next will be the value at the next exercise date
    # Start at maturity: V_12(S) = max(K-S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Boundaries: exercise_boundary[i] is the threshold at exercise date i
    # where intrinsic >= continuation
    boundaries = np.zeros(n_dates)
    
    for i in range(n_dates - 1, -1, -1):
        # Check deadline
        if _time.time() - start_time > _DEADLINE - 5:
            break
        
        # Compute continuation value at exercise date i
        # C_i(S) = e^{-r*dt} * E[V_next(S * exp(drift + vol*Z))]
        # = e^{-r*dt} * ∫ V_next(S * exp(drift + vol*z)) φ(z) dz
        
        # For each grid point S_j, compute the integral
        # S_j * exp(drift + vol*z_quad) gives the next-period spot values
        # Shape: (n_grid, n_quad)
        next_spots = S_grid[:, None] * np.exp(drift + vol * z_quad[None, :])
        
        # Interpolate V_next at next_spots
        # V_next is defined on S_grid, so we need to interpolate
        # Use np.interp for each row - but that's slow. Vectorize.
        # Actually, np.interp works on 1D arrays. We can do it in chunks or use a different approach.
        
        # Alternative: since next_spots may be outside [S_min, S_max], clip and interpolate
        # For efficiency, let's use a coarser approach: evaluate V_next at next_spots via interpolation
        
        # Vectorized interpolation: for each S_j, we need V_next at next_spots[j, :]
        # This is equivalent to interpolating a 2D array, but V_next is 1D.
        
        # Let's do it in a loop over grid points for clarity, but vectorize the inner part
        # Actually, with 10000 grid points and 200 quadrature points, we can do:
        
        # Clip next_spots to grid range
        next_spots_clipped = np.clip(next_spots, S_min, S_max)
        
        # Interpolate: for each row, interpolate V_next at next_spots_clipped[row, :]
        # np.interp requires 1D. We can use:
        # V_at_next = np.array([np.interp(next_spots_clipped[j], S_grid, V_next) for j in range(n_grid)])
        # This is a Python loop of 10000 iterations, each doing a 200-point interp. Should be OK.
        
        V_at_next = np.empty_like(next_spots_clipped)
        for j in range(n_grid):
            V_at_next[j] = np.interp(next_spots_clipped[j], S_grid, V_next)
        
        # Continuation value
        continuation = np.exp(-r * dt) * np.sum(V_at_next * z_weights[None, :], axis=1)
        
        # Bermudan value: max(intrinsic, continuation)
        intrinsic = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(intrinsic, continuation)
        
        # Exercise boundary: largest S where intrinsic >= continuation
        # intrinsic >= continuation means K - S >= continuation (for S < K)
        # Find where intrinsic >= continuation
        exercise_mask = intrinsic >= continuation
        if np.any(exercise_mask):
            # Find the largest S where exercise is optimal
            idx = np.where(exercise_mask)[0]
            if len(idx) > 0:
                boundaries[i] = S_grid[idx[-1]]
            else:
                boundaries[i] = 0.0
        else:
            boundaries[i] = 0.0
        
        V_next = V_current
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, S_grid, K):
            self.boundaries = boundaries
            self.S_grid = S_grid
            self.K = K
            self.n_dates = len(boundaries)
        
        def exercise(self, current_state, time):
            """
            Return boolean array indicating whether to exercise.
            current_state: array of spot prices
            time: current time (scalar)
            """
            # Determine which exercise date index we're at
            # Exercise dates are at T/12, 2T/12, ..., T
            # time is given as a scalar
            
            # Find the exercise date index
            # Exercise date i is at time (i+1)*T/n_dates for i=0,...,n_dates-1
            # We need to find which exercise date 'time' corresponds to
            
            # Compute the index
            # t_i = (i+1) * dt, where dt = T/n_dates
            # So i = time/dt - 1
            dt = T / self.n_dates
            idx_float = time / dt - 1
            idx = int(round(idx_float))
            
            # Clamp to valid range
            if idx < 0:
                idx = 0
            if idx >= self.n_dates:
                idx = self.n_dates - 1
            
            # Exercise if S <= boundary
            boundary = self.boundaries[idx]
            return current_state <= boundary
    
    return ExercisePolicy(boundaries, S_grid, K)
# EVOLVE-BLOCK-END
