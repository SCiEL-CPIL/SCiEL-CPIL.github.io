"""Bermudan exercise-policy: model-and-optimize with PCHIP and adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    training_paths = np.asarray(training_paths, dtype=np.float64)
    # n_paths is not strictly needed for the grid-based solver, but good for context

    # Gauss-Hermite quadrature setup
    n_gh = 200
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n_gh)
    z_nodes = np.sqrt(2.0) * gh_nodes
    z_weights = gh_weights / np.sqrt(np.pi)
    z_weights /= z_weights.sum()

    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    exp_factors = np.exp(drift + vol * z_nodes)  # (n_gh,)
    ef = exp_factors
    zw = z_weights

    def run_backward(grid):
        """
        Run backward induction on the given grid using PCHIP interpolation.
        Returns V (value array) and decisions (dict of time -> boolean array).
        """
        g = grid
        n_g = len(g)
        # Terminal condition: Put payoff
        V = np.maximum(K - g, 0.0)
        decisions = {}

        for t in range(n_steps, 0, -1):
            # Calculate next state spots: (n_grid, n_gh)
            # We need to interpolate V at these next states.
            # To speed up PCHIP, we might need to handle out-of-bounds or ensure grid covers range.
            # Grid is log-spaced from s_min to s_max.
            
            ns = g[:, None] * ef[None, :]
            
            # Clip to grid bounds to avoid extrapolation issues in PCHIP
            # PCHIP can extrapolate, but linear or constant is safer for tails.
            # However, for a put, value -> 0 as S -> inf.
            # For S -> 0, value -> K.
            # We will clip to [g[0], g[-1]]
            ns_c = np.clip(ns, g[0], g[-1])
            
            # Interpolate V at ns_c
            # PCHIP is vectorized? Yes, but creating the interpolant every step is expensive.
            # We only create it once per time step.
            try:
                pchip = PchipInterpolator(g, V)
                Vi = pchip(ns_c)
            except Exception:
                # Fallback to linear if PCHIP fails (e.g., duplicate points)
                idx = np.searchsorted(g, ns_c)
                idx = np.clip(idx, 1, len(g) - 1)
                sl = g[idx - 1]
                su = g[idx]
                denom = su - sl
                denom = np.where(denom < 1e-12, 1e-12, denom)
                w = np.clip((ns_c - sl) / denom, 0.0, 1.0)
                Vl = V[idx - 1]
                Vu = V[idx]
                Vi = (1.0 - w) * Vl + w * Vu

            # Expected continuation value
            cont = discount * (Vi @ zw)
            
            # Exercise value
            ex = np.maximum(K - g, 0.0)
            
            # Optimal value
            V_new = np.maximum(ex, cont)
            
            # Record decisions for exercise (if t < n_steps, i.e., not terminal)
            if t < n_steps:
                # Exercise if ex >= cont
                decisions[t] = ex >= cont
            
            V = V_new

        return V, decisions

    # Initial Grid
    # Spot range: 0.1 to 3*K is usually sufficient for sigma=0.2, T=1
    s_min = 0.01
    s_max = 3.0 * K
    n_grid_base = 20000
    grid = np.exp(np.linspace(np.log(s_min), np.log(s_max), n_grid_base))

    V, dec = run_backward(grid)
    grid_final = grid
    decisions_final = dec

    # Adaptive Refinement Passes
    # Strategy: Find where the exercise decision flips (exercise front).
    # Add more points in a local window around these flips.
    
    for pass_num in range(3):
        if time.time() > deadline - 20:
            break
            
        # Identify regions to refine
        # We look at the decisions for each time step t in 1..n_steps-1
        # The "front" is where d[i] != d[i+1]
        
        refine_points = []
        
        for t in range(1, n_steps):
            if t not in decisions_final:
                continue
            d = decisions_final[t]
            # Find indices where decision changes
            # d is boolean array. 
            # If d[i] is True (Exercise) and d[i+1] is False (Hold), or vice versa.
            # For a put, we exercise at low S, hold at high S.
            # So we expect True -> False as S increases.
            flips = np.where(d[:-1] != d[1:])[0]
            
            if len(flips) == 0:
                continue
            
            for f in flips:
                # Define a window around the flip
                # The flip is between index f and f+1.
                # We want to add points in [grid[f-10], grid[f+10]]
                lo_idx = max(0, f - 20)
                hi_idx = min(len(grid_final) - 1, f + 20)
                
                if hi_idx > lo_idx:
                    # Add 500 points in this log-linear interval
                    s_lo = grid_final[lo_idx]
                    s_hi = grid_final[hi_idx]
                    new_pts = np.exp(np.linspace(np.log(s_lo), np.log(s_hi), 500))
                    refine_points.append(new_pts)
        
        if not refine_points:
            break
            
        # Combine existing grid and new points
        extra_grid = np.concatenate(refine_points)
        new_grid = np.unique(np.concatenate([grid_final, extra_grid]))
        
        # Sort is guaranteed by np.unique on sorted inputs? 
        # grid_final is sorted. extra_grid is not necessarily globally sorted, but np.unique sorts.
        
        # Run backward induction on the refined grid
        V_ref, dec_ref = run_backward(new_grid)
        
        grid_final = new_grid
        decisions_final = dec_ref
        
        # Optional: Check if the boundary moved significantly or if we converged
        # For simplicity, we just do fixed passes.

    # Final Polish: If time permits, do one more very tight pass on the most recent boundary
    if time.time() < deadline - 10:
        refine_points_final = []
        for t in range(1, n_steps):
            if t not in decisions_final:
                continue
            d = decisions_final[t]
            flips = np.where(d[:-1] != d[1:])[0]
            for f in flips:
                lo_idx = max(0, f - 5)
                hi_idx = min(len(grid_final) - 1, f + 5)
                if hi_idx > lo_idx:
                    s_lo = grid_final[lo_idx]
                    s_hi = grid_final[hi_idx]
                    # Very dense points here
                    new_pts = np.exp(np.linspace(np.log(s_lo), np.log(s_hi), 1000))
                    refine_points_final.append(new_pts)
        
        if refine_points_final:
            extra_grid_final = np.concatenate(refine_points_final)
            new_grid_final = np.unique(np.concatenate([grid_final, extra_grid_final]))
            V_final, dec_final = run_backward(new_grid_final)
            grid_final = new_grid_final
            decisions_final = dec_final

    class ExercisePolicy:
        def __init__(self, grid, decisions, n_steps, K):
            self.grid = grid
            self.decisions = decisions
            self.n_steps = n_steps
            self.K = K

        def exercise(self, current_state, time):
            if time < 1 or time > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            cs = np.asarray(current_state, dtype=np.float64)
            if time not in self.decisions:
                # Fallback: Exercise if deep ITM?
                # For a put, exercise if S < K.
                return cs < self.K
            
            # Map current states to grid indices
            # Since we have a dense grid, we can use searchsorted
            idx = np.searchsorted(self.grid, cs)
            # Clip indices to valid range
            idx = np.clip(idx, 0, len(self.grid) - 1)
            
            # Return the decision for these indices
            return self.decisions[time][idx]

    return ExercisePolicy(grid_final, decisions_final, n_steps, K)

# EVOLVE-BLOCK-END
