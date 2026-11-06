import numpy as np
import time
from scipy.stats import norm
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12
    dt = T / n_steps
    
    # Grid for State S
    # S ranges from 0 to 2*K (put value is 0 for S >> K)
    S_max = 2.0 * K
    n_grid = 1000
    S_grid = np.linspace(1e-6, S_max, n_grid)
    
    # Gauss-Hermite quadrature for the expectation
    # E[f(Z)] where Z ~ N(0,1)
    # Using 50 points is usually sufficient for smooth functions
    n_quad = 50
    x, w = roots_hermite(n_quad)
    # Transform to standard normal: Z = sqrt(2) * x
    # Weights need to be normalized: w_i * exp(-x_i^2) / sqrt(pi)
    # roots_hermite returns weights for integral exp(-x^2) f(x) dx
    # We want integral f(z) exp(-z^2/2) / sqrt(2*pi) dz
    # Let z = sqrt(2) x, dz = sqrt(2) dx
    # Integral becomes sum w_i f(sqrt(2)x_i) * exp(-2x_i^2) * sqrt(2) / sqrt(2*pi)
    # Wait, standard Gauss-Hermite: integral_{-inf}^{inf} exp(-x^2) f(x) dx ~ sum w_i f(x_i)
    # We need E[g(Z)] = 1/sqrt(2*pi) integral_{-inf}^{inf} exp(-z^2/2) g(z) dz
    # Let z = sqrt(2) x. dz = sqrt(2) dx. exp(-z^2/2) = exp(-x^2).
    # Integral = 1/sqrt(2*pi) * sqrt(2) * integral_{-inf}^{inf} exp(-x^2) g(sqrt(2)x) dx
    # = 1/sqrt(pi) * sum w_i g(sqrt(2)x_i)
    z_nodes = np.sqrt(2) * x
    z_weights = w / np.sqrt(np.pi)
    
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    disc = np.exp(-r * dt)
    
    # V_next[S_grid]
    # At maturity (step 12), value is intrinsic
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store thresholds
    thresholds = np.zeros(n_steps)
    
    # Backward induction
    for k in range(n_steps - 1, 0, -1):
        # Calculate Continuation Value C(S) for all S in S_grid
        # S_next = S * exp(mu + vol * Z)
        # We need to evaluate V_next at S * exp(mu + vol * z_nodes)
        
        # S_grid shape (n_grid, 1)
        # z_nodes shape (1, n_quad)
        # S_next shape (n_grid, n_quad)
        
        log_S_next = np.log(S_grid)[:, np.newaxis] + mu + vol * z_nodes[np.newaxis, :]
        S_next = np.exp(log_S_next)
        
        # Interpolate V_next at S_next
        # V_next is defined on S_grid
        # Clip S_next to S_grid range
        S_next_clipped = np.clip(S_next, S_grid[0], S_grid[-1])
        
        # Linear interpolation
        # np.interp expects 1D xp. We can flatten or use broadcasting.
        # For efficiency, we can use np.interp on flattened arrays or loop.
        # Since n_grid=1000, n_quad=50, total 50k points.
        # np.interp is fast.
        
        # Reshape for np.interp
        S_next_flat = S_next_clipped.ravel()
        V_next_vals = np.interp(S_next_flat, S_grid, V_next)
        V_next_vals = V_next_vals.reshape(n_grid, n_quad)
        
        # Weighted sum
        # C(S) = disc * sum_j w_j V_next(S_j)
        C_vals = disc * np.sum(V_next_vals * z_weights[np.newaxis, :], axis=1)
        
        # Intrinsic Value
        I_vals = np.maximum(K - S_grid, 0.0)
        
        # Value
        V_curr = np.maximum(I_vals, C_vals)
        
        # Exercise Decision: Exercise if I_vals >= C_vals
        # Due to floating point, use a small epsilon
        exercise_mask = (I_vals >= C_vals - 1e-10)
        
        # Find threshold
        # The mask is True for low S, False for high S.
        # Find the last index where it is True.
        true_indices = np.where(exercise_mask)[0]
        if len(true_indices) > 0:
            idx = true_indices[-1]
            # Refine threshold using bisection between S_grid[idx] and S_grid[idx+1] (if exists)
            S_low = S_grid[idx]
            if idx + 1 < n_grid:
                S_high = S_grid[idx + 1]
            else:
                S_high = S_max
            
            # Bisection to find exact point where I(S) = C(S)
            # Function to evaluate I(S) - C(S)
            def diff_func(S):
                # Calculate C(S) for single S
                log_S_next_single = np.log(S) + mu + vol * z_nodes
                S_next_single = np.exp(log_S_next_single)
                S_next_single_clipped = np.clip(S_next_single, S_grid[0], S_grid[-1])
                V_next_single = np.interp(S_next_single_clipped, S_grid, V_next)
                C_single = disc * np.sum(V_next_single * z_weights)
                I_single = max(K - S, 0.0)
                return I_single - C_single
            
            # Bisection
            a, b = S_low, S_high
            fa = diff_func(a)
            fb = diff_func(b)
            
            # If fa is positive and fb is negative, root is in [a,b]
            if fa > 0 and fb < 0:
                for _ in range(20):
                    m = (a + b) / 2.0
                    fm = diff_func(m)
                    if fm > 0:
                        a = m
                    else:
                        b = m
                thresholds[k-1] = (a + b) / 2.0
            else:
                # Fallback to grid point
                thresholds[k-1] = S_low
        else:
            # Never exercise? (Should not happen for put at t>0 unless S is very high, 
            # but at t=11, if S is high, we don't exercise. 
            # However, for k=1, S=90, we should exercise.
            # If no true indices, threshold is 0.
            thresholds[k-1] = 0.0
            
        V_next = V_curr
        
        if time.time() > deadline:
             break

    class ExercisePolicy:
        def __init__(self, thresholds):
            self.thresholds = thresholds
            self.n_steps = n_steps
            
        def exercise(self, current_state, time_index):
            # time_index is 0-indexed? 
            # "12 monthly exercise dates (no time-0 exercise)"
            # Usually time_index 0 corresponds to t_1, ..., 11 corresponds to t_12.
            # Or time_index 1..12?
            # The contract says "exercise(current_state, time) returning a boolean vector".
            # "time" is likely the index of the exercise date.
            # Let's assume time is 0-indexed for the 12 dates, i.e., 0..11.
            # If time is 1..12, we adjust.
            # Given "no time-0 exercise", the first exercise is at t=1/12.
            # If the evaluator passes t in [0, 1, ..., 11], we use thresholds[t].
            # If it passes t in [1, ..., 12], we use thresholds[t-1].
            # Standard convention in such tasks: if there are N dates, indices are 0..N-1.
            
            t = int(time)
            if t < 0 or t >= self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            
            thresh = self.thresholds[t]
            return np.array(current_state) <= thresh

    return ExercisePolicy(thresholds)
