"""Rewritten program using simulated annealing with batched flips for the autoconvolution upper bound."""
import numpy as np
import time

N = 400

def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Random seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Initialize with a random signed step function
    h = rng.choice([-1, 1], size=N).astype(np.float64)
    
    # Precompute FFT size
    n_fft = 1 << (2 * N - 1).bit_length()
    
    def _compute_acf(h):
        """Compute autocorrelation via FFT."""
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft).real[:N]
        return acf
    
    def _max_sidelobe(acf):
        """Return the maximum absolute value of non-zero lags."""
        return np.max(np.abs(acf[1:]))
    
    def _delta_acf_from_flip(h, indices):
        """
        Compute the change in autocorrelation when flipping the signs at given indices.
        
        For a flip at indices S, h -> h - 2*h_S (where h_S is h with zeros outside S).
        The new autocorrelation is:
          acf_new = acf_old - 2 * acf(h, h_S) - 2 * acf(h_S, h) + 4 * acf(h_S, h_S)
        But acf(h, h_S) is the cross-correlation.
        
        Actually, let's think more carefully.
        
        acf(h) = h * h (circular, via FFT)
        If we flip indices S, let delta = 2 * h_S (so h_new = h - delta)
        Then acf_new = (h - delta) * (h - delta) = h*h - h*delta - delta*h + delta*delta
        = acf_old - acf(h, delta) - acf(delta, h) + acf(delta)
        
        Since delta = 2 * h_S, and h_S is h restricted to S (zero elsewhere):
        acf(h, delta) = 2 * acf(h, h_S)
        
        This is getting complex. Let's use a simpler approach:
        For a single flip at index i, the change in autocorrelation is well-known.
        For multiple flips, we can compute the new acf directly via FFT, which is O(N log N).
        
        Given N=400, and we're doing many iterations, let's benchmark:
        - FFT of size ~1024 is very fast (~microseconds)
        - For batch flips, we can just recompute the full acf via FFT
        
        So the simplest correct approach: just recompute acf via FFT after each batch flip.
        """
        # For correctness and simplicity, just return None and recompute
        # This function is kept for potential optimization but we'll use direct recompute
        return None
    
    # Initial acf
    acf = _compute_acf(h)
    best_val = _max_sidelobe(acf)
    best_h = h.copy()
    current_val = best_val
    
    # Simulated annealing parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.9995  # cooling rate
    
    T = T_start
    batch_size = 10  # flip 10 random indices per step
    
    while time.time() < deadline:
        # Choose random indices to flip
        if batch_size < N:
            indices = rng.choice(N, size=batch_size, replace=False)
        else:
            indices = rng.permutation(N)
        
        # Flip the signs
        h[indices] *= -1
        
        # Recompute acf
        new_acf = _compute_acf(h)
        new_val = _max_sidelobe(new_acf)
        
        # Decide whether to accept
        if new_val < current_val:
            accept = True
        else:
            delta = new_val - current_val
            if T < 1e-9:
                accept = False
            else:
                accept_prob = np.exp(-delta / T)
                accept = rng.random() < accept_prob
        
        if not accept:
            # Revert the flip
            h[indices] *= -1
            new_acf = acf  # revert to old acf
            new_val = current_val
        
        # Update state
        acf = new_acf
        current_val = new_val
        
        # Track best
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
        
        # Cool down
        T *= alpha
        
        # If temperature is too low, restart with a random initialization
        if T < T_end:
            if time.time() < deadline - 5.0:
                # Random restart
                h = rng.choice([-1, 1], size=N).astype(np.float64)
                acf = _compute_acf(h)
                current_val = _max_sidelobe(acf)
                T = T_start * 0.5  # restart with lower initial temperature
    
    return best_h
