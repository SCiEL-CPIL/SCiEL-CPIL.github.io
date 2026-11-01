"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_ac(h, F):
    """
    Compute the aperiodic autocorrelation sidelobes using pre-computed FFT.
    h: numpy array of length N
    F: numpy array of length 2N (complex), FFT of h
    Returns: max absolute value of sidelobes.
    """
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=2 * N)
    # Sidelobes are lags 1 to N-1
    return np.max(np.abs(ac[1:N]))

def _update_fft_incremental(F, h, idx, sign_change):
    """
    Update the FFT array F when h[idx] changes by sign_change * 2 * h[idx].
    Since h[idx] is +/-1, the change is -2*h[idx] (flip) or similar.
    Here we assume we are flipping h[idx], so delta = -2 * h[idx].
    F_new[k] = F[k] + delta * exp(-2j * pi * k * idx / n_fft)
    """
    n_fft = 2 * N
    delta = -2.0 * h[idx]
    # Vectorized update
    # k = 0..N-1 for rfft
    k = np.arange(N)
    # exp(-2j * pi * k * idx / n_fft)
    # Note: rfft returns 0..N/2, but numpy rfft length is N/2+1 for even n_fft.
    # Wait, n_fft=800. rfft returns 401 points.
    # The formula works for all k.
    phase = np.exp(-2j * np.pi * k * idx / n_fft)
    F += delta * phase
    return F

def _sa_search(h, F, time_limit=10.0):
    """
    Simulated Annealing search.
    """
    start_time = time.time()
    current_cost = _compute_ac(h, F)
    best_h = h.copy()
    best_cost = current_cost
    
    # Cooling schedule
    # Start T high enough to allow moves, end low.
    # Autocorrelation values for random N=400 are around sqrt(N) ~ 20.
    # Max possible is N.
    # Let's start T=10.0 and cool down to 0.1.
    T = 10.0
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    # Pre-compute phases for all indices? 
    # phases[idx, k] = exp(-2j * pi * k * idx / n_fft)
    # This is an N x N/2+1 matrix. 400 x 401 complex. ~ 1.6 MB. Very fast to precompute.
    n_fft = 2 * N
    k = np.arange(len(F))
    idxs = np.arange(N)
    # Matrix of phases: rows are indices 0..N-1, cols are FFT bins 0..N/2
    # phase[i, j] = exp(-2j * pi * j * i / n_fft)
    # Using broadcasting
    # k is shape (M,), idxs is shape (N,)
    # We want (N, M)
    phases = np.exp(-2j * np.pi * np.outer(idxs, k) / n_fft)
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Calculate new cost
        # Save current F state? No, we can update F, compute cost, and revert if needed.
        # But _compute_ac uses F.
        
        old_val = h[idx]
        h[idx] = -old_val
        F_new = F.copy() # Copying F (401 complex) is fast, but we can do in-place and revert
        # Actually, let's do in-place update on F, compute cost, then revert F if we reject.
        
        # Update F
        F += -2.0 * old_val * phases[idx]
        
        new_cost = _compute_ac(h, F)
        
        # Decide acceptance
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0:
            # Improvement
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            if T > T_min:
                if np.random.rand() < np.exp(-delta_cost / T):
                    current_cost = new_cost
                else:
                    # Reject: revert h and F
                    h[idx] = old_val
                    F -= -2.0 * old_val * phases[idx]
            else:
                # Reject: revert h and F
                h[idx] = old_val
                F -= -2.0 * old_val * phases[idx]
        
        # Cool down
        T *= alpha
        
    return best_h, best_cost

def construct():
    start_time = time.time()
    time_budget = 110.0 # Leave margin for 120s limit
    
    best_h = None
    best_score = float('inf')
    
    # Random restarts
    while time.time() - start_time < time_budget:
        # Initialize with random signs
        h = np.random.choice([-1, 1], size=N).astype(float)
        
        # Compute initial FFT
        F = np.fft.rfft(h, n=2 * N)
        
        # Run SA search
        # Allocate time based on remaining budget
        # If we have 100s left, do a 5s search.
        # If we have 10s left, do a 5s search.
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5.0:
            break
            
        # Use 5s per restart to allow many restarts
        # Or dynamic? Fixed 5s is safe.
        h_res, score = _sa_search(h, F, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_h = h_res
            
    if best_h is None:
        return np.ones(N, dtype=int)
        
    return best_h.astype(int)
# EVOLVE-BLOCK-END
