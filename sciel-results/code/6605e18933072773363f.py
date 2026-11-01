"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    Metric: max_k |(h*h)_k| / (sum_i |h_i|)^2
    """
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h == 0:
        return float('inf')
    return max_conv / (sum_abs_h ** 2)


def _fast_c3_bound(h):
    """
    Faster computation of the C3 upper bound using FFT for convolution.
    """
    # Use FFT for faster convolution
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2*n-1]
    max_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h == 0:
        return float('inf')
    return max_conv / (sum_abs_h ** 2)


def _ising_energy(s):
    """
    Compute the energy of an Ising spin configuration.
    Energy = max_k |autocorr(s)[k]|
    We want to minimize this.
    """
    conv = np.convolve(s, s, mode='full')
    return np.max(np.abs(conv))


def _ising_energy_fast(s):
    """
    Faster Ising energy using FFT.
    """
    n = len(s)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    S = np.fft.rfft(s.astype(float), size)
    conv = np.fft.irfft(S * S, size)[:2*n-1]
    return np.max(np.abs(conv))


def _simulated_annealing(s_init, t_start, t_stop, n_iter, deadline):
    """
    Simulated annealing on Ising spins.
    s_init: initial spin configuration (+1 or -1)
    t_start: initial temperature
    t_stop: final temperature
    n_iter: number of iterations
    deadline: time limit (seconds)
    """
    s = s_init.copy().astype(float)
    current_energy = _ising_energy_fast(s)
    best_s = s.copy()
    best_energy = current_energy
    
    t = t_start
    # Geometric cooling
    if n_iter > 1:
        ratio = (t_stop / t_start) ** (1.0 / (n_iter - 1))
    else:
        ratio = 1.0
    
    start_time = time.time()
    
    for it in range(n_iter):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Compute energy change from flipping s[idx]
        # E = max_k |conv(s)[k]|
        # Flipping s[idx] changes conv by: conv_new[k] = conv_old[k] - 2*s[idx]*s[k-idx] (for relevant k)
        # But we need to recompute the max, which is expensive.
        # Instead, let's use a simpler approach: just recompute.
        
        old_val = s[idx]
        s[idx] = -s[idx]
        
        new_energy = _ising_energy_fast(s)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / max(t, 1e-10)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_s = s.copy()
        else:
            s[idx] = old_val
        
        t *= ratio
        if t < t_stop:
            t = t_stop
    
    return best_s, best_energy


def _local_search(s, max_iter, deadline):
    """
    Simple local search: try flipping each spin and keep improvements.
    """
    s = s.copy().astype(float)
    current_energy = _ising_energy_fast(s)
    best_s = s.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            old_val = s[idx]
            s[idx] = -s[idx]
            new_energy = _ising_energy_fast(s)
            
            if new_energy < current_energy:
                current_energy = new_energy
                improved = True
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_s = s.copy()
            else:
                s[idx] = old_val
        
        if not improved:
            break
    
    return best_s, best_energy


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the autoconvolution upper bound on C3.
    
    Uses simulated annealing on an Ising spin glass formulation.
    """
    deadline = time.time() + 110.0  # Leave 10s safety margin
    
    rng = np.random.RandomState(42)
    
    best_s = None
    best_val = float('inf')
    
    # Multiple restarts with simulated annealing
    n_restarts = 10
    t_start = 5.0
    t_stop = 0.01
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial spin configuration
        s_init = rng.choice([-1.0, 1.0], N)
        
        # Simulated annealing
        n_iter = 5000
        s_sa, e_sa = _simulated_annealing(s_init, t_start, t_stop, n_iter, deadline)
        
        # Local search refinement
        s_ls, e_ls = _local_search(s_sa, max_iter=50, deadline=deadline)
        
        if e_ls < best_val:
            best_val = e_ls
            best_s = s_ls
        
        # If we're doing well, we can use more aggressive parameters
        if best_val < 10.0:
            t_start = 3.0
            t_stop = 0.005
    
    # Convert to heights: the step function heights are the spins themselves
    # The metric is max|conv| / (sum|s_i|)^2
    # For binary spins, sum|s_i| = N = 400, so denominator = 400^2 = 160000
    # We return the spin configuration as the heights
    
    if best_s is None:
        # Fallback: random signs
        best_s = rng.choice([-1.0, 1.0], N)
    
    return best_s

# EVOLVE-BLOCK-END
