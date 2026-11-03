"""Rewritten program using cellular-automaton / statistical-mechanics ground-state search."""
import numpy as np
import time

N = 400


def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv(h)
    return np.max(np.abs(ac)) / (s * s)


def _spectral_density(h):
    """
    Local spectral density: |FFT(h)|^2, which is the autoconvolution in frequency domain.
    We use this as a local energy proxy: high |H_k|^2 at certain k means poor spectral flatness.
    Actually, the autoconvolution in time domain is related to |H|^2 in freq domain.
    For the energy, we want to minimize the peak of the autoconvolution, which corresponds
    to minimizing the max of the inverse FFT of |H|^2. But a simpler local energy is the
    variance of the power spectrum: sum(|H_k|^4) / (sum|H_k|^2)^2.
    """
    n = len(h)
    fsize = 1
    while fsize < 2 * n - 1:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    power = np.abs(H) ** 2
    return power


def _local_energy(h, window=5):
    """
    Compute a local energy based on the spectral density.
    We define the energy as the sum over positions of the max spectral density
    in a local window, which encourages local spectral flatness.
    """
    power = _spectral_density(h)
    # Use only the positive frequencies (rfft gives n//2+1 bins)
    # For local energy, we can just use the total power as a global measure
    # But to make it local, we can compute the autoconvolution and use local max
    ac = _autoconv(h)
    # Local max of |ac| in a sliding window
    abs_ac = np.abs(ac)
    local_max = np.zeros_like(abs_ac)
    for i in range(len(abs_ac)):
        lo = max(0, i - window)
        hi = min(len(abs_ac), i + window + 1)
        local_max[i] = np.max(abs_ac[lo:hi])
    return np.sum(local_max)


def _ca_step(h, beta, power):
    """
    One step of the cellular automaton / simulated annealing.
    For each position, compute the change in energy if we flip the sign.
    Accept with probability exp(-beta * delta_energy) if delta_energy > 0.
    """
    n = len(h)
    ac = _autoconv(h)
    abs_ac = np.abs(ac)
    current_max = np.max(abs_ac)
    s = np.sum(h)
    
    # Precompute the contribution of each element to the autoconvolution
    # Flipping h[i] changes ac by -2*h[i]*h[j] for all j, but this is complex.
    # Instead, we use a simpler approach: compute the change in the score.
    
    # For efficiency, we'll use a random subset of positions per step
    rng = np.random.default_rng()
    n_trials = min(n, 50)
    indices = rng.choice(n, size=n_trials, replace=False)
    
    for i in indices:
        old_val = h[i]
        h[i] = -old_val
        
        new_ac = _autoconv(h)
        new_abs_ac = np.abs(new_ac)
        new_max = np.max(new_abs_ac)
        new_s = np.sum(h)
        
        if abs(new_s) < 1e-9:
            h[i] = old_val
            continue
            
        old_score = current_max / (s * s)
        new_score = new_max / (new_s * new_s)
        
        delta = new_score - old_score
        if delta < 0 or rng.random() < np.exp(-beta * delta):
            current_max = new_max
            s = new_s
        else:
            h[i] = old_val
    
    return h


def _ground_state_search(h_init, t_start, deadline, max_iters=500):
    """
    Iterative refinement map: evolve the sign pattern by minimizing a discrete energy function.
    """
    h = h_init.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = h_init.copy() * 1.0
        s = np.sum(h)
    
    best_h = h.copy()
    best_val = _score(h)
    
    # Simulated annealing schedule
    beta_start = 1.0
    beta_end = 100.0
    n_steps = max_iters
    
    for it in range(n_steps):
        if time.time() > deadline:
            break
        
        # Annealing schedule: beta increases over time
        progress = it / n_steps
        beta = beta_start + (beta_end - beta_start) * progress**2
        
        # One CA step
        h = _ca_step(h, beta, None)
        
        # Evaluate
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val


def _gen_initial(n, seed):
    """Generate diverse initial patterns."""
    rng = np.random.default_rng(seed)
    
    # Pattern 1: Random signs
    h1 = rng.choice([-1.0, 1.0], size=n)
    
    # Pattern 2: Chirp
    i = np.arange(n)
    h2 = np.cos(np.pi * i**2 / n)
    
    # Pattern 3: Cosine window
    h3 = np.cos(np.pi * i / n)
    
    # Pattern 4: Gaussian window
    mu = n / 2
    sigma = n / 4
    h4 = np.exp(-(i - mu)**2 / (2 * sigma**2))
    
    # Pattern 5: Random smoothed
    noise = rng.standard_normal(n)
    kernel = np.ones(10) / 10
    h5 = np.convolve(noise, kernel, mode='same')
    
    # Pattern 6: Barker-like
    h6 = np.zeros(n)
    for j in range(n):
        if j % 4 == 0:
            h6[j] = 1.0
        elif j % 4 == 1:
            h6[j] = -1.0
        elif j % 4 == 2:
            h6[j] = 1.0
        else:
            h6[j] = -1.0
    
    return [h1, h2, h3, h4, h5, h6]


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Baseline
    base_score = _score(best_h)
    best_score = base_score
    
    # Generate initial patterns
    inits = _gen_initial(N, seed=42)
    
    # Run ground state search on each initial pattern
    for idx, h_init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        # Normalize to have sum = 1 for consistent comparison
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Run ground state search
        h_opt, val_opt = _ground_state_search(h_norm, t_start, deadline - 5 - idx * 5, max_iters=200)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
    
    # Final local search refinement on the best solution
    if time.time() < deadline - 2:
        h_final = best_h.copy()
        current_val = best_score
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                if time.time() > deadline - 1:
                    break
                h_final[i] = -h_final[i]
                new_val = _score(h_final)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    h_final[i] = -h_final[i]  # revert
        
        if current_val < best_score:
            best_score = current_val
            best_h = h_final.copy()
    
    return best_h
