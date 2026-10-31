"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
            
    # Phase 1: Structured Candidates
    # 1. All ones
    # 2. Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    try_candidate(alt)
    
    # 3. Zadoff-Chu sequences
    # Zadoff-Chu: x[n] = exp(-j * pi * u * n * (n+1) / N) for n=0..N-1
    # We take the real part or threshold it? The task asks for "signed step heights". 
    # Usually this implies +/- 1. Let's try thresholding the real part of ZC.
    for u in [1, 2, 3, 4, 5, 7, 11, 13, 17, 19]:
        if time.time() > deadline:
            break
        n = np.arange(N)
        # Ensure u and N are coprime for ideal properties, but let's just try
        zc = np.exp(-1j * np.pi * u * n * (n + 1) / N)
        x = np.where(zc.real > 0, 1.0, -1.0)
        try_candidate(x)
        
    # 4. Random +/- 1 with specific bias to ensure sum is not too small
    rng = np.random.default_rng(123)
    for bias in [0.5, 0.6, 0.7, 0.8, 0.9]:
        p = bias
        x = rng.choice([-1.0, 1.0], size=N, p=[1-p, p])
        try_candidate(x)

    # Phase 2: Simulated Annealing
    # Start from the best candidate found so far
    x = best_x.copy()
    current_score = best_score
    
    # Pre-compute FFT size
    size = 2 * N - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
        
    # Helper to quickly compute score from FFT
    def compute_score_from_fft(F):
        A = ifft(F * np.conj(F)).real
        m = np.max(np.abs(A[:size]))
        s = np.sum(x) # This is the bottleneck, summing x every time
        # Optimization: Keep track of sum separately
        if abs(s) < 1e-12:
            return np.inf
        return m / (s * s)
    
    # Simulated Annealing Loop
    # We will update x and F incrementally
    # F = fft(x, size2)
    F = fft(x, size2)
    current_sum = np.sum(x)
    
    # Initial temperature
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.9995
    
    # Number of steps per temperature
    steps_per_T = 50
    
    # Track best in SA
    sa_best_x = x.copy()
    sa_best_score = current_score
    
    while T > T_min:
        for _ in range(steps_per_T):
            if time.time() > deadline:
                return best_x
            
            # Flip a random set of bits (e.g., 1 to 5)
            n_flips = rng.integers(1, 6)
            flip_idx = rng.choice(N, size=n_flips, replace=False)
            
            # Calculate delta for sum
            # x[flip_idx] *= -1
            # new_val = -old_val
            # delta_sum = -2 * sum(x[flip_idx])
            old_vals = x[flip_idx]
            delta_sum = -2.0 * np.sum(old_vals)
            new_sum = current_sum + delta_sum
            
            # If new sum is too small, skip
            if abs(new_sum) < 1e-12:
                continue
                
            # Update x
            x[flip_idx] *= -1.0
            
            # Update F
            # F_new = F + fft(delta_x, size2)
            # delta_x is zero except at flip_idx
            delta_x = np.zeros(size2)
            delta_x[flip_idx] = -2.0 * old_vals # Since new = -old, delta = -old - old = -2old
            F += fft(delta_x, size2)
            
            # Compute new score
            A = ifft(F * np.conj(F)).real
            m_new = np.max(np.abs(A[:size]))
            score_new = m_new / (new_sum * new_sum)
            
            # Acceptance probability
            if score_new < current_score:
                # Always accept improvements
                current_score = score_new
                current_sum = new_sum
                if score_new < sa_best_score:
                    sa_best_score = score_new
                    sa_best_x = x.copy()
            else:
                # Boltzmann probability
                # Delta E = score_new - current_score
                delta_E = score_new - current_score
                if delta_E < 0:
                    accept = True
                else:
                    accept = np.exp(-delta_E / T) > rng.random()
                
                if accept:
                    current_score = score_new
                    current_sum = new_sum
                else:
                    # Revert
                    x[flip_idx] *= -1.0
                    F -= fft(delta_x, size2)
                    current_sum = new_sum - delta_sum # Revert sum
                    # Note: current_score remains unchanged
            
        T *= cooling_rate
    
    # Final best from SA
    if sa_best_score < best_score:
        best_x = sa_best_x
        best_score = sa_best_score
        
    # Phase 3: Local Refinement on Best
    # Simple hill climbing on the best found
    x = best_x.copy()
    sc = best_score
    F = fft(x, size2)
    current_sum = np.sum(x)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try flipping each bit one by one
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            
            old_val = x[i]
            new_val = -old_val
            delta_sum = -2.0 * old_val
            new_sum = current_sum + delta_sum
            
            if abs(new_sum) < 1e-12:
                continue
                
            # Update x
            x[i] = new_val
            
            # Update F
            delta_x = np.zeros(size2)
            delta_x[i] = -2.0 * old_val
            F += fft(delta_x, size2)
            
            A = ifft(F * np.conj(F)).real
            m_new = np.max(np.abs(A[:size]))
            score_new = m_new / (new_sum * new_sum)
            
            if score_new < sc - 1e-12:
                sc = score_new
                current_sum = new_sum
                improved = True
                best_x = x.copy()
                best_score = sc
            else:
                # Revert
                x[i] = old_val
                F -= fft(delta_x, size2)
                current_sum = new_sum - delta_sum
                
    return best_x
# EVOLVE-BLOCK-END
