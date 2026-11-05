"""Rewritten program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_value(h):
    """Compute the objective value: abs(2*N*max(conv(h,h))/sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start with the seed
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    # Strategy: Use structured candidates with sign changes to reduce autoconv peak.
    # The key insight: if we can make the autoconvolution have a smaller maximum
    # while keeping the sum non-zero, we improve the score.
    
    # Approach 1: Try simple analytic forms
    candidates = []
    
    # Sine wave with various frequencies
    for freq in range(1, 20):
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h = np.sin(2*np.pi*freq*np.arange(N)/N + phase)
            candidates.append(h)
    
    # Cosine wave
    for freq in range(1, 20):
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h = np.cos(2*np.pi*freq*np.arange(N)/N + phase)
            candidates.append(h)
    
    # Multi-lobe patterns
    for nlobes in [2, 3, 4, 5]:
        x = np.linspace(0, 2*np.pi, N)
        h = np.sin(nlobes * x)
        candidates.append(h)
        h = np.cos(nlobes * x)
        candidates.append(h)
    
    # Sawtooth-like patterns
    for period in [2, 4, 8, 10, 16, 20, 40, 50, 80, 100]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = (i % period) / period - 0.5
        candidates.append(h)
    
    # Random structured patterns with sign changes
    rng = np.random.RandomState(42)
    for trial in range(100):
        # Random frequency mix
        n_freqs = rng.randint(1, 5)
        h = np.zeros(N)
        for _ in range(n_freqs):
            freq = rng.uniform(0.5, 20)
            phase = rng.uniform(0, 2*np.pi)
            amp = rng.uniform(0.5, 1.0)
            h += amp * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        # Normalize to keep sum reasonable
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Approach 2: Local refinement using gradient-free optimization
    # Use Nelder-Mead style perturbation around best candidate
    if time.time() < deadline - 10:
        current_h = best_h.copy()
        current_val = best_val
        
        # Try small perturbations
        rng2 = np.random.RandomState(123)
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            step_size = 0.1 * np.std(current_h) + 1e-10
            perturbation = rng2.randn(N) * step_size
            trial_h = current_h + perturbation
            
            val = _compute_value(trial_h)
            if val < current_val:
                current_h = trial_h
                current_val = val
                if val < best_val:
                    best_val = val
                    best_h = current_h.copy()
            else:
                # Try smaller step
                trial_h = current_h + perturbation * 0.1
                val = _compute_value(trial_h)
                if val < current_val:
                    current_h = trial_h
                    current_val = val
                    if val < best_val:
                        best_val = val
                        best_h = current_h.copy()
        
        # More systematic local search
        if time.time() < deadline - 5:
            # Try shifting the best pattern
            for shift in range(1, 20):
                shifted = np.roll(best_h, shift)
                val = _compute_value(shifted)
                if val < best_val:
                    best_val = val
                    best_h = shifted.copy()
            
            # Try scaling
            for scale in [0.5, 0.7, 1.3, 1.5, 2.0]:
                scaled = best_h * scale
                val = _compute_value(scaled)
                if val < best_val:
                    best_val = val
                    best_h = scaled.copy()
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
