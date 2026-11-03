"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _compute_acf(h):
    """Compute aperiodic autocorrelation via FFT, return ac[0..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    ac = _compute_acf(h)
    return float(np.max(np.abs(ac[1:])))


def _continuous_to_binary(f):
    """Convert continuous vector to binary signs."""
    return np.sign(f)


def _island_anneal(seed, time_budget, start_h=None):
    """
    Anneal starting from a given initial state (or random if None).
    Uses block flips with varying sizes.
    """
    rng = np.random.default_rng(seed)
    
    if start_h is not None:
        h = start_h.copy()
    else:
        h = rng.choice([-1.0, 1.0], size=N)
    
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 2.0
    T_min = 1e-8
    start = time.time()
    
    # Block size distribution: mostly small, occasionally large
    # Small blocks (1-4) for fine tuning, medium (5-16), large (17-32)
    
    while time.time() - start < time_budget:
        r = rng.random()
        if r < 0.4:
            L = int(rng.integers(1, 5))
        elif r < 0.7:
            L = int(rng.integers(4, 13))
        elif r < 0.9:
            L = int(rng.integers(8, 25))
        else:
            L = int(rng.integers(16, 33))
        L = min(L, N)
        
        i = int(rng.integers(0, N - L + 1))
        
        # Apply flip
        h[i:i+L] *= -1.0
        ac_new = _compute_acf(h)
        new_val = float(np.max(np.abs(ac_new[1:])))
        d = new_val - cur
        
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_val
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Undo
            h[i:i+L] *= -1.0
        
        T = max(T_min, T * 0.99998)
    
    return best_h, best


def _greedy_polish(h, time_budget, start_time):
    """
    Greedy single-flip polish: try each position, keep the best flip.
    """
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    
    while time.time() - start_time < time_budget:
        best_i = -1
        best_c = cur
        best_ac = None
        
        for i in range(N):
            h[i] *= -1.0
            ac_try = _compute_acf(h)
            c = float(np.max(np.abs(ac_try[1:])))
            if c < best_c:
                best_c = c
                best_i = i
                best_ac = ac_try.copy()
            h[i] *= -1.0
        
        if best_i < 0:
            break
        
        h[best_i] *= -1.0
        ac = best_ac
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    
    return best_h, best


def _island_continuous_init(seed):
    """
    Generate initial continuous vector via flat-spectrum optimization,
    then binarize. This provides a better starting point for annealing.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a random continuous vector
    f = rng.standard_normal(N)
    
    # Simple gradient descent on L2 norm of autocorrelation sidelobes
    # to push toward flat spectrum
    lr = 0.01
    for step in range(200):
        ac = _compute_acf(f)
        # Gradient of sum |ac[k]|^2 for k>=1
        # ac[k] = sum_j f[j] f[j+k]
        # d/d f[i] sum_k |ac[k]|^2 = 2 * sum_k ac[k] * (f[i+k] + f[i-k])
        # For simplicity, use a simpler update
        ac_sidelobes = ac[1:]
        # Gradient approximation: correlate ac_sidelobes with f
        # This is a heuristic; exact gradient is complex
        grad = np.zeros(N)
        for k in range(1, N):
            ac_k = ac[k]
            if abs(ac_k) > 1e-6:
                # Contribution to gradient at position i from lag k
                # d ac[k] / d f[i] = f[i+k] + f[i-k] (with boundaries)
                idx1 = np.arange(k, N)
                idx2 = np.arange(0, N-k)
                grad[idx1] += ac_k * f[idx2]
                grad[idx2] += ac_k * f[idx1]
        
        f -= lr * grad
        # Clip to avoid explosion
        f = np.clip(f, -10, 10)
    
    return _continuous_to_binary(f)


def construct():
    try:
        hard_deadline = 140.0
        start = time.time()
        
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)
        
        # Strategy: Multiple islands with different initializations
        # Island 1: Start from continuous flat-spectrum init
        budget1 = min(25.0, hard_deadline - (time.time() - start) - 10.0)
        if budget1 > 1.0:
            init_h = _island_continuous_init(42)
            h, c = _island_anneal(12345, budget1, start_h=init_h)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Island 2: Random init
        remaining = hard_deadline - (time.time() - start)
        budget2 = min(20.0, remaining - 15.0)
        if budget2 > 1.0:
            h, c = _island_anneal(99999, budget2)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Island 3: Another random restart
        remaining = hard_deadline - (time.time() - start)
        budget3 = min(15.0, remaining - 10.0)
        if budget3 > 1.0:
            h, c = _island_anneal(77777, budget3)
            if c < global_best:
                global_best = c
                global_best_h = h
        
        # Final greedy polish
        remaining = hard_deadline - (time.time() - start)
        if remaining > 3.0:
            h, c = _greedy_polish(global_best_h, remaining - 5.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
        
        return global_best_h
    except Exception:
        return np.ones(N)
# EVOLVE-BLOCK-END
