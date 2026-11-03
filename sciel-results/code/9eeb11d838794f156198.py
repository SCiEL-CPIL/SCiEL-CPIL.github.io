"""Search for a low-autocorrelation signed step function (400 equal intervals).

Combines simulated annealing with fast O(N) in-place autocorrelation updates
for the peak sidelobe objective. Uses Thue-Morse initialization and optimized
inner loop for higher iteration throughput.
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (values -1, +1)."""
    h = np.ones(n)
    for i in range(1, n):
        h[i] = -h[i // 2]
    return h


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _fast_flip_update(h, ac, i):
    """
    Flip h[i] and update ac in place.
    Returns the new peak sidelobe value.
    """
    val = h[i]
    
    # ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
    # Flipping h[i] affects terms where j=i (if i+k<N) or j+k=i i.e. j=i-k (if i-k>=0)
    # delta_ac[k] = -2*val*(h[i+k]*1_{i+k<N} + h[i-k]*1_{i-k>=0})
    
    # For k where i+k < N: delta = -2*val*h[i+k]
    # k ranges from 1 to N-1-i
    if i + 1 < N:
        ac[1:N-i] -= 2.0 * val * h[i+1:N]
    # For k where i-k >= 0: delta = -2*val*h[i-k]
    # k ranges from 1 to i
    if i > 0:
        ac[1:i+1] -= 2.0 * val * h[i-1::-1]
        
    # Flip h[i]
    h[i] = -val
    
    # Calculate new peak sidelobe
    return float(np.max(np.abs(ac[1:])))


def _compute_acf(h):
    """Compute aperiodic autocorrelation via FFT, return ac[0..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _anneal(seed, time_budget, init_type="thue"):
    """Simulated annealing with fast single-flip updates."""
    rng = np.random.default_rng(seed)
    
    # Initialize based on type
    if init_type == "thue":
        h = _thue_morse(N).copy()
        # Apply random perturbations for diversity
        n_perturb = N // 4
        idx = rng.choice(N, size=n_perturb, replace=False)
        h[idx] *= -1.0
    else:
        h = rng.choice([-1.0, 1.0], size=N)
    
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 2.0
    T_min = 1e-9
    start = time.time()
    n_steps = 0
    step_counter = 0
    
    while time.time() - start < time_budget:
        # Use single flips with fast update
        i = int(rng.integers(0, N))
        
        # Apply flip
        new_val = _fast_flip_update(h, ac, i)
        
        # Use exact peak for acceptance (faster than soft peak for N=400)
        d = new_val - cur
        
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_val
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Undo
            _fast_flip_update(h, ac, i)
            
        n_steps += 1
        step_counter += 1
        
        # Adaptive cooling: exponential decay
        frac = min(1.0, n_steps / 50000.0)
        T = max(T_min, 2.0 * np.exp(-3.0 * frac))
        
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Greedy single-flip polish: try each position, keep the best flip."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        
        for i in range(N):
            # Do the flip, check peak, undo
            _fast_flip_update(h, ac, i)
            c = float(np.max(np.abs(ac[1:])))
            
            if c < best_c:
                best_c = c
                best_i = i
            # Undo i
            _fast_flip_update(h, ac, i)
            
        if best_i < 0:
            break
            
        # Apply the best flip
        _fast_flip_update(h, ac, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
            
    return best_h, best


def construct():
    try:
        hard_deadline = 145.0
        start = time.time()
        global_best_h = _thue_morse(N).copy()
        global_best = _peak_sidelobe(global_best_h)

        # Long initial anneal from Thue-Morse with perturbations.
        budget = min(60.0, hard_deadline - (time.time() - start) - 15.0)
        if budget > 1.0:
            h, c = _anneal(12345, budget, init_type="thue")
            if c < global_best:
                global_best = c
                global_best_h = h

        # Multiple shorter restarts for diversity.
        restart_id = 1
        while time.time() - start < hard_deadline - 25.0:
            sub = min(10.0, hard_deadline - (time.time() - start) - 25.0)
            if sub < 1.0:
                break
            # Alternate between Thue-Morse and random inits
            init_type = "thue" if restart_id % 2 == 1 else "random"
            h, c = _anneal(100000 + restart_id, sub, init_type=init_type)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1

        # Final greedy single-flip polish with whatever time remains.
        remaining = hard_deadline - (time.time() - start)
        if remaining > 5.0:
            h, c = _greedy_polish(global_best_h, remaining - 5.0, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Safety: ensure we return before hard deadline
        if time.time() - start > hard_deadline - 3.0:
            return global_best_h

        return global_best_h
    except Exception:
        return np.ones(N)
