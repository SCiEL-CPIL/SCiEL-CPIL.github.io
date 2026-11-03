import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _compute_ac(h):
    """Compute aperiodic autocorrelation sidelobes ac[1..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac[1:]


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    ac = _compute_ac(h)
    return float(np.max(np.abs(ac)))


def _flip_delta_ac(h, i, ac):
    """
    Compute the change in autocorrelation sidelobes when flipping h[i].
    Returns the new ac array (sidelobes only, length N-1).
    
    ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k] for k=1..N-1
    
    When h[i] -> -h[i]:
    For each lag k (1..N-1):
      - If i < N-k: h[i]*h[i+k] is in the sum, changes by -2*h[i]*h[i+k]
      - If i-k >= 0 and i-k < N-k: h[i-k]*h[i] is in the sum, changes by -2*h[i-k]*h[i]
    """
    new_ac = ac.copy()
    hi = h[i]
    
    # contrib1: k in 1..min(N-1, N-i-1)
    # ac indices 0..c1_end-1 correspond to lags 1..c1_end
    c1_end = min(N - 1, N - i - 1)
    if c1_end >= 1:
        # h[i+1..i+c1_end]
        new_ac[:c1_end] -= 2.0 * hi * h[i+1:i+1+c1_end]
    
    # contrib2: k in 1..min(N-1, i)
    # ac indices 0..c2_end-1 correspond to lags 1..c2_end
    c2_end = min(N - 1, i)
    if c2_end >= 1:
        # h[i-1..i-c2_end] in reverse
        new_ac[:c2_end] -= 2.0 * hi * h[i-1:i-1-c2_end:-1]
    
    return new_ac


def _swap_delta_ac(h, i, j, ac):
    """
    Compute the change in autocorrelation sidelobes when swapping h[i] and h[j].
    If h[i] == h[j], swap is identity.
    If h[i] != h[j], it's equivalent to flipping both.
    """
    if h[i] == h[j]:
        return ac
    
    # Flip i then flip j
    # But delta for flip depends on current h.
    # Let's just compute new ac from scratch for swaps? 
    # N=400, swap is O(N^2) if done naively via convolution, but we can use delta.
    # Actually, swapping i and j is flipping i and flipping j.
    # The interaction term between i and j appears in the delta.
    # Let's stick to single flips for the main loop, it's simpler and effective.
    # Or we can just do two sequential flips if we update h.
    # For now, let's just use single flips in the annealer.
    pass

def _sine_init(frequency):
    """Generate a binary sequence based on a sine wave."""
    t = np.arange(N)
    # Scale frequency to cycles per length N
    # We want to avoid trivial periodicity
    wave = np.sin(2 * np.pi * frequency * t / N)
    h = np.where(wave >= 0, 1.0, -1.0)
    return h

def _anneal(seed, time_budget):
    """Simulated annealing with O(N) flip evaluation."""
    rng = np.random.default_rng(seed)
    
    # Try a few different sine frequencies as starting points
    # or just start with a good one
    # Let's start with a random one first, then try sine if stuck?
    # No, let's use sine init for the main anneal.
    
    # Pick a frequency that is not an integer to avoid perfect periodicity
    # e.g., 0.5, 0.7, 1.3
    freqs = [0.5, 0.7, 1.3, 2.1, 0.3]
    start_freq = freqs[int(rng.integers(0, len(freqs)))]
    h = _sine_init(start_freq)
    
    # Add some noise to break symmetry
    noise_mask = rng.random(N) < 0.05
    h[noise_mask] *= -1.0
    
    ac = _compute_ac(h)
    cur = float(np.max(np.abs(ac)))
    best_h = h.copy()
    best = cur
    T = 10.0
    T_min = 1e-6
    start = time.time()
    stuck_counter = 0
    
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        # Cool down
        T = max(T_min, T * 0.99)
        
        # If stuck, increase T and kick
        if stuck_counter > 200:
            T = max(T, 5.0)
            # Kick: flip a random block
            L = int(rng.integers(5, 20))
            i = int(rng.integers(0, N - L))
            h[i:i+L] *= -1.0
            ac = _compute_ac(h)
            cur = float(np.max(np.abs(ac)))
            stuck_counter = 0
            
        # Pick a random index to flip
        i = int(rng.integers(0, N))
        
        # Calculate new ac
        new_ac = _flip_delta_ac(h, i, ac)
        new_val = float(np.max(np.abs(new_ac)))
        
        d = new_val - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            h[i] *= -1.0
            ac = new_ac
            cur = new_val
            if cur < best:
                best = cur
                best_h = h.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            stuck_counter += 1
            
    return best_h, best

def _greedy_polish(h, time_budget, start):
    """
    Greedy polish with correct O(N) per-flip evaluation.
    """
    h = h.copy()
    ac = _compute_ac(h)
    cur = float(np.max(np.abs(ac)))
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        improved = False
        
        # Single-flip sweep: find the flip that most reduces peak sidelobe
        best_i = -1
        best_c = cur
        best_new_ac = None
        for i in range(N):
            # _flip_delta_ac(h, i, ac) gives new ac if h[i] is flipped
            new_ac = _flip_delta_ac(h, i, ac)
            c = float(np.max(np.abs(new_ac)))
            if c < best_c - 1e-15:
                best_c = c
                best_i = i
                best_new_ac = new_ac
        
        if best_i >= 0:
            h[best_i] *= -1.0
            ac = best_new_ac
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        
        if not improved:
            break
            
    return best_h, best


def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = _sine_init(0.5)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: Multiple annealing runs with different seeds/frequencies
        # We have ~140s. Let's do 4 runs of 30s each, then polish.
        seeds = [1, 2, 3, 4]
        
        for s in seeds:
            if _remaining() < 35.0:
                break
            budget = min(30.0, _remaining() - 10.0)
            if budget < 5.0:
                break
            
            h, c = _anneal(s, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            
            # Quick polish after each anneal
            if _remaining() > 5.0:
                h2, c2 = _greedy_polish(global_best_h, min(2.0, _remaining() - 5.0), time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: Final high-precision polish
        remaining = _remaining() - 2.0
        if remaining > 1.0:
            h, c = _greedy_polish(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        # Fallback to all ones
        return np.ones(N)
