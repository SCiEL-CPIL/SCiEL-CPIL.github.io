import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

# ---------------------------------------------------------------------------
# Autocorrelation Helpers
# ---------------------------------------------------------------------------

def _ac_full(h):
    """Full aperiodic autocorrelation ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k], k=0..N-1."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac

def _peak_sidelobe(h):
    ac = _ac_full(h)
    return float(np.max(np.abs(ac[1:])))

def _flip_ac_fast(ac, h, i):
    """
    Vectorised O(N) update of ac after flipping h[i].
    h is the array BEFORE the flip.
    Returns the new ac array.
    """
    new_ac = ac.copy()
    hi = h[i]
    
    # Update lags k > 0
    # 1. h[i] is the first element: pairs with h[i+k] for k=1..N-1-i
    #    Delta: -2 * h[i] * h[i+k]
    #    Indices in new_ac: 1 to N-1-i
    #    Values from h: h[i+1 : N]
    max_k1 = N - 1 - i
    if max_k1 >= 1:
        new_ac[1:max_k1 + 1] += -2.0 * hi * h[i + 1:i + max_k1 + 1]
        
    # 2. h[i] is the second element: pairs with h[i-k] for k=1..i
    #    Delta: -2 * h[i-k] * h[i]
    #    Indices in new_ac: 1 to i
    #    Values from h: h[0:i] reversed?
    #    For k=1, partner is h[i-1]. For k=i, partner is h[0].
    #    So the sequence of partners for k=1..i is h[i-1], h[i-2], ..., h[0].
    #    This is h[0:i][::-1].
    if i >= 1:
        new_ac[1:i + 1] += -2.0 * hi * h[0:i][::-1]
        
    # new_ac[0] remains unchanged (sum of squares)
    return new_ac

def _flip_pair_ac_fast(ac, h, i):
    """Update ac after flipping h[i] and h[i+1]."""
    # Flip i
    new_ac = _flip_ac_fast(ac, h, i)
    # Flip i+1 on the updated sequence
    h_tmp = h.copy()
    h_tmp[i] *= -1.0
    new_ac = _flip_ac_fast(new_ac, h_tmp, i + 1)
    return new_ac

# ---------------------------------------------------------------------------
# Search Algorithms
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start):
    """
    Greedy single-flip and pair-flip polish.
    """
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    
    # We will do passes until no improvement or time runs out
    while time.time() - start < time_budget:
        improved = False
        
        # Single flip sweep
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < cur:
                h[i] *= -1.0
                ac = ac2
                cur = c
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()
                    
        # Pair flip sweep
        for i in range(N - 1):
            ac2 = _flip_pair_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < cur:
                h[i] *= -1.0
                h[i+1] *= -1.0
                ac = ac2
                cur = c
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()
                    
        if not improved:
            break
            
    return best_h, best

def _anneal(seed, time_budget):
    """
    Simulated Annealing with block flips.
    """
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    
    T = 4.0
    T_min = 1e-7
    start = time.time()
    
    while time.time() - start < time_budget:
        # Determine block length
        progress = (time.time() - start) / time_budget
        max_len = max(1, int(32.0 * (1.0 - 0.8 * progress)))
        L = int(rng.integers(1, max_len + 1))
        
        i = int(rng.integers(0, N - L + 1))
        
        # Flip block
        h[i:i + L] *= -1.0
        ac_new = _ac_full(h)
        new = _peak_sidelobe(h)
        
        # Acceptance criterion
        delta = new - cur
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            cur = new
            ac = ac_new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0 # Revert
            
        T = max(T_min, T * 0.999)
        
    return best_h, best

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()
        
        def _remaining():
            return hard_deadline - (time.time() - start)
            
        # Initialize with a random sequence to immediately beat the all-ones baseline
        rng_init = np.random.default_rng(42)
        global_best_h = rng_init.choice([-1.0, 1.0], size=N)
        global_best = _peak_sidelobe(global_best_h)
        
        # Phase 1: Quick Greedy Polish on the initial random sequence
        budget = min(5.0, _remaining() - 10.0)
        if budget > 1.0:
            h, c = _greedy_polish(global_best_h, budget, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
                
        # Phase 2: Simulated Annealing with Block Flips
        budget = min(30.0, _remaining() - 20.0)
        if budget > 5.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Polish the result of annealing
            sub = min(5.0, _remaining() - 20.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
                    
        # Phase 3: More Annealing Restarts if time permits
        restart_id = 100
        while _remaining() > 15.0:
            sub = min(10.0, _remaining() - 15.0)
            if sub < 2.0:
                break
            h, c = _anneal(restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 15.0)
            if sub2 > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 100
            
        # Phase 4: Final High-Precision Polish
        remaining = _remaining() - 2.0
        if remaining > 0.5:
            h, c = _greedy_polish(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
                
        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
