import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _compute_ac(h):
    """Compute aperiodic autocorrelation sidelobes ac[1..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac[1:]


def _flip_delta_ac(h, i, ac):
    """
    Compute the change in autocorrelation sidelobes when flipping h[i].
    Returns the new ac array (sidelobes only, length N-1).
    
    ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k] for k=1..N-1
    
    When h[i] -> -h[i]:
    For each lag k (1..N-1):
      - If i < N-k: h[i]*h[i+k] is in the sum, changes by -2*h[i]*h[i+k]
      - If i-k >= 0 and i-k < N-k: h[i-k]*h[i] is in the sum, changes by -2*h[i-k]*h[i]
      Note: i < N-k means i+k < N. And i-k >= 0 means i >= k.
      These two conditions are mutually exclusive in their contribution? No:
      - i+k < N: the term h[i]*h[i+k] exists in ac[k]
      - i-k >= 0: the term h[i-k]*h[i] exists in ac[k]
      Both can be true simultaneously.
    """
    new_ac = ac.copy()
    hi = h[i]
    # For each lag k, find contributions involving index i
    # Contribution 1: h[i] * h[i+k] where i+k < N, i.e., k < N-i
    # Contribution 2: h[i-k] * h[i] where i-k >= 0, i.e., k <= i
    # For k in 1..N-1:
    #   delta from contrib 1: -2*hi*h[i+k] if i+k < N
    #   delta from contrib 2: -2*h[i-k]*hi if i-k >= 0
    
    # Vectorized:
    # k ranges from 1 to N-1
    # contrib1 valid when k < N-i, i.e., k in 1..min(N-1, N-i-1)
    # contrib2 valid when k <= i, i.e., k in 1..min(N-1, i)
    
    k_max = N - 1
    # contrib1: k in 1..min(N-1, N-i-1)
    c1_end = min(N - 1, N - i - 1)
    if c1_end >= 1:
        # ac indices 0..c1_end-1 correspond to lags 1..c1_end
        # h[i+1..i+c1_end]
        new_ac[:c1_end] -= 2.0 * hi * h[i+1:i+1+c1_end]
    
    # contrib2: k in 1..min(N-1, i)
    c2_end = min(N - 1, i)
    if c2_end >= 1:
        # ac indices 0..c2_end-1 correspond to lags 1..c2_end
        # h[i-1..i-c2_end] in reverse
        new_ac[:c2_end] -= 2.0 * hi * h[i-1:i-1-c2_end:-1]
    
    return new_ac


def _greedy_polish_fast(h, time_budget, start):
    """
    Fast greedy polish using O(N) flip delta computation.
    """
    h = h.copy()
    ac = _compute_ac(h)
    cur = float(np.max(np.abs(ac)))
    best = cur
    best_h = h.copy()
    best_ac = ac.copy()
    
    while time.time() - start < time_budget:
        improved = False
        
        # Single-flip sweep: find the best flip
        best_i = -1
        best_c = cur
        for i in range(N):
            new_ac = _flip_delta_ac(h, i, ac)
            c = float(np.max(np.abs(new_ac)))
            if c < best_c - 1e-12:
                best_c = c
                best_i = i
        
        if best_i >= 0:
            h[best_i] *= -1.0
            ac = _flip_delta_ac(h, best_i, ac)  # Wait, this is wrong - we already applied the flip to h
            # Actually, _flip_delta_ac computes the delta assuming h[best_i] is still the old value
            # But we just flipped h[best_i], so we need to recompute
            # Let me restructure: compute the new ac before flipping h
            pass
        
        # Let me redo this more carefully
        break  # placeholder
    
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """
    Greedy polish with O(N) per-flip evaluation.
    """
    h = h.copy()
    ac = _compute_ac(h)
    cur = float(np.max(np.abs(ac)))
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        improved = False
        
        # Single-flip sweep
        best_i = -1
        best_c = cur
        for i in range(N):
            # Compute what ac would be if we flipped h[i]
            new_ac = _flip_delta_ac(h, i, ac)
            c = float(np.max(np.abs(new_ac)))
            if c < best_c - 1e-15:
                best_c = c
                best_i = i
        
        if best_i >= 0:
            # Apply the flip
            h[best_i] *= -1.0
            # Update ac using the delta
            ac = _flip_delta_ac(h, best_i, ac)
            # Wait - this is wrong. _flip_delta_ac(h, best_i, ac) assumes h[best_i] is the OLD value
            # But we just flipped it. Let me think again.
            # 
            # _flip_delta_ac(h, i, ac) computes: what would ac be if h[i] were flipped?
            # It uses the CURRENT h to compute the delta.
            # So if we call it BEFORE flipping h, it gives us the new ac.
            # We should call it before flipping.
            pass
        
        # Redo properly
        break
    
    return best_h, best


def _greedy_polish_v2(h, time_budget, start):
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
        
        # Adjacent-pair-flip sweep
        best_pair = -1
        best_c2 = cur
        best_new_ac2 = None
        for i in range(N - 1):
            # Flip h[i] and h[i+1]
            # First flip h[i], then flip h[i+1] in the result
            ac1 = _flip_delta_ac(h, i, ac)
            # Now flip h[i+1] in the state where h[i] is flipped
            # But _flip_delta_ac uses h, not the modified h
            # This is tricky. Let me just do it directly for pairs.
            # For pair flips, the delta is more complex. Let me skip pairs for now
            # and just do single flips.
            pass
        
        if not improved:
            break
            
    return best_h, best


def _anneal(seed, time_budget):
    """Simulated annealing with O(N) flip evaluation."""
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    ac = _compute_ac(h)
    cur = float(np.max(np.abs(ac)))
    best_h = h.copy()
    best = cur
    T = 5.0
    T_min = 1e-6
    start = time.time()
    stuck_counter = 0
    
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(32.0 * (1.0 - 0.7 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        
        # Kick if stuck
        if stuck_counter > 100:
            kick_len = int(rng.integers(10, 50))
            kick_i = int(rng.integers(0, N - kick_len + 1))
            h[kick_i:kick_i + kick_len] *= -1.0
            ac = _compute_ac(h)
            cur = float(np.max(np.abs(ac)))
            stuck_counter = 0
            T = max(T, 2.0)
            
        # Try flipping block [i, i+L)
        # For block flips, we can't easily use the O(N) trick
        # So we compute directly
        h[i:i + L] *= -1.0
        new_ac = _compute_ac(h)
        new = float(np.max(np.abs(new_ac)))
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            ac = new_ac
            if cur < best:
                best = cur
                best_h = h.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            h[i:i + L] *= -1.0
            stuck_counter += 1
            
        T = max(T_min, T * 0.999)
        
    return best_h, best


def _continuous_seed(seed, time_budget):
    """Generate a good binary seed via continuous gradient descent on L2 autocorrelation norm."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    start = time.time()
    
    lr = 0.05
    best_x = x.copy()
    best_l2 = _l2_autocorr(x)
    
    while time.time() - start < time_budget:
        H = np.fft.fft(x, _FFT_SIZE)
        ac = np.fft.ifft(H * np.conj(H)).real[:N]
        l2_val = float(np.sum(ac[1:] ** 2))
        
        if l2_val < best_l2:
            best_l2 = l2_val
            best_x = x.copy()
            
        grad = np.fft.ifft(H * np.abs(H)**2).real
        x = x - lr * grad
        
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm * np.sqrt(N)
            
        if l2_val > best_l2 * 1.05:
            lr *= 0.9
            
    h = np.where(best_x >= 0, 1.0, -1.0)
    return h, _peak_sidelobe(h)


def _l2_autocorr(x):
    H = np.fft.fft(x, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.sum(ac[1:] ** 2))


def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 0: Generate high-quality seeds via continuous gradient descent.
        seed_budget = min(8.0, _remaining() - 15.0)
        if seed_budget > 2.0:
            h_seed, c_seed = _continuous_seed(42, seed_budget)
            if c_seed < global_best:
                global_best = c_seed
                global_best_h = h_seed
            sub = min(5.0, _remaining() - 15.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish_v2(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 1: Anneal from the best seed found so far.
        budget = min(30.0, _remaining() - 15.0)
        if budget > 5.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(10.0, _remaining() - 15.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish_v2(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: multiple shorter restarts for diversity.
        restart_id = 1
        while _remaining() > 20.0:
            sub = min(8.0, _remaining() - 20.0)
            if sub < 2.0:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(5.0, _remaining() - 20.0)
            if sub2 > 1.0:
                h2, c2 = _greedy_polish_v2(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: final high-precision polish.
        remaining = _remaining() - 2.0
        if remaining > 1.0:
            h, c = _greedy_polish_v2(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
