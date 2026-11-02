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
    """
    new_ac = ac.copy()
    hi = h[i]
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


def _pair_flip_delta_ac(h, i, j, ac):
    """
    Compute the new ac array when flipping both h[i] and h[j].
    """
    if i == j:
        return _flip_delta_ac(h, i, ac)
    
    # Flip i first, then flip j in the result
    # But we need to be careful: after flipping i, the state changes
    # Let's compute the combined effect directly
    
    new_ac = ac.copy()
    hi = h[i]
    hj = h[j]
    
    # Effect of flipping i:
    # contrib1: k in 1..min(N-1, N-i-1): -2*hi*h[i+k]
    # contrib2: k in 1..min(N-1, i): -2*h[i-k]*hi
    
    # Effect of flipping j:
    # contrib1: k in 1..min(N-1, N-j-1): -2*hj*h[j+k]
    # contrib2: k in 1..min(N-1, j): -2*h[j-k]*hj
    
    # But there's an interaction: if i and j are related by a lag, 
    # the cross terms matter.
    # 
    # Actually, let's just apply the flips sequentially using _flip_delta_ac
    # but we need the intermediate state. Let me just do it directly.
    
    # For each lag k:
    # ac_new[k] = ac[k] - 2*hi*(h[i+k] if i+k<N else 0) - 2*h[i-k]*(1 if i-k>=0 else 0)*hi
    #            - 2*hj*(h[j+k] if j+k<N else 0) - 2*h[j-k]*(1 if j-k>=0 else 0)*hj
    # But wait, this isn't quite right because the two flips interact.
    # 
    # Let me think more carefully. The autocorrelation is:
    # ac[k] = sum_{m} h[m]*h[m+k]
    # 
    # After flipping i and j:
    # ac_new[k] = sum_{m} h'[m]*h'[m+k]
    # where h'[m] = -h[m] if m in {i,j}, else h[m]
    # 
    # = sum_{m not in {i,j}} h[m]*h[m+k] 
    #   + (if i not in {m, m+k} for the terms involving i)
    # 
    # This is getting complex. Let me just use the sequential approach:
    # First compute ac1 = _flip_delta_ac(h, i, ac)
    # Then compute ac2 = _flip_delta_ac(h_with_i_flipped, j, ac1)
    # But we don't have h_with_i_flipped easily.
    
    # Simpler: just apply both deltas, accounting for the interaction.
    # The interaction term: if |i-j| = k, then the term h[i]*h[j] appears in ac[k].
    # When both are flipped, h[i]*h[j] -> (-h[i])*(-h[j]) = h[i]*h[j], so no change.
    # But the individual terms change.
    
    # Let me just do it sequentially with a temporary array.
    h_temp = h.copy()
    h_temp[i] *= -1.0
    ac1 = _flip_delta_ac(h, i, ac)
    ac2 = _flip_delta_ac(h_temp, j, ac1)
    
    return ac2


def _greedy_polish_v3(h, time_budget, start):
    """
    Greedy polish with correct O(N) per-flip evaluation and pair flips.
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
            ac1 = _flip_delta_ac(h, i, ac)
            # Now flip h[i+1] in the state where h[i] is flipped
            h_temp = h.copy()
            h_temp[i] *= -1.0
            ac2 = _flip_delta_ac(h_temp, i + 1, ac1)
            c = float(np.max(np.abs(ac2)))
            if c < best_c2 - 1e-15:
                best_c2 = c
                best_pair = i
                best_new_ac2 = ac2
        
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            ac = best_new_ac2
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        
        if not improved:
            break
            
    return best_h, best


def _anneal_v2(seed, time_budget):
    """Simulated annealing with O(N) flip evaluation and adaptive step size."""
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


def _continuous_seed_v2(seed, time_budget):
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


def _sine_init(freq):
    """Initialize with a sine wave pattern."""
    t = np.linspace(0, freq * 2 * np.pi, N)
    h = np.where(np.sin(t) >= 0, 1.0, -1.0)
    return h


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
            h_seed, c_seed = _continuous_seed_v2(42, seed_budget)
            if c_seed < global_best:
                global_best = c_seed
                global_best_h = h_seed
            sub = min(5.0, _remaining() - 15.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish_v3(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Try sine wave initializations
        for freq in [1, 2, 3, 5, 7, 10, 13, 17, 20, 25]:
            if _remaining() < 15.0:
                break
            h_sine = _sine_init(freq)
            c_sine = _peak_sidelobe(h_sine)
            if c_sine < global_best:
                global_best = c_sine
                global_best_h = h_sine
            sub = min(3.0, _remaining() - 15.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish_v3(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 1: Anneal from the best seed found so far.
        budget = min(30.0, _remaining() - 15.0)
        if budget > 5.0:
            h, c = _anneal_v2(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(10.0, _remaining() - 15.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish_v3(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: multiple shorter restarts for diversity.
        restart_id = 1
        while _remaining() > 20.0:
            sub = min(8.0, _remaining() - 20.0)
            if sub < 2.0:
                break
            h, c = _anneal_v2(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(5.0, _remaining() - 20.0)
            if sub2 > 1.0:
                h2, c2 = _greedy_polish_v3(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: final high-precision polish.
        remaining = _remaining() - 2.0
        if remaining > 1.0:
            h, c = _greedy_polish_v3(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
