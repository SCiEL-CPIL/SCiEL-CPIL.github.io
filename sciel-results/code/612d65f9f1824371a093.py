import numpy as np
import time

N = 400
DX = 0.5 / N  # Interval width

def _compute_c3_bound(h):
    """Compute the upper bound on C3 from the autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    
    # Compute autoconvolution via FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    
    # The autoconvolution at grid points is DX * acf[k]
    # Max absolute value over all lags
    max_abs = np.max(np.abs(acf))
    return DX * max_abs

def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _gradient_step_c3(h, acf):
    """
    Find the best flip to minimize the C3 bound.
    The C3 bound is DX * max|acf|, so minimizing max|acf| is equivalent.
    """
    n = len(h)
    current_c3 = DX * np.max(np.abs(acf))
    
    best_i = -1
    best_new_c3 = float('inf')
    
    acf_lags = acf[1:n]
    
    for i in range(n):
        neighbor_sum = np.zeros(n-1)
        
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
            
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        
        # New full acf: lag 0 changes, non-zero lags change
        new_acf_0 = acf[0] - 2.0 * h[i] * np.sum(h)  # This isn't quite right
        # Actually, flipping h[i] changes acf[0] by -2*h[i]*h[i] = -2 (since h[i]^2=1)
        # Wait, acf[0] = sum(h[j]^2) = N for ±1, so it's constant!
        # So only non-zero lags change.
        
        new_max = np.max(np.abs(new_acf_lags))
        new_c3 = DX * new_max
        
        if new_c3 < best_new_c3:
            best_new_c3 = new_c3
            best_i = i
            
    return best_i, float(best_new_c3), current_c3

def _fast_c3_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast C3 bound optimization using gradient method.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = np.fft.irfft(np.abs(np.fft.rfft(h, n=2*n)) ** 2, n=2*n)
    best_c3 = _compute_c3_bound(h)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_c3, cur_c3 = _gradient_step_c3(h, acf)
        
        if i == -1 or new_c3 >= cur_c3:
            break
        
        h[i] = -h[i]
        passes += 1
        
        acf = np.fft.irfft(np.abs(np.fft.rfft(h, n=2*n)) ** 2, n=2*n)
        cur_c3 = _compute_c3_bound(h)
        
        if cur_c3 < best_c3:
            best_c3 = cur_c3
            best_h = h.copy()
        
        if best_c3 < 1e-10:
            break
    
    return best_h, best_c3

def _simulated_annealing_c3(h, time_budget=30.0, start_time=None, rng=None):
    """
    Simulated annealing on C3 bound.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = np.fft.irfft(np.abs(np.fft.rfft(h, n=2*n)) ** 2, n=2*n)
    current_c3 = _compute_c3_bound(h)
    
    best_h = h.copy()
    best_c3 = current_c3
    
    T = 0.1
    T_min = 0.001
    cooling = 0.95
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = int(rng.integers(n))
        
        neighbor_sum = np.zeros(n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf[1:n] + delta
        new_c3 = DX * np.max(np.abs(new_acf_lags))
        
        delta_c3 = new_c3 - current_c3
        
        if delta_c3 < 0 or np.random.random() < np.exp(-delta_c3 / T):
            h[i] = -h[i]
            acf = np.fft.irfft(np.abs(np.fft.rfft(h, n=2*n)) ** 2, n=2*n)
            current_c3 = _compute_c3_bound(h)
            
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_c3

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random
    for i in range(15):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(8):
        phases = rng.uniform(0, 2*np.pi, size=6)
        amps = rng.uniform(0.3, 1.0, size=6)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 7):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 6, 7, 8, 11, 13, 17]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 4. Block structure
    for block_size in [5, 10, 20, 40]:
        for i in range(3):
            h = np.zeros(N)
            for b in range(N // block_size):
                h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
            candidates.append(h)
            
    # 5. Alternating blocks
    for block_size in [10, 20, 50]:
        h = np.zeros(N)
        sign = 1.0
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = sign
            sign = -sign
        candidates.append(h)
        
    # Evaluate all candidates and keep the best few
    scored_candidates = []
    for h in candidates:
        s = _compute_c3_bound(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 8 candidates with fast C3 optimization
    for i, (s, h) in enumerate(scored_candidates[:8]):
        if time.time() > deadline - 15:
            break
        time_limit = deadline - 10
        h_refined, s_refined = _fast_c3_optimize(h, max_passes=300, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # Random restarts with fast C3 optimization
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 3)
        h_refined, s_refined = _fast_c3_optimize(h_init, max_passes=80, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # SA on the best found so far
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(8.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_c3(best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
                
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
