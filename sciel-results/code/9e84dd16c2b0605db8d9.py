import numpy as np
import time

N = 400

def _soft_max_acf(acf_lags, beta=50.0):
    """Smooth surrogate for max|acf| using log-sum-exp."""
    x = np.abs(acf_lags)
    m = np.max(x)
    if m < 1e-15:
        return 0.0
    # Soft max: (1/beta) * log(sum(exp(beta*x)))
    # For stability, use: m + (1/beta)*log(sum(exp(beta*(x-m))))
    scaled = beta * (x - m)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:N]))

def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _gradient_step(h, acf):
    """
    Gordon's fast PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    best_i = -1
    best_new_psl = float('inf')
    
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
        new_psl = np.max(np.abs(new_acf_lags))
        
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i
            
    return best_i, float(best_new_psl), current_psl

def _soft_gradient_step(h, acf, beta=50.0):
    """
    Gradient step using smooth surrogate (soft-max) of PSL.
    Returns best flip based on soft-max objective.
    """
    n = len(h)
    acf_lags = acf[1:n]
    
    best_i = -1
    best_soft = float('inf')
    best_hard = float('inf')
    
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
        new_soft = _soft_max_acf(new_acf_lags, beta)
        new_hard = np.max(np.abs(new_acf_lags))
        
        if new_soft < best_soft:
            best_soft = new_soft
            best_i = i
            best_hard = new_hard
            
    return best_i, float(best_hard), float(best_soft)

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None, use_soft=False):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    If use_soft=True, use soft-max surrogate to escape local minima.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    stuck_count = 0
    
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        if use_soft:
            i, new_psl, _ = _soft_gradient_step(h, acf, beta=50.0)
        else:
            i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if i == -1 or new_psl >= _psl_value(acf, n):
            # Stuck - if using soft, try a few more with higher beta
            stuck_count += 1
            if use_soft and stuck_count < 3:
                # Try with different beta values
                for b in [20.0, 100.0, 200.0]:
                    i2, new_psl2, _ = _soft_gradient_step(h, acf, beta=b)
                    if i2 != -1 and new_psl2 < _psl_value(acf, n):
                        i = i2
                        new_psl = new_psl2
                        break
                else:
                    break
            else:
                break
        
        h[i] = -h[i]
        passes += 1
        stuck_count = 0
        
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _simulated_annealing_psl(h, time_budget=30.0, start_time=None, rng=None):
    """
    Simulated annealing on PSL using the fast gradient to evaluate moves.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 10.0
    T_min = 0.1
    cooling = 0.95
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = int(rng.integers(n))
        
        # Compute delta for this specific i
        neighbor_sum = np.zeros(n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(5):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 4. Block structure
    block_size = 10
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
    
    # 5. Thue-Morse sequence
    tm = np.zeros(N)
    for i in range(N):
        # Count bits in i
        tm[i] = -1.0 if (bin(i).count('1') % 2 == 0) else 1.0
    candidates.append(tm)
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Phase 1: Refine top candidates with standard greedy
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 30:
            break
        time_limit = deadline - 20
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=500, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Phase 2: Try soft-max guided optimization on best candidates
    if best_h is not None and time.time() < deadline - 15:
        # Use soft-max to potentially escape local minima
        h_soft, s_soft = _fast_psl_optimize(best_h, max_passes=300, time_budget=deadline - 10, start_time=start_time, use_soft=True)
        if s_soft < best_score:
            best_score = s_soft
            best_h = h_soft.copy()
    
    # Phase 3: Random restarts with mixed greedy/soft approach
    while time.time() < deadline - 15:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Alternate between standard and soft-guided optimization
        use_soft = (rng.random() < 0.3)
        time_limit = min(deadline - 10, time.time() + 8)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=150, time_budget=time_limit, start_time=start_time, use_soft=use_soft)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Phase 4: Final polish with SA on the best found
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(12.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 3.0:
            h_sa, s_sa = _simulated_annealing_psl(best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
    
    # Phase 5: Final greedy polish pass
    if best_h is not None and time.time() < deadline - 2:
        h_polish, s_polish = _fast_psl_optimize(best_h, max_passes=200, time_budget=deadline - 1, start_time=start_time)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()
                
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
