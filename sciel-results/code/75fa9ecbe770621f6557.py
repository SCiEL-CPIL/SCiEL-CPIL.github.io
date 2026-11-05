import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return float(np.max(np.abs(acf[1:n])))

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

def _soft_max_psl(acf_lags, temperature=1.0):
    """
    Compute a smooth surrogate for max|acf_lags| using log-sum-exp.
    This provides gradient information about which lags are contributing
    to the peak, helping escape local minima.
    """
    x = np.abs(acf_lags)
    # Soft max: exp(x/T) / sum(exp(x/T)) * T
    # For numerical stability, subtract max
    x_max = np.max(x)
    if x_max < 1e-10:
        return 0.0
    exp_x = np.exp((x - x_max) / temperature)
    soft_max = x_max + temperature * np.log(np.sum(exp_x))
    return float(soft_max)

def _gradient_step_smooth(h, acf, temperature=0.5):
    """
    Gordon's fast PSL minimization step with smooth surrogate.
    Uses soft-max to get a smoother objective for finding better flip directions.
    Returns (best_i, best_new_psl, current_psl).
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    acf_lags = acf[1:n]
    
    # Compute soft-max surrogate value
    current_soft = _soft_max_psl(acf_lags, temperature)
    
    best_i = -1
    best_new_psl = float('inf')
    best_new_soft = float('inf')
    
    # Precompute neighbor sums efficiently using FFT-based convolution
    # For each position i, neighbor_sum[k-1] = h[i-k] + h[i+k]
    # We can compute all neighbor sums at once using correlation
    
    # h_conv_left[i, k] = h[i-k] for k=1..n-1 (left neighbors)
    # h_conv_right[i, k] = h[i+k] for k=1..n-1 (right neighbors)
    
    # Use the fact that we need: for each i, sum over k of |acf_lags[k-1] - 2*h[i]*(h[i-k]+h[i+k])|
    # But that's still O(n^2). Let's use a smarter approach.
    
    # For the smooth surrogate, the gradient w.r.t. flipping h[i] is:
    # d(soft_max)/dh[i] = sum_k softmax_weight[k] * d|acf_lags[k]|/dh[i]
    # where d|acf_lags[k]|/dh[i] = -2*h[i]*sign(acf_lags[k])*(h[i-k]+h[i+k])
    
    # Compute softmax weights
    x = np.abs(acf_lags)
    x_max = np.max(x)
    if x_max > 1e-10:
        exp_x = np.exp((x - x_max) / temperature)
        softmax_weights = exp_x / np.sum(exp_x)
        # Gradient of soft_max w.r.t. each |acf_lags[k]|
        grad_weights = softmax_weights
    else:
        grad_weights = np.ones(n-1) / (n-1)
    
    # Now for each i, the change in soft_max when flipping h[i] is approximately:
    # sum_k grad_weights[k] * (-2*h[i]*sign(acf_lags[k])*(h[i-k]+h[i+k]))
    # = -2*h[i] * sum_k grad_weights[k] * sign(acf_lags[k]) * (h[i-k]+h[i+k])
    
    # Let's define: g[k] = grad_weights[k] * sign(acf_lags[k])
    # Then for each i: delta_soft[i] = -2*h[i] * (sum_k g[k]*h[i-k] + sum_k g[k]*h[i+k])
    # = -2*h[i] * (correlation of h with g at lag i, left + right)
    
    g = grad_weights * np.sign(acf_lags)
    
    # Compute sum_k g[k]*h[i-k] for all i: this is cross-correlation
    # conv(h, g) at appropriate lags
    # For left: sum_{k=1}^{n-1} g[k-1] * h[i-k]
    # For right: sum_{k=1}^{n-1} g[k-1] * h[i+k]
    
    # Use FFT for efficiency
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    H = np.fft.rfft(h, n=fft_size)
    G = np.fft.rfft(g, n=fft_size)
    
    # Cross-correlation: sum_k g[k]*h[i+k] is like convolving h with reversed g
    # Let's compute both correlations
    # corr_left[i] = sum_{k=1}^{min(i, n-1)} g[k-1] * h[i-k]
    # corr_right[i] = sum_{k=1}^{min(n-1-i, n-1)} g[k-1] * h[i+k]
    
    # corr_right[i] = (h * g)[i] in convolution sense... let me think more carefully
    # sum_{k=1}^{n-1} g[k-1] * h[i+k] = sum_{j=i+1}^{i+n-1} g[j-i-1] * h[j]
    # This is the cross-correlation of h and g shifted
    
    # Use np.correlate or manual FFT
    # corr_right[i] = sum_k g[k-1] * h[i+k]
    # Let's define g_full = [0, g[0], g[1], ..., g[n-2]] (length n)
    # Then corr_right[i] = sum_{k=0}^{n-2} g_full[k] * h[i+1+k]
    # = (g_full * h)[i+1] in convolution (with appropriate indexing)
    
    # Simpler: just compute it directly using the FFT of h and g
    # corr_right[i] = sum_{k=1}^{n-1-i} g[k-1] * h[i+k]
    # This is like the cross-correlation at lag (i+1) of sequences g and h
    
    # Let me just do it with a direct approach but vectorized
    # For memory efficiency, process in batches
    
    best_i = -1
    best_new_psl = float('inf')
    
    batch_size = 100
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        
        for i in range(start, end):
            # Compute neighbor sum weighted by g
            left_sum = 0.0
            max_k1 = min(i, n-1)
            if max_k1 > 0:
                left_sum = np.dot(g[:max_k1], h[i-1:i-max_k1-1:-1])
            
            right_sum = 0.0
            max_k2 = min(n-1-i, n-1)
            if max_k2 > 0:
                right_sum = np.dot(g[:max_k2], h[i+1:i+max_k2+1])
            
            # Delta in soft-max objective
            delta_soft = -2.0 * h[i] * (left_sum + right_sum)
            
            # Estimate new PSL (use exact for accuracy)
            neighbor_sum = np.zeros(n-1)
            if max_k1 > 0:
                neighbor_sum[:max_k1] += h[i-1:i-max_k1-1:-1]
            if max_k2 > 0:
                neighbor_sum[:max_k2] += h[i+1:i+max_k2+1]
            
            delta = -2.0 * h[i] * neighbor_sum
            new_acf_lags = acf_lags + delta
            new_psl = float(np.max(np.abs(new_acf_lags)))
            
            if new_psl < best_new_psl:
                best_new_psl = new_psl
                best_i = i
    
    return best_i, float(best_new_psl), current_psl

def _gradient_step(h, acf):
    """
    Gordon's fast PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    Returns (best_i, best_new_psl, current_psl).
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    best_i = -1
    best_new_psl = float('inf')
    
    acf_lags = acf[1:n]
    
    # Use soft-max guided search: prioritize positions that reduce the smooth surrogate
    # But still evaluate exact PSL for the actual flip decision
    
    # Compute gradient weights from soft-max
    x = np.abs(acf_lags)
    temperature = 0.3
    x_max = np.max(x)
    if x_max > 1e-10:
        exp_x = np.exp((x - x_max) / temperature)
        softmax_weights = exp_x / np.sum(exp_x)
    else:
        softmax_weights = np.ones(n-1) / (n-1)
    
    g = softmax_weights * np.sign(acf_lags)
    
    # For efficiency, compute the weighted neighbor sum using FFT
    # weighted_neighbor[i] = sum_k g[k-1] * (h[i-k] + h[i+k])
    
    # Let's compute this using convolution
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    # corr_left[i] = sum_{k=1}^{i} g[k-1] * h[i-k]
    # corr_right[i] = sum_{k=1}^{n-1-i} g[k-1] * h[i+k]
    
    # Use np.correlate-like approach with FFT
    H_fft = np.fft.rfft(h, n=fft_size)
    G_fft = np.fft.rfft(g, n=fft_size)
    
    # corr_right: sum_{k=1}^{n-1} g[k-1] * h[i+k]
    # = sum_{j} g[j] * h[i+1+j] for j=0..n-2
    # This is cross-correlation of h and g at offset i+1
    # corr_right[i] = (g * h)[i+1] in linear convolution
    
    # corr_left: sum_{k=1}^{i} g[k-1] * h[i-k]
    # = sum_{j} g[j] * h[i-1-j] for j=0..i-1
    # = cross-correlation of h and g at offset i-1 (reversed)
    
    # Let's just compute directly but vectorized per batch
    best_i = -1
    best_new_psl = float('inf')
    
    batch_size = 80
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        
        for i in range(start, end):
            max_k1 = min(i, n-1)
            max_k2 = min(n-1-i, n-1)
            
            # Weighted neighbor sum (for guiding)
            w_neighbor = 0.0
            if max_k1 > 0:
                w_neighbor += np.dot(g[:max_k1], h[i-1:i-max_k1-1:-1])
            if max_k2 > 0:
                w_neighbor += np.dot(g[:max_k2], h[i+1:i+max_k2+1])
            
            # Exact PSL evaluation
            neighbor_sum = np.zeros(n-1)
            if max_k1 > 0:
                neighbor_sum[:max_k1] += h[i-1:i-max_k1-1:-1]
            if max_k2 > 0:
                neighbor_sum[:max_k2] += h[i+1:i+max_k2+1]
            
            delta = -2.0 * h[i] * neighbor_sum
            new_acf_lags = acf_lags + delta
            new_psl = float(np.max(np.abs(new_acf_lags)))
            
            if new_psl < best_new_psl:
                best_new_psl = new_psl
                best_i = i
    
    return best_i, float(best_new_psl), current_psl

def _update_acf_flip(h, acf, i):
    """
    Incrementally update ACF after flipping h[i].
    """
    n = len(h)
    delta = np.zeros(2*n - 1)
    
    for k in range(-(n-1), n):
        if 0 <= i+k < n:
            delta[k + n - 1] -= 2.0 * h[i] * h[i+k]
        if 0 <= i-k < n:
            delta[k + n - 1] -= 2.0 * h[i] * h[i-k]
    
    acf_new = acf + delta
    return acf_new

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    no_improve_count = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if i == -1 or new_psl >= cur_psl:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
            continue
        
        h[i] = -h[i]
        passes += 1
        no_improve_count = 0
        
        acf = _update_acf_flip(h, acf, i)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _pairwise_flip_search(h, acf, max_pairs=200, time_budget=None, start_time=None):
    """
    Try flipping pairs of positions to escape local minima.
    """
    n = len(h)
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    best_psl = _psl_value(acf, n)
    
    rng = np.random.default_rng(12345)
    attempts = 0
    while attempts < max_pairs:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, j = rng.choice(n, size=2, replace=False)
        if i == j:
            attempts += 1
            continue
        
        h[i] = -h[i]
        h[j] = -h[j]
        
        acf_new = _compute_acf(h)
        psl_new = _psl_value(acf_new, n)
        
        if psl_new < best_psl:
            best_psl = psl_new
            best_h = h.copy()
            acf = acf_new
        else:
            h[i] = -h[i]
            h[j] = -h[j]
        
        attempts += 1
    
    return best_h, best_psl

def _triple_flip_search(h, acf, max_triples=100, time_budget=None, start_time=None):
    """
    Try flipping triples of positions to escape deeper local minima.
    """
    n = len(h)
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    best_psl = _psl_value(acf, n)
    
    rng = np.random.default_rng(54321)
    attempts = 0
    while attempts < max_triples:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        idx = rng.choice(n, size=3, replace=False)
        for i in idx:
            h[i] = -h[i]
        
        acf_new = _compute_acf(h)
        psl_new = _psl_value(acf_new, n)
        
        if psl_new < best_psl:
            best_psl = psl_new
            best_h = h.copy()
            acf = acf_new
        else:
            for i in idx:
                h[i] = -h[i]
        
        attempts += 1
    
    return best_h, best_psl

def _logistic_init(r, n, seed):
    """
    Generate a sequence of +/- 1 based on the logistic map.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9)
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            x = rng.uniform(0.1, 0.9)
    return signs

def _zadoff_chu_init(n, p1, p2, seed):
    """
    Generate a Zadoff-Chu-like sequence using the sign of the imaginary part
    of the product of two quadratic phase chirps with distinct prime moduli.
    
    Zadoff-Chu sequence: z_c[k] = exp(-j*pi*c*k*(k+1)/p) for p odd
    We take sign(Im(z_c1 * conj(z_c2))) to get a signed step function.
    
    This creates multiplicative spectral cancellation properties.
    """
    rng = np.random.default_rng(seed)
    
    # Choose roots for the two chirps
    c1 = rng.integers(1, p1)
    c2 = rng.integers(1, p2)
    
    k = np.arange(n)
    
    # Quadratic phase for chirp 1 (mod p1)
    # phase1 = -pi * c1 * k * (k+1) / p1
    k_mod1 = k % p1
    phase1 = -np.pi * c1 * k_mod1 * (k_mod1 + 1) / p1
    
    # Quadratic phase for chirp 2 (mod p2)
    k_mod2 = k % p2
    phase2 = -np.pi * c2 * k_mod2 * (k_mod2 + 1) / p2
    
    # Product: z1 * conj(z2) = exp(j*(phase1 - phase2))
    phase_diff = phase1 - phase2
    
    # Sign of imaginary part
    signs = np.sign(np.sin(phase_diff))
    # Handle zeros (map to +1)
    signs[signs == 0] = 1.0
    
    return signs

def _zadoff_chu_init_v2(n, p1, p2, p3, seed):
    """
    Enhanced Zadoff-Chu: product of three chirps for richer structure.
    sign(Im(z1 * z2 * conj(z3)))
    """
    rng = np.random.default_rng(seed)
    
    c1 = rng.integers(1, p1)
    c2 = rng.integers(1, p2)
    c3 = rng.integers(1, p3)
    
    k = np.arange(n)
    
    k_mod1 = k % p1
    k_mod2 = k % p2
    k_mod3 = k % p3
    
    phase1 = -np.pi * c1 * k_mod1 * (k_mod1 + 1) / p1
    phase2 = -np.pi * c2 * k_mod2 * (k_mod2 + 1) / p2
    phase3 = -np.pi * c3 * k_mod3 * (k_mod3 + 1) / p3
    
    # z1 * z2 * conj(z3) -> phase = phase1 + phase2 - phase3
    phase_sum = phase1 + phase2 - phase3
    
    signs = np.sign(np.sin(phase_sum))
    signs[signs == 0] = 1.0
    
    return signs

def _smooth_polish(h, time_budget, start_time):
    """
    Final polish pass: use soft-max surrogate to guide a more targeted search.
    Tries flips that most reduce the smooth objective, then verifies with exact PSL.
    """
    n = len(h)
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    # Multiple temperatures for exploration
    for temp in [0.1, 0.3, 0.5, 1.0]:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        acf_lags = acf[1:n]
        x = np.abs(acf_lags)
        x_max = np.max(x)
        if x_max < 1e-10:
            break
        
        exp_x = np.exp((x - x_max) / temp)
        softmax_weights = exp_x / np.sum(exp_x)
        g = softmax_weights * np.sign(acf_lags)
        
        # Find top candidates by soft-max reduction potential
        # For each i, estimate delta_soft = -2*h[i] * (left_corr + right_corr)
        # where left_corr = sum_k g[k-1]*h[i-k], right_corr = sum_k g[k-1]*h[i+k]
        
        # Compute all candidate scores efficiently
        scores = np.zeros(n)
        for i in range(n):
            max_k1 = min(i, n-1)
            max_k2 = min(n-1-i, n-1)
            left_corr = 0.0
            right_corr = 0.0
            if max_k1 > 0:
                left_corr = np.dot(g[:max_k1], h[i-1:i-max_k1-1:-1])
            if max_k2 > 0:
                right_corr = np.dot(g[:max_k2], h[i+1:i+max_k2+1])
            scores[i] = -2.0 * h[i] * (left_corr + right_corr)
        
        # Try top 10 candidates
        top_indices = np.argsort(scores)[-10:]
        for i in top_indices:
            if time_budget is not None and (time.time() - start_time) > time_budget:
                break
            
            h[i] = -h[i]
            acf_new = _update_acf_flip(h, acf, i)
            psl_new = _psl_value(acf_new, n)
            
            if psl_new < best_psl:
                best_psl = psl_new
                best_h = h.copy()
                acf = acf_new
            else:
                h[i] = -h[i]
                # Keep acf as is (it wasn't changed)
    
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Random
    for i in range(8):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(4):
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
    for i in range(3):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
        
    # 5. Logistic map initializations
    for r in [3.5, 3.7, 3.9, 3.99, 4.0]:
        for seed in [1, 2, 3]:
            candidates.append(_logistic_init(r, N, seed))
            
    # 6. Thue-Morse sequence
    thue = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(thue)
    
    # 7. Zadoff-Chu chirp products with distinct prime moduli
    # Primes near N/2 for good coverage
    primes = [197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
    for i in range(min(8, len(primes))):
        p1 = primes[i]
        p2 = primes[(i+3) % len(primes)]
        candidates.append(_zadoff_chu_init(N, p1, p2, seed=100+i))
    
    # 8. Triple chirp products
    for i in range(4):
        p1 = primes[i]
        p2 = primes[(i+2) % len(primes)]
        p3 = primes[(i+5) % len(primes)]
        candidates.append(_zadoff_chu_init_v2(N, p1, p2, p3, seed=200+i))
    
    # 9. More diverse random
    for density in [0.3, 0.5, 0.7]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = 1.0 if rng.random() < density else -1.0
        candidates.append(h)
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates with fast PSL optimization
    top_k = min(6, len(scored_candidates))
    for i, (s, h) in enumerate(scored_candidates[:top_k]):
        if time.time() > deadline - 30:
            break
        time_limit = deadline - 20
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=300, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
        # Pairwise flips
        if time.time() < deadline - 15:
            acf_best = _compute_acf(best_h)
            h_pair, s_pair = _pairwise_flip_search(best_h, acf_best, max_pairs=80, 
                                                    time_budget=deadline - 15, start_time=start_time)
            if s_pair < best_score:
                best_score = s_pair
                best_h = h_pair.copy()
        
        # Triple flips for deeper escape
        if time.time() < deadline - 10:
            acf_best = _compute_acf(best_h)
            h_triple, s_triple = _triple_flip_search(best_h, acf_best, max_pairs=50, 
                                                      time_budget=deadline - 10, start_time=start_time)
            if s_triple < best_score:
                best_score = s_triple
                best_h = h_triple.copy()
    
    # Random restarts with fast PSL optimization
    while time.time() < deadline - 15:
        if rng.random() < 0.4:
            h_init = rng.choice([-1.0, 1.0], size=N)
        elif rng.random() < 0.7:
            r = rng.uniform(3.5, 4.0)
            h_init = _logistic_init(r, N, int(rng.integers(0, 10000)))
        else:
            # Chirp-based restart
            p1 = primes[int(rng.integers(0, len(primes)))]
            p2 = primes[int(rng.integers(0, len(primes)))]
            if p1 == p2:
                p2 = primes[(primes.index(p1)+1) % len(primes)]
            h_init = _zadoff_chu_init(N, p1, p2, seed=int(rng.integers(0, 10000)))
            
        time_limit = min(deadline - 12, time.time() + 5)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=200, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Final smooth polish
    if best_h is not None and time.time() < deadline - 3:
        h_polished, s_polished = _smooth_polish(best_h, time_budget=deadline - 2, start_time=start_time)
        if s_polished < best_score:
            best_score = s_polished
            best_h = h_polished.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
