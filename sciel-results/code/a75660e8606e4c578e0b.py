import numpy as np
import time

N = 400
SOFTMAX_BETA = 20.0  # Temperature for soft-max surrogate

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
    
    # Efficiently compute the sum of neighbors for each position
    # S[i, k-1] = h[i-k] + h[i+k] (where defined)
    # We want delta[k-1] for a flip at i: -2 * h[i] * (h[i-k] + h[i+k])
    # Instead of building a full n x (n-1) matrix, we can just iterate over i
    # and compute the resulting max.
    
    best_i = -1
    best_new_psl = float('inf')
    
    # Pre-compute the current ACF lags
    acf_lags = acf[1:n]
    
    # To speed this up, we can vectorize the computation of the new ACF for a given i.
    # new_acf = acf + delta_i
    # delta_i[k-1] = -2 * h[i] * (h[i-k] + h[i+k])
    
    # Let's try a slightly more optimized loop or vectorization.
    # For each i, the change in ACF is:
    # d_acf = -2 * h[i] * (convolve-like term)
    
    # Actually, let's just loop over i, but make it tight.
    # N=400, so 400 iterations. Inside, we do vector ops of length 399.
    # This should be fast enough.
    
    for i in range(n):
        # Construct the change in ACF
        # h[i-k] is valid for k <= i
        # h[i+k] is valid for k < n-i
        # So the neighbor sum vector:
        neighbor_sum = np.zeros(n-1)
        
        # h[i-k] for k=1..min(i, n-1)
        # indices: i-1, i-2, ..., max(0, i-(n-1))
        # corresponding k: 1, 2, ..., min(i, n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1] # h[i-1], h[i-2]...
            
        # h[i+k] for k=1..min(n-1-i, n-1)
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1] # h[i+1], h[i+2]...
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        new_psl = np.max(np.abs(new_acf_lags))
        
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i
            
    return best_i, float(best_new_psl), current_psl

def _soft_max_psl(acf_lags, beta):
    """Compute soft-max of |acf_lags| using log-sum-exp for numerical stability."""
    x = np.abs(acf_lags)
    if len(x) == 0:
        return 0.0
    x_max = np.max(x)
    if x_max < 1e-12:
        return x_max
    # log-sum-exp trick
    log_sum = np.log(np.sum(np.exp(beta * (x - x_max)))) / beta
    return x_max + log_sum

def _soft_gradient_descent(h, max_iters=500, lr=0.05, beta=SOFTMAX_BETA, time_budget=None, start_time=None):
    """
    Gradient descent on continuous variables using soft-max surrogate.
    This smooths the non-differentiable PSL objective.
    """
    n = len(h)
    h_cont = h.copy().astype(np.float64)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h_cont.copy()
    best_psl = _score(h_cont)
    
    for it in range(max_iters):
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        # Compute ACF
        acf = _compute_acf(h_cont)
        acf_lags = acf[1:n]
        
        # Soft-max surrogate
        soft_psl = _soft_max_psl(acf_lags, beta)
        
        # Compute gradient of soft-max w.r.t. acf_lags
        # d(soft_max)/d(x_j) = exp(beta * (x_j - x_max)) / sum(exp(beta * (x - x_max)))
        x = np.abs(acf_lags)
        x_max = np.max(x)
        if x_max < 1e-12:
            grad_acf = np.zeros_like(x)
        else:
            exp_terms = np.exp(beta * (x - x_max))
            sum_exp = np.sum(exp_terms)
            grad_acf = exp_terms / sum_exp
        
        # Gradient of |acf_lags| w.r.t. acf_lags
        # d|z|/dz = sign(z) for z != 0
        sign_acf = np.sign(acf_lags)
        sign_acf[acf_lags == 0] = 1.0  # arbitrary at zero
        
        # Chain rule: gradient w.r.t. acf_lags
        grad_acf_lags = grad_acf * sign_acf
        
        # Now compute gradient w.r.t. h[i] for each i
        # acf[k-1] = sum_j h[j] * h[j+k-1] (approximately, via FFT)
        # For continuous h, the gradient is more complex.
        # Use automatic differentiation or finite differences.
        # For simplicity, use finite differences with a small step.
        
        eps = 1e-6
        grad_h = np.zeros(n)
        
        for i in range(n):
            h_plus = h_cont.copy()
            h_plus[i] += eps
            acf_plus = _compute_acf(h_plus)
            soft_plus = _soft_max_psl(acf_plus[1:n], beta)
            
            h_minus = h_cont.copy()
            h_minus[i] -= eps
            acf_minus = _compute_acf(h_minus)
            soft_minus = _soft_max_psl(acf_minus[1:n], beta)
            
            grad_h[i] = (soft_plus - soft_minus) / (2 * eps)
        
        # Gradient step
        h_cont = h_cont - lr * grad_h
        
        # Project back to [-1, 1] to keep variables bounded
        h_cont = np.clip(h_cont, -1.0, 1.0)
        
        # Evaluate actual PSL (not soft) for tracking
        actual_psl = _score(h_cont)
        if actual_psl < best_psl:
            best_psl = actual_psl
            best_h = h_cont.copy()
        
        # Adaptive learning rate: reduce if loss is increasing
        if it > 0 and it % 50 == 0:
            lr = max(0.01, lr * 0.9)
    
    return best_h, best_psl

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if i == -1 or new_psl >= cur_psl:
            break
        
        h[i] = -h[i]
        passes += 1
        
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

def _snap_to_binary(h_cont):
    """Snap continuous values to ±1 based on sign."""
    h_bin = np.sign(h_cont)
    h_bin[h_bin == 0] = 1.0  # Handle zeros
    return h_bin

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n mapped to +/- 1."""
    # Thue-Morse: t(n) is parity of number of 1s in binary representation of n
    # t(0)=0, t(1)=1, t(2)=1, t(3)=0, ...
    # Map 0 -> 1, 1 -> -1 (or vice versa, sign doesn't matter for PSL)
    idx = np.arange(n)
    # Efficient way to compute popcount parity for numpy arrays
    # Using bit operations
    # We can use a simple loop for small N or vectorized bit tricks
    # Since N=400, a simple loop is fine, but let's vectorize if possible.
    # np.bitwise_xor.reduce on bits?
    # Easier: generate recursively or use lookup table for bytes.
    
    # Lookup table for parity of 0..255
    parity_table = np.array([bin(i).count('1') % 2 for i in range(256)], dtype=np.int8)
    
    # Split idx into bytes
    # idx < 2^8 for N=400, so just one byte is enough
    # If N was larger, we'd need to sum parities of bytes
    if n <= 256:
        parities = parity_table[idx]
    else:
        # General case for robustness, though N=400
        parities = np.zeros(n, dtype=np.int8)
        shift = 0
        while (1 << shift) < n:
            byte_val = (idx >> shift) & 0xFF
            parities ^= parity_table[byte_val]
            shift += 8
            
    # Map 0 -> 1.0, 1 -> -1.0
    # t=0 -> 1, t=1 -> -1
    return 1.0 - 2.0 * parities.astype(np.float64)

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
        
    # 5. Thue-Morse based initializations
    # Thue-Morse has low PSL. Shifts and sign flips might help avoid local minima.
    tm = _thue_morse(N)
    candidates.append(tm)
    candidates.append(-tm)
    # Shifted Thue-Morse (cyclic shifts)
    for shift in [1, 2, 3, 5, 10]:
        candidates.append(np.roll(tm, shift))
        candidates.append(-np.roll(tm, shift))
        
    # 6. Van der Corput low-discrepancy sequence
    def _van_der_corput_signs(n, base=2):
        """Generate van der Corput sequence mapped to ±1."""
        vdc = []
        for i in range(n):
            # Van der Corput: reverse digits of i in base
            frac = 0.0
            f = 1.0 / base
            x = i
            while x > 0:
                frac += f * (x % base)
                x //= base
                f /= base
            vdc.append(frac)
        # Map to ±1: values below 0.5 -> 1, above -> -1
        return np.sign(0.5 - np.array(vdc))
    
    vdc = _van_der_corput_signs(N)
    candidates.append(vdc)
    candidates.append(-vdc)
    for shift in [1, 2, 3]:
        candidates.append(np.roll(vdc, shift))
        candidates.append(-np.roll(vdc, shift))
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Phase 1: Smooth gradient descent on continuous variables for top candidates
    # This helps escape local minima by using a smooth surrogate
    top_k_smooth = 5
    for i, (s, h) in enumerate(scored_candidates[:top_k_smooth]):
        if time.time() > deadline - 30:
            break
        
        # Allocate time for smooth descent
        time_limit = min(15.0, deadline - 30 - (time.time() - start_time))
        if time_limit < 2.0:
            break
        
        # Run smooth gradient descent
        h_cont, s_cont = _soft_gradient_descent(h, max_iters=200, lr=0.05, beta=SOFTMAX_BETA, time_budget=time_limit, start_time=start_time)
        
        # Snap to binary and evaluate
        h_bin = _snap_to_binary(h_cont)
        s_bin = _score(h_bin)
        
        if s_bin < best_score:
            best_score = s_bin
            best_h = h_bin.copy()
    
    # Phase 2: Fast PSL optimization (Gordon's method) on top candidates
    top_k = 8 
    for i, (s, h) in enumerate(scored_candidates[:top_k]):
        if time.time() > deadline - 15:
            break
        # Allocate time based on rank, but ensure at least some time for each
        time_limit = deadline - 15
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=1000, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # If time remains, do a few random restarts with fast PSL optimization
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Quick optimization with limited time
        time_limit = min(deadline - 5, time.time() + 5)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=100, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # If we still have time, try SA on the best found so far
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(10.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_psl(best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
                
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
