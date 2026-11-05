import numpy as np
import time

N = 400

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

def _legendre_sequence(length):
    """Generate a Legendre sequence of given length."""
    # Find next prime >= length
    p = length + 1
    while True:
        is_prime = True
        for i in range(2, int(np.sqrt(p)) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            break
        p += 1
    
    # Legendre symbol
    h = np.zeros(p)
    for i in range(1, p):
        # Euler's criterion: (a/p) = a^((p-1)/2) mod p
        # 1 if 1, -1 if p-1, 0 if 0
        val = pow(i, (p-1)//2, p)
        if val == 1:
            h[i] = 1.0
        elif val == p-1:
            h[i] = -1.0
        else:
            h[i] = 0.0 # Should not happen for i < p
    
    # Take first 'length' elements, replace 0 with 1 (or random)
    h = h[:length]
    h[h == 0] = 1.0
    return h

def _fast_flip_psl(h, i, acf):
    """
    Calculate new PSL if h[i] is flipped, given current h and acf.
    O(N) complexity.
    """
    n = len(h)
    # delta_acf[k] = -2 * h[i] * (h[i-k] + h[i+k])
    # k ranges from 1 to n-1
    
    # Create index arrays for lags
    # left indices: i-1, i-2, ..., i-(n-1)
    # right indices: i+1, i+2, ..., i+(n-1)
    
    # We only need valid indices
    # left: max(0, i-(n-1)) to i-1
    # right: i+1 to min(n-1, i+(n-1))
    
    # Efficient vectorized computation
    lags = np.arange(1, n)
    left_idx = i - lags
    right_idx = i + lags
    
    neighbors = np.zeros(n-1)
    
    valid_left = left_idx >= 0
    valid_right = right_idx < n
    
    if np.any(valid_left):
        neighbors[valid_left] += h[left_idx[valid_left]]
    if np.any(valid_right):
        neighbors[valid_right] += h[right_idx[valid_right]]
        
    delta = -2.0 * h[i] * neighbors
    new_acf_lags = acf[1:n] + delta
    return np.max(np.abs(new_acf_lags))

def _update_acf_flip(h, i, acf):
    """
    Update the ACF array in-place (or return new) after flipping h[i].
    Returns the updated acf array.
    """
    n = len(h)
    lags = np.arange(1, n)
    left_idx = i - lags
    right_idx = i + lags
    
    neighbors = np.zeros(n-1)
    valid_left = left_idx >= 0
    valid_right = right_idx < n
    
    if np.any(valid_left):
        neighbors[valid_left] += h[left_idx[valid_left]]
    if np.any(valid_right):
        neighbors[valid_right] += h[right_idx[valid_right]]
        
    # Note: h[i] has already been flipped in h? 
    # The formula for delta uses the OLD h[i].
    # If we call this AFTER flipping h[i], we need to be careful.
    # Let's assume h[i] is the NEW value.
    # Old h[i] = -new h[i].
    # delta = -2 * old_h_i * neighbors = -2 * (-new_h_i) * neighbors = 2 * new_h_i * neighbors.
    # Wait, the neighbors sum uses the OLD h values for j != i.
    # Since only h[i] changed, h[j] for j!=i is the same.
    # So neighbors computed from current h (where h[i] is new) is valid for the sum of neighbors.
    
    delta = 2.0 * h[i] * neighbors
    acf[1:n] += delta
    return acf

def _simulated_annealing_fast(h_init, time_budget=30.0, start_time=None, rng=None):
    """
    Fast Simulated Annealing using O(N) flip evaluation and ACF update.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h = h_init.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995 # Slow cooling for many steps
    
    steps = 0
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = int(rng.integers(n))
        
        # Calculate PSL if we flip i
        # We use the pre-flip h to calculate delta
        new_psl = _fast_flip_psl(h, i, acf)
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or (T > 0 and np.random.random() < np.exp(-delta_psl / T)):
            # Accept
            h[i] = -h[i]
            # Update ACF incrementally
            acf = _update_acf_flip(h, i, acf)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
                
        T *= cooling
        steps += 1
        
    return best_h, best_psl

def _soft_max_score(acf, n, beta=10.0):
    """
    Compute a smooth approximation of the max absolute autocorrelation.
    Uses log-sum-exp on the squared values.
    """
    acf_lags = acf[1:n]
    vals = np.abs(acf_lags)
    max_val = np.max(vals)
    if max_val == 0:
        return 0.0
    # log-sum-exp trick for stability
    # We want approx max(vals).
    # log(sum(exp(beta * vals))) / beta approximates max(vals) as beta -> inf.
    shifted = vals - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted)))
    return (log_sum + beta * max_val) / beta

def _gradient_step(h, acf, step_size=0.1, beta=10.0, n_steps=50):
    """
    Perform gradient descent on the soft-max score using continuous relaxation.
    h is treated as continuous variables in [-1, 1].
    """
    n = len(h)
    h_cont = h.astype(np.float64)
    
    for _ in range(n_steps):
        # Compute gradient of soft-max score w.r.t. h
        # score = (1/beta) * log(sum_k exp(beta * |acf[k]|))
        # acf[k] = sum_j h[j] h[j+k]
        # d|acf[k]|/dh[i] is complex, but we can use the sign.
        # Let's use the gradient of the squared soft-max? Or just approximate.
        # Actually, let's just do a simple heuristic:
        # Move h[i] in the direction that reduces the largest sidelobes.
        
        # Find the lag with the largest absolute value
        acf_lags = acf[1:n]
        max_idx = np.argmax(np.abs(acf_lags))
        max_lag = max_idx + 1
        max_val = acf_lags[max_idx]
        
        # Gradient of acf[max_lag] w.r.t h[i]
        # acf[l] = sum_j h[j] h[j+l]
        # d acf[l] / d h[i] = h[i-l] + h[i+l] (with boundary handling)
        
        grad = np.zeros(n)
        for i in range(n):
            g = 0.0
            j = i - max_lag
            if 0 <= j < n:
                g += h[j]
            j = i + max_lag
            if 0 <= j < n:
                g += h[j]
            grad[i] = g
            
        # We want to minimize |max_val|.
        # If max_val > 0, we want to decrease acf[max_lag].
        # If max_val < 0, we want to increase acf[max_lag].
        # So we move h in the direction -sign(max_val) * grad.
        
        direction = -np.sign(max_val) * grad
        h_cont += step_size * direction
        
        # Clip to [-1, 1]
        h_cont = np.clip(h_cont, -1.0, 1.0)
        
        # Recompute ACF for gradient update (expensive but n_steps is small)
        acf = _compute_acf(h_cont)
        
    # Round to binary
    h_bin = np.sign(h_cont)
    h_bin[h_bin == 0] = 1.0
    return h_bin

def _greedy_polish(h, time_budget=10.0, start_time=None):
    """
    Greedy local search: find the best single flip and apply.
    """
    if start_time is None:
        start_time = time.time()
        
    h = h.copy().astype(np.float64)
    n = len(h)
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    best_h = h.copy()
    best_psl = current_psl
    
    while time.time() - start_time < time_budget:
        best_i = -1
        best_new_psl = current_psl
        
        # Sample a subset of positions to speed up? 
        # For N=400, checking all is 400 * O(N) = 160,000 ops.
        # This is fast enough to do every iteration.
        
        for i in range(n):
            new_psl = _fast_flip_psl(h, i, acf)
            if new_psl < best_new_psl:
                best_new_psl = new_psl
                best_i = i
                
        if best_i == -1 or best_new_psl >= current_psl:
            break
            
        h[best_i] = -h[best_i]
        acf = _update_acf_flip(h, best_i, acf)
        current_psl = best_new_psl
        
        if current_psl < best_psl:
            best_psl = current_psl
            best_h = h.copy()
            
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # 1. Legendre Sequence Start
    h_leg = _legendre_sequence(N)
    
    # 2. Random Starts
    candidates = [h_leg]
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # Evaluate and pick best initial
    scored = []
    for h in candidates:
        s = _score(h)
        scored.append((s, h))
    scored.sort(key=lambda x: x[0])
    
    # Run SA on the best initial candidates
    for s, h in scored[:3]:
        if time.time() > deadline - 20:
            break
        h_refined, s_refined = _simulated_annealing_fast(h, time_budget=min(10.0, deadline - 20 - (time.time() - start_time)), start_time=start_time, rng=rng)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # Gradient Polish (New)
    if best_h is not None and time.time() < deadline - 10:
        # Use gradient step on the best_h
        acf = _compute_acf(best_h)
        h_grad = _gradient_step(best_h, acf, step_size=0.5, n_steps=20)
        s_grad = _score(h_grad)
        if s_grad < best_score:
            best_score = s_grad
            best_h = h_grad.copy()
            
    # Polish the best result
    if best_h is not None and time.time() < deadline - 5:
        h_polished, s_polished = _greedy_polish(best_h, time_budget=min(10.0, deadline - 5 - (time.time() - start_time)), start_time=start_time)
        if s_polished < best_score:
            best_score = s_polished
            best_h = h_polished.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
