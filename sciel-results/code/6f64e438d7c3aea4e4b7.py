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

def _thue_morse_sequence(length):
    """Generate a Thue-Morse sequence of given length."""
    h = np.zeros(length)
    for i in range(length):
        # Count number of 1s in binary representation
        count = bin(i).count('1')
        h[i] = 1.0 if count % 2 == 0 else -1.0
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

def _fast_swap_psl(h, i, j, acf):
    """
    Calculate new PSL if h[i] and h[j] are swapped, given current h and acf.
    O(N) complexity.
    """
    n = len(h)
    hi, hj = h[i], h[j]
    
    # If values are the same, swap does nothing
    if hi == hj:
        return _psl_value(acf, n)
    
    # Delta for swapping: 
    # acf[k] = sum_{l} h[l] * h[l+k]
    # When we swap h[i] and h[j], the change is:
    # For lag k:
    #   terms involving i: h[i]*h[i+k] + h[i-k]*h[i] become h[j]*h[i+k] + h[i-k]*h[j]
    #   terms involving j: h[j]*h[j+k] + h[j-k]*h[j] become h[i]*h[j+k] + h[j-k]*h[i]
    
    lags = np.arange(1, n)
    
    # Compute delta for each lag
    delta = np.zeros(n-1)
    
    # For lag k, the old contribution from position i:
    #   h[i] * h[i+k] (if i+k < n) + h[i-k] * h[i] (if i-k >= 0)
    # New contribution:
    #   h[j] * h[i+k] (if i+k < n) + h[i-k] * h[j] (if i-k >= 0)
    # So delta_i[k] = (h[j] - h[i]) * (h[i+k] + h[i-k]) with boundary checks
    
    # For lag k, the old contribution from position j:
    #   h[j] * h[j+k] (if j+k < n) + h[j-k] * h[j] (if j-k >= 0)
    # New contribution:
    #   h[i] * h[j+k] (if j+k < n) + h[j-k] * h[i] (if j-k >= 0)
    # So delta_j[k] = (h[i] - h[j]) * (h[j+k] + h[j-k]) with boundary checks
    
    # Vectorized computation
    # For i:
    left_i = i - lags
    right_i = i + lags
    contrib_i = np.zeros(n-1)
    valid_left_i = left_i >= 0
    valid_right_i = right_i < n
    if np.any(valid_left_i):
        contrib_i[valid_left_i] += h[left_i[valid_left_i]]
    if np.any(valid_right_i):
        contrib_i[valid_right_i] += h[right_i[valid_right_i]]
    delta += (hj - hi) * contrib_i
    
    # For j:
    left_j = j - lags
    right_j = j + lags
    contrib_j = np.zeros(n-1)
    valid_left_j = left_j >= 0
    valid_right_j = right_j < n
    if np.any(valid_left_j):
        contrib_j[valid_left_j] += h[left_j[valid_left_j]]
    if np.any(valid_right_j):
        contrib_j[valid_right_j] += h[right_j[valid_right_j]]
    delta += (hi - hj) * contrib_j
    
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
        
    # h[i] has been flipped, so old h[i] = -new h[i]
    # delta = -2 * old_h_i * neighbors = 2 * new_h_i * neighbors
    delta = 2.0 * h[i] * neighbors
    acf[1:n] += delta
    return acf

def _update_acf_swap(h, i, j, acf):
    """
    Update the ACF array after swapping h[i] and h[j].
    Assumes h[i] and h[j] have already been swapped.
    """
    n = len(h)
    hi, hj = h[i], h[j]  # These are the NEW values after swap
    
    lags = np.arange(1, n)
    
    # For position i (now has value hj, was hi):
    left_i = i - lags
    right_i = i + lags
    contrib_i = np.zeros(n-1)
    valid_left_i = left_i >= 0
    valid_right_i = right_i < n
    if np.any(valid_left_i):
        contrib_i[valid_left_i] += h[left_i[valid_left_i]]
    if np.any(valid_right_i):
        contrib_i[valid_right_i] += h[right_i[valid_right_i]]
    # Old value was hi, new is hj. Delta = (hj - hi) * contrib_i
    acf[1:n] += (hj - hi) * contrib_i
    
    # For position j (now has value hi, was hj):
    left_j = j - lags
    right_j = j + lags
    contrib_j = np.zeros(n-1)
    valid_left_j = left_j >= 0
    valid_right_j = right_j < n
    if np.any(valid_left_j):
        contrib_j[valid_left_j] += h[left_j[valid_left_j]]
    if np.any(valid_right_j):
        contrib_j[valid_right_j] += h[right_j[valid_right_j]]
    # Old value was hj, new is hi. Delta = (hi - hj) * contrib_j
    acf[1:n] += (hi - hj) * contrib_j
    
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
    cooling = 0.9995
    
    steps = 0
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # With some probability, try a swap instead of a flip
        if rng.random() < 0.3:
            # Try a swap
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            if i == j:
                continue
            if h[i] == h[j]:
                continue
            
            new_psl = _fast_swap_psl(h, i, j, acf)
            delta_psl = new_psl - current_psl
            
            if delta_psl < 0 or (T > 0 and rng.random() < np.exp(-delta_psl / T)):
                # Accept swap
                h[i], h[j] = h[j], h[i]
                acf = _update_acf_swap(h, i, j, acf)
                current_psl = _psl_value(acf, n)
                
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_h = h.copy()
        else:
            # Try a flip
            i = int(rng.integers(n))
            
            new_psl = _fast_flip_psl(h, i, acf)
            delta_psl = new_psl - current_psl
            
            if delta_psl < 0 or (T > 0 and rng.random() < np.exp(-delta_psl / T)):
                # Accept
                h[i] = -h[i]
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
        acf_lags = acf[1:n]
        max_idx = np.argmax(np.abs(acf_lags))
        max_lag = max_idx + 1
        max_val = acf_lags[max_idx]
        
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
            
        direction = -np.sign(max_val) * grad
        h_cont += step_size * direction
        h_cont = np.clip(h_cont, -1.0, 1.0)
        
        acf = _compute_acf(h_cont)
        
    h_bin = np.sign(h_cont)
    h_bin[h_bin == 0] = 1.0
    return h_bin

def _greedy_polish(h, time_budget=10.0, start_time=None):
    """
    Greedy local search: find the best single flip OR swap and apply.
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
        best_move = None
        best_new_psl = current_psl
        best_type = None
        best_i, best_j = -1, -1
        
        # Check all flips
        for i in range(n):
            new_psl = _fast_flip_psl(h, i, acf)
            if new_psl < best_new_psl:
                best_new_psl = new_psl
                best_move = 'flip'
                best_i = i
                best_j = -1
        
        # Check a subset of swaps (sample to keep it fast)
        num_swap_samples = 50
        for _ in range(num_swap_samples):
            i = int(np.random.randint(n))
            j = int(np.random.randint(n))
            if i == j or h[i] == h[j]:
                continue
            new_psl = _fast_swap_psl(h, i, j, acf)
            if new_psl < best_new_psl:
                best_new_psl = new_psl
                best_move = 'swap'
                best_i = i
                best_j = j
        
        if best_move is None:
            break
            
        if best_move == 'flip':
            h[best_i] = -h[best_i]
            acf = _update_acf_flip(h, best_i, acf)
        else:
            h[best_i], h[best_j] = h[best_j], h[best_i]
            acf = _update_acf_swap(h, best_i, best_j, acf)
        
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
    
    # 2. Thue-Morse Sequence Start
    h_tm = _thue_morse_sequence(N)
    
    # 3. Random Starts
    candidates = [h_leg, h_tm]
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
            
    # Gradient Polish
    if best_h is not None and time.time() < deadline - 10:
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
