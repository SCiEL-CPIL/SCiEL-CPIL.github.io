import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _flip_delta(s, i):
    """
    Compute the change in ACF if we flip s[i] -> -s[i].
    Returns delta array where A_new = A + delta.
    """
    delta = np.zeros(N)
    old_si = s[i]
    factor = -2.0 * old_si
    
    # For lags k=0..N-1-i: A[k] changes by factor * s[i+k]
    if i < N:
        delta[:N-i] += factor * s[i:N]
    
    # For lags k=1..i: A[k] also changes by factor * s[i-k] (from the conjugate part)
    if i > 0:
        delta[1:i+1] += factor * s[i-1::-1]
    
    return delta

def _block_flip_delta(s, A, start, length):
    """
    Compute the change in ACF if we flip a block of signs from start to start+length-1.
    """
    delta = np.zeros(N)
    block = s[start:start+length]
    factor = -2.0
    
    # For each position j in the block, its flip contributes to delta
    for j in range(length):
        i = start + j
        old_si = s[i]
        f = -2.0 * old_si
        
        if i < N:
            delta[:N-i] += f * s[i:N]
        
        if i > 0:
            delta[1:i+1] += f * s[i-1::-1]
    
    return delta

def _halton_init(base, N):
    """Generate a sign sequence based on Halton sequence in given base."""
    s = np.zeros(N)
    for n in range(N):
        f = 0.0
        p = 1.0 / base
        num = n
        while num > 0:
            f += p * (num % base)
            num //= base
            p /= base
        if f < 0.5:
            s[n] = 1.0
        else:
            s[n] = -1.0
    return s

def _continuous_seed(deadline, max_iters=20000, lr=0.01):
    """
    Generate a good initial binary seed by running continuous gradient descent
    on the L2 norm of the autocorrelation (flat spectrum optimization),
    then thresholding to binary.
    """
    # Start with random continuous values in [-1, 1]
    x = np.random.uniform(-1, 1, size=N)
    
    # Gradient descent on sum of |A[k]|^2 for k=1..N-1
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute ACF and its gradient
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        
        F = np.fft.rfft(x, n=n_fft)
        P = F * np.conj(F)
        A = np.fft.irfft(P, n=n_fft)[:N]
        
        # L2 objective: sum of A[k]^2 for k=1..N-1
        # Gradient of ACF w.r.t. x: 
        # A[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
        # d(A[k])/dx[j] = sum_{i: i+k=j} x[i] + sum_{i: i=j-k} x[i]
        #                = x[j-k] (if j>=k) + x[j+k] (if j+k<N)
        # 
        # More efficiently, use FFT:
        # The gradient of sum_k A[k]^2 can be computed via FFT
        
        # Actually, let's use a simpler approach:
        # d/dx_j [sum_k A[k]^2] = 2 * sum_k A[k] * dA[k]/dx_j
        # 
        # dA[k]/dx_j = x[j-k] (if 0<=j-k<N and k>0) + x[j+k] (if 0<=j+k<N and k>0)
        #            + x[j-k] (from conjugate part) + x[j+k] (from conjugate part)
        # 
        # Let's compute this efficiently:
        # For each k from 1 to N-1:
        #   dA[k]/dx = x[k:] (shifted) + x[:N-k] (shifted)
        
        grad = np.zeros(N)
        for k in range(1, N):
            # dA[k]/dx[j] = x[j-k] for j>=k, and x[j+k] for j+k<N
            # Using FFT-based approach would be faster, but let's do it directly for now
            # This is O(N^2) which is too slow. Let's use FFT.
            pass
        
        # Better approach: use the fact that A = ifft(|fft(x)|^2)
        # The gradient can be computed as:
        # grad = ifft(2 * fft(A) * conj(fft(x))) * 2  (approximately)
        
        # Actually, let's use a simpler continuous optimization:
        # Minimize ||A[1:]||_2^2 using projected gradient descent
        
        # Compute gradient via FFT
        # A[k] = sum_i x[i] x[i+k]
        # d/dx_j A[k] = x[j-k] (if j>=k) + x[j+k] (if j+k<N)
        
        # Vectorized: for each k, the contribution to grad is:
        #   grad[k:] += A[k] * x[:N-k]   (from x[j+k] term)
        #   grad[:N-k] += A[k] * x[k:]   (from x[j-k] term)
        
        # This is still O(N^2). Let's use a different approach.
        
        # Use the property that the gradient of the autocorrelation energy
        # can be computed via FFT:
        # E = sum_{k=1}^{N-1} A[k]^2
        # dE/dx = 4 * ifft(fft(A) * conj(fft(x)))  (approximately, with some corrections)
        
        # Let's just do simple random search in continuous space with local improvements
        # This is getting complicated. Let's use a simpler initialization strategy.
        
        break  # Skip this for now, use alternative
    
    # Alternative: use a structured continuous initialization
    # Mix of low-frequency and high-frequency components
    t = np.arange(N)
    freqs = [1, 2, 3, 5, 8, 13, 21]
    x = np.zeros(N)
    for f in freqs:
        x += np.sin(2 * np.pi * f * t / N + np.random.uniform(0, 2*np.pi))
    
    # Threshold with some noise
    threshold = np.median(x)
    s = np.where(x > threshold, 1.0, -1.0)
    
    return s

def _fbm_init(H=0.5, seed=None):
    """
    Generate initial sign sequence from discretized fractional Brownian motion.
    fBm with H=0.5 is standard Brownian motion, which has self-similar structure.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Generate fBm using the Cholesky decomposition of the covariance matrix
    # For H=0.5, this is just standard Brownian motion
    # Covariance: E[B(t)B(s)] = 0.5*(|t|^{2H} + |s|^{2H} - |t-s|^{2H})
    
    # For simplicity, use the fact that H=0.5 fBm is Brownian motion
    # B(t) = sum of independent increments
    
    # Generate Brownian motion path
    increments = rng.normal(0, 1, size=N)
    bm = np.cumsum(increments)
    
    # Normalize
    bm -= np.mean(bm)
    bm /= (np.std(bm) + 1e-10)
    
    # Threshold to get signs
    s = np.where(bm > 0, 1.0, -1.0)
    
    return s

def _structured_init(type_idx):
    """
    Generate structured initializations that have good spectral properties.
    """
    if type_idx == 0:
        # fBm-based
        return _fbm_init(H=0.5)
    elif type_idx == 1:
        # Multi-scale sinusoidal
        t = np.arange(N)
        x = np.zeros(N)
        for f in [1, 2, 3, 5, 8, 13, 21, 34]:
            x += np.sin(2 * np.pi * f * t / N + np.random.uniform(0, 2*np.pi)) / f
        s = np.where(x > 0, 1.0, -1.0)
        return s
    elif type_idx == 2:
        # Random with low-pass filtering
        s = np.random.randint(0, 2, size=N) * 2 - 1
        # Apply a simple moving average filter
        kernel = np.ones(5) / 5
        filtered = np.convolve(s, kernel, mode='same')
        s = np.where(filtered > 0, 1.0, -1.0)
        return s
    elif type_idx == 3:
        # Halton-based
        base = np.random.choice([3, 5, 7])
        return _halton_init(base, N)
    elif type_idx == 4:
        # Sine-based with random phase
        t = np.linspace(0, 4*np.pi, N, endpoint=False)
        phase = np.random.uniform(0, 2*np.pi)
        s = np.sign(np.sin(t + phase))
        s[s == 0] = 1
        return s
    else:
        # Pure random
        return np.random.randint(0, 2, size=N) * 2 - 1

def _local_search(s, A, current_val, deadline, max_passes=50, max_patience=2000):
    """
    Efficient local search with single-flip moves.
    """
    improved = True
    passes = 0
    patience = 0
    
    while improved and passes < max_passes and patience < max_patience and time.time() < deadline:
        improved = False
        passes += 1
        patience = 0
        
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                return s, A, current_val
            
            delta = _flip_delta(s, i)
            A_new = A + delta
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val:
                s[i] = -s[i]
                A = A_new
                current_val = val_new
                improved = True
                patience = 0
            else:
                patience += 1
    
    return s, A, current_val

def _anneal_block_flips(s, A, current_val, deadline, max_time=5.0):
    """
    Simulated annealing with block-flip moves to escape local minima.
    """
    block_deadline = min(deadline, time.time() + max_time)
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    
    while T > T_min and time.time() < block_deadline:
        # Random block flip
        length = np.random.choice([2, 3, 4, 5, 8, 10, 15, 20])
        if length >= N:
            continue
        start = np.random.randint(0, N - length + 1)
        
        # Compute delta for block flip
        delta = np.zeros(N)
        for j in range(length):
            i = start + j
            f = -2.0 * s[i]
            if i < N:
                delta[:N-i] += f * s[i:N]
            if i > 0:
                delta[1:i+1] += f * s[i-1::-1]
        
        A_new = A + delta
        val_new = _max_abs_acf_from_array(A_new)
        
        if val_new < current_val or (T > T_min and np.random.random() < np.exp(-(val_new - current_val) / T)):
            s[start:start+length] = -s[start:start+length]
            A = A_new
            current_val = val_new
        
        T *= cooling
    
    return s, A, current_val

def _final_polish(s, A, current_val, deadline, max_time=3.0):
    """
    Final high-precision polish with multiple passes.
    """
    polish_deadline = min(deadline, time.time() + max_time)
    
    for _ in range(100):
        if time.time() > polish_deadline:
            break
        improved = False
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > polish_deadline:
                return s, A, current_val
            delta = _flip_delta(s, i)
            A_new = A + delta
            val_new = _max_abs_acf_from_array(A_new)
            if val_new < current_val:
                s[i] = -s[i]
                A = A_new
                current_val = val_new
                improved = True
    
    return s, A, current_val

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Try multiple different initialization strategies
    n_init_strategies = 6
    
    for init_idx in range(n_init_strategies):
        if time.time() > deadline:
            break
        
        # Generate initial sequence
        if init_idx < 5:
            s = _structured_init(init_idx)
        else:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search
        s, A, current_val = _local_search(s, A, current_val, deadline, max_passes=30, max_patience=1500)
        
        # Block flip annealing
        if time.time() < deadline:
            s, A, current_val = _anneal_block_flips(s, A, current_val, deadline, max_time=3.0)
        
        # Final polish
        if time.time() < deadline:
            s, A, current_val = _final_polish(s, A, current_val, deadline, max_time=2.0)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # If we have time left, do more restarts with random initializations
    while time.time() < deadline:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        s, A, current_val = _local_search(s, A, current_val, deadline, max_passes=20, max_patience=1000)
        
        if time.time() < deadline:
            s, A, current_val = _anneal_block_flips(s, A, current_val, deadline, max_time=2.0)
        
        if time.time() < deadline:
            s, A, current_val = _final_polish(s, A, current_val, deadline, max_time=1.5)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    return best_seq
