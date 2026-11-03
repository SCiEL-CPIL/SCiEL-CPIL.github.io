import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT."""
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

def _smooth_surrogate(A, beta=50.0):
    """Log-sum-exp smooth surrogate of max|A[1:]|."""
    x = np.abs(A[1:])
    m = float(np.max(x))
    if m < 1e-15:
        return m
    z = beta * (x - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta

def _update_acf_flip_vectorized(A, s, i, old_si):
    """
    Update ACF after flipping s[i] in-place.
    old_si is the value before the flip.
    Returns new A array (copy) using vectorized operations.
    """
    A_new = A.copy()
    # Delta for each k: -2 * old_si * (s[i+k] + s[i-k]) where valid
    # k ranges 1..N-1
    ks = np.arange(1, N)
    
    # s[i+k] for k where i+k < N
    valid_plus = ks < (N - i)
    if np.any(valid_plus):
        idx_plus = i + ks[valid_plus]
        A_new[ks[valid_plus]] += -2.0 * old_si * s[idx_plus]
    
    # s[i-k] for k where i-k >= 0
    valid_minus = ks <= i
    if np.any(valid_minus):
        idx_minus = i - ks[valid_minus]
        A_new[ks[valid_minus]] += -2.0 * old_si * s[idx_minus]
    
    return A_new

def _local_search(s, A, current_val, deadline, max_passes=50):
    """Perform local search with incremental ACF updates."""
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline:
                break
            old_si = s[i]
            s[i] = -old_si
            A_new = _update_acf_flip_vectorized(A, s, i, old_si)
            val_new = _max_abs_acf_from_array(A_new)
            if val_new < current_val:
                current_val = val_new
                A = A_new
                improved = True
            else:
                s[i] = old_si
    return s, A, current_val

def _block_local_search(s, A, current_val, deadline, max_passes=20):
    """Perform block-flip local search with vectorized updates."""
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        for block_size in [1, 2, 3, 4, 8, 16]:
            if time.time() > deadline:
                break
            num_attempts = min(N // block_size, 30)
            for _ in range(num_attempts):
                if time.time() > deadline:
                    break
                start = np.random.randint(0, N - block_size + 1)
                indices = np.arange(start, start + block_size)
                old_vals = s[indices].copy()
                s[indices] = -old_vals
                
                # Compute new A by accumulating changes
                A_new = A.copy()
                for idx, old_v in zip(indices, old_vals):
                    ks = np.arange(1, N)
                    valid_plus = ks < (N - idx)
                    if np.any(valid_plus):
                        idx_plus = idx + ks[valid_plus]
                        A_new[ks[valid_plus]] += -2.0 * old_v * s[idx_plus]
                    valid_minus = ks <= idx
                    if np.any(valid_minus):
                        idx_minus = idx - ks[valid_minus]
                        A_new[ks[valid_minus]] += -2.0 * old_v * s[idx_minus]
                
                val_new = _max_abs_acf_from_array(A_new)
                if val_new < current_val:
                    current_val = val_new
                    A = A_new
                    improved = True
                else:
                    s[indices] = old_vals
    return s, A, current_val

def _simulated_annealing_smooth(s, A, current_val, deadline, T0=0.05, cooling=0.999, beta=50.0):
    """Simulated annealing using smooth surrogate for guidance."""
    T = T0
    best_val = current_val
    best_s = s.copy()
    best_A = A.copy()
    cur_smooth = _smooth_surrogate(A, beta)
    
    while time.time() < deadline and T > 1e-6:
        # Choose move type
        if np.random.random() < 0.7:
            # Single flip
            i = np.random.randint(0, N)
            old_si = s[i]
            s[i] = -old_si
            A_new = _update_acf_flip_vectorized(A, s, i, old_si)
            val_new = _max_abs_acf_from_array(A_new)
            smooth_new = _smooth_surrogate(A_new, beta)
            delta_smooth = smooth_new - cur_smooth
            if val_new < current_val or (delta_smooth <= 0 or np.random.random() < math.exp(-delta_smooth / T)):
                current_val = val_new
                A = A_new
                cur_smooth = smooth_new
                if val_new < best_val:
                    best_val = val_new
                    best_s = s.copy()
                    best_A = A.copy()
            else:
                s[i] = old_si
        else:
            # Block flip
            size = int(np.random.choice([2, 4, 8, 16, 32]))
            if size >= N:
                size = N - 1
            start = np.random.randint(0, N - size + 1)
            indices = np.arange(start, start + size)
            old_vals = s[indices].copy()
            s[indices] = -old_vals
            
            A_new = A.copy()
            for idx, old_v in zip(indices, old_vals):
                ks = np.arange(1, N)
                valid_plus = ks < (N - idx)
                if np.any(valid_plus):
                    idx_plus = idx + ks[valid_plus]
                    A_new[ks[valid_plus]] += -2.0 * old_v * s[idx_plus]
                valid_minus = ks <= idx
                if np.any(valid_minus):
                    idx_minus = idx - ks[valid_minus]
                    A_new[ks[valid_minus]] += -2.0 * old_v * s[idx_minus]
            
            val_new = _max_abs_acf_from_array(A_new)
            smooth_new = _smooth_surrogate(A_new, beta)
            delta_smooth = smooth_new - cur_smooth
            if val_new < current_val or (delta_smooth <= 0 or np.random.random() < math.exp(-delta_smooth / T)):
                current_val = val_new
                A = A_new
                cur_smooth = smooth_new
                if val_new < best_val:
                    best_val = val_new
                    best_s = s.copy()
                    best_A = A.copy()
            else:
                s[indices] = old_vals
        
        T *= cooling
    
    return best_s, best_A, best_val

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    result = np.ones(n)
    for i in range(1, n):
        count = bin(i).count('1')
        result[i] = 1.0 if count % 2 == 0 else -1.0
    return result

def _legendre_sequence(n):
    """Generate a sequence based on Legendre-like pattern."""
    result = np.ones(n)
    for i in range(n):
        phase = (i * i) % 4
        if phase in [1, 2]:
            result[i] = -1.0
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Generate diverse initializations
    initializations = []
    
    # Thue-Morse sequence
    initializations.append(_thue_morse(N))
    
    # Legendre-like sequence
    initializations.append(_legendre_sequence(N))
    
    # Halton sequences
    for base in [2, 3, 5, 7, 11]:
        halton_vals = np.zeros(N)
        for i in range(1, N + 1):
            f = 1.0 / base
            val = 0.0
            j = i
            while j > 0:
                f /= base
                val += f * (j % base)
                j //= base
            halton_vals[i - 1] = val
        s = np.where(halton_vals < 0.5, 1.0, -1.0)
        initializations.append(s)
    
    # Random initializations
    for _ in range(8):
        s = np.random.randint(0, 2, size=N) * 2 - 1
        initializations.append(s)
    
    # All ones and all minus ones
    initializations.append(np.ones(N))
    initializations.append(-np.ones(N))
    
    # Alternating
    s_alt = np.ones(N)
    s_alt[1::2] = -1.0
    initializations.append(s_alt)
    
    # Sine-based
    for freq in [0.1, 0.2, 0.3, 0.5]:
        x = np.linspace(0, 2 * np.pi * freq, N)
        s_sine = np.where(np.sin(x) >= 0, 1.0, -1.0)
        initializations.append(s_sine)
    
    # Quadratic phase
    idx = np.arange(N)
    for denom in [N, 2*N, 3*N]:
        s_quad = np.where(np.cos(2 * np.pi * idx ** 2 / denom) >= 0, 1.0, -1.0)
        initializations.append(s_quad)
    
    # Evaluate each initialization with local search
    for s in initializations:
        if time.time() > deadline - 15:
            break
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search
        s, A, current_val = _local_search(s, A, current_val, deadline - 15, max_passes=30)
        
        # Block local search
        s, A, current_val = _block_local_search(s, A, current_val, deadline - 15, max_passes=10)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Continue with random restarts if time permits
    while time.time() < deadline - 15:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search
        s, A, current_val = _local_search(s, A, current_val, deadline - 15, max_passes=20)
        
        # Block local search
        s, A, current_val = _block_local_search(s, A, current_val, deadline - 15, max_passes=5)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Final polish with smooth simulated annealing on the best sequence
    if time.time() < deadline - 5:
        s = best_seq.copy()
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        s, A, current_val = _simulated_annealing_smooth(s, A, current_val, deadline - 5, T0=0.03, cooling=0.9995, beta=50.0)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    return best_seq
