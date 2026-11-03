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

def _update_acf_flip(A, s, i, old_si):
    """
    Update ACF after flipping s[i] in-place.
    old_si is the value before the flip.
    Returns new A array (copy).
    """
    A_new = A.copy()
    for k in range(1, N):
        change = 0.0
        if i + k < N:
            change -= 2.0 * old_si * s[i + k]
        if i - k >= 0:
            change -= 2.0 * s[i - k] * old_si
        A_new[k] += change
    return A_new

def _update_acf_flip_block(A, s, indices, old_vals):
    """
    Update ACF after flipping multiple indices in s (already flipped).
    indices: array of indices to flip
    old_vals: array of old values at those indices (before flip)
    Returns new A array (copy).
    """
    A_new = A.copy()
    for k in range(1, N):
        change = 0.0
        for idx, old_v in zip(indices, old_vals):
            if idx + k < N:
                change -= 2.0 * old_v * s[idx + k]
            if idx - k >= 0:
                change -= 2.0 * s[idx - k] * old_v
        A_new[k] += change
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
            A_new = _update_acf_flip(A, s, i, old_si)
            val_new = _max_abs_acf_from_array(A_new)
            if val_new < current_val:
                current_val = val_new
                A = A_new
                improved = True
            else:
                s[i] = old_si
    return s, A, current_val

def _block_local_search(s, A, current_val, deadline, max_passes=30):
    """Perform block-flip local search."""
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        # Try flipping blocks of size 1-4
        for block_size in [1, 2, 3, 4]:
            if time.time() > deadline:
                break
            num_blocks = N // block_size
            for _ in range(min(num_blocks, 20)):
                if time.time() > deadline:
                    break
                start = np.random.randint(0, N - block_size + 1)
                indices = np.arange(start, start + block_size)
                old_vals = s[indices].copy()
                s[indices] = -old_vals
                A_new = _update_acf_flip_block(A, s, indices, old_vals)
                val_new = _max_abs_acf_from_array(A_new)
                if val_new < current_val:
                    current_val = val_new
                    A = A_new
                    improved = True
                else:
                    s[indices] = old_vals
    return s, A, current_val

def _simulated_annealing(s, A, current_val, deadline, T0=1.0, cooling=0.95):
    """Perform simulated annealing."""
    T = T0
    best_val = current_val
    best_s = s.copy()
    best_A = A.copy()
    while time.time() < deadline and T > 1e-6:
        i = np.random.randint(0, N)
        old_si = s[i]
        s[i] = -old_si
        A_new = _update_acf_flip(A, s, i, old_si)
        val_new = _max_abs_acf_from_array(A_new)
        delta = val_new - current_val
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_val = val_new
            A = A_new
            if val_new < best_val:
                best_val = val_new
                best_s = s.copy()
                best_A = A.copy()
        else:
            s[i] = old_si
        T *= cooling
    return best_s, best_A, best_val

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    result = np.ones(n)
    for i in range(1, n):
        # Thue-Morse: t(i) = parity of number of 1s in binary representation of i
        count = bin(i).count('1')
        result[i] = 1.0 if count % 2 == 0 else -1.0
    return result

def _legendre_sequence(n):
    """Generate a sequence based on Legendre-like pattern."""
    result = np.ones(n)
    for i in range(n):
        # Use a simple quadratic phase
        phase = (i * i) % 4
        if phase in [1, 2]:
            result[i] = -1.0
    return result

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Generate diverse initializations
    initializations = []
    
    # Thue-Morse sequence
    initializations.append(_thue_morse(N))
    
    # Legendre-like sequence
    initializations.append(_legendre_sequence(N))
    
    # Halton sequences
    bases = [2, 3, 5, 7, 11, 13]
    for base in bases:
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
    for _ in range(10):
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
    
    # Evaluate each initialization with local search
    for s in initializations:
        if time.time() > deadline - 10:
            break
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search
        s, A, current_val = _local_search(s, A, current_val, deadline - 10, max_passes=30)
        
        # Block local search
        s, A, current_val = _block_local_search(s, A, current_val, deadline - 10, max_passes=10)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Continue with random restarts if time permits
    while time.time() < deadline - 10:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search
        s, A, current_val = _local_search(s, A, current_val, deadline - 10, max_passes=20)
        
        # Block local search
        s, A, current_val = _block_local_search(s, A, current_val, deadline - 10, max_passes=5)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Final polish with simulated annealing on the best sequence
    if time.time() < deadline - 5:
        s = best_seq.copy()
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        s, A, current_val = _simulated_annealing(s, A, current_val, deadline - 5, T0=0.5, cooling=0.97)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    return best_seq
