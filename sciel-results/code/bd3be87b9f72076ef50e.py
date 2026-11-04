import numpy as np
import time

N = 100

def _overlap_max(h):
    """Compute max_s sum_i h[i]*(1-h[(i+s)%N]) for all shifts s."""
    # autocorrelation: corr[s] = sum_i h[i]*h[(i+s)%N]
    # overlap[s] = sum(h) - corr[s]
    # We want max_s overlap[s] = sum(h) - min_s corr[s]
    # Use FFT for fast autocorrelation
    h_ext = np.zeros(2 * N)
    h_ext[:N] = h
    fft_h = np.fft.rfft(h_ext, n=2*N)
    corr = np.fft.irfft(fft_h * np.conj(fft_h), n=2*N)
    # corr[s] for s=0..N-1 gives sum_i h[i]*h[(i+s)%N]
    # Actually, the circular autocorrelation
    corr_circ = np.zeros(N)
    for s in range(N):
        corr_circ[s] = np.sum(h * np.roll(h, -s))
    min_corr = np.min(corr_circ)
    return np.sum(h) - min_corr

def _overlap_max_fast(h):
    """Faster version using FFT for circular autocorrelation."""
    # Circular autocorrelation via FFT
    fft_h = np.fft.rfft(h, n=N)
    power = fft_h * np.conj(fft_h)
    corr = np.fft.irfft(power, n=N)
    min_corr = np.min(corr)
    return np.sum(h) - min_corr

def _project(h):
    """Project h to satisfy sum(h)=N/2 and 0<=h[i]<=1."""
    h = np.clip(h, 0, 1)
    # Adjust to make sum exactly N/2
    target = N / 2.0
    current_sum = np.sum(h)
    if current_sum == target:
        return h
    # Simple projection: scale and clip iteratively
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to increase sum; add to elements that are < 1
            mask = h < 1.0
            if not np.any(mask):
                break
            add_per = diff / np.sum(mask)
            h[mask] = np.clip(h[mask] + add_per, 0, 1)
        else:
            # Need to decrease sum; subtract from elements that are > 0
            mask = h > 0.0
            if not np.any(mask):
                break
            sub_per = (-diff) / np.sum(mask)
            h[mask] = np.clip(h[mask] - sub_per, 0, 1)
    # Final exact fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute remaining diff to one element
        idx = np.argmax(h < 1.0) if diff > 0 else np.argmin(h > 0.0)
        if diff > 0:
            h[idx] = min(1.0, h[idx] + diff)
        else:
            h[idx] = max(0.0, h[idx] + diff)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max_fast(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Sinusoidal patterns
    for freq in range(1, 10):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            x = np.arange(N)
            h = 0.5 + amp * np.cos(2 * np.pi * freq * x / N)
            h = np.clip(h, 0, 1)
            h = _project(h)
            candidates.append(h.copy())
    
    # 3. Block patterns
    for block_size in [2, 3, 4, 5, 7, 10]:
        for offset in range(block_size):
            h = np.zeros(N)
            for i in range(N):
                if (i + offset) % (2 * block_size) < block_size:
                    h[i] = 1.0
                else:
                    h[i] = 0.0
            h = _project(h)
            candidates.append(h.copy())
    
    # 4. Random patterns
    np.random.seed(42)
    for _ in range(50):
        h = np.random.uniform(0, 1, N)
        h = _project(h)
        candidates.append(h.copy())
    
    # 5. Two-level patterns with varying duty cycle
    for duty in np.linspace(0.1, 0.4, 8):
        for freq in [1, 2, 3, 4, 5]:
            x = np.arange(N)
            phase = (freq * x / N) % 1.0
            h = np.where(phase < duty, 1.0, 0.0)
            h = _project(h)
            candidates.append(h.copy())
    
    # Evaluate all candidates
    for h in candidates:
        score = _overlap_max_fast(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if time.time() > deadline:
        return best_h
    
    # Local search / simulated annealing refinement
    current_h = best_h.copy()
    current_score = best_score
    
    # Simulated annealing
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    
    step_count = 0
    while time.time() < deadline - 5.0:
        # Random perturbation
        idx1 = np.random.randint(N)
        idx2 = np.random.randint(N)
        if idx1 == idx2:
            continue
        
        delta = np.random.uniform(-0.1, 0.1)
        new_h = current_h.copy()
        new_h[idx1] = np.clip(new_h[idx1] + delta, 0, 1)
        new_h[idx2] = np.clip(new_h[idx2] - delta, 0, 1)
        new_h = _project(new_h)
        
        new_score = _overlap_max_fast(new_h)
        
        if new_score < current_score:
            current_h = new_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
        else:
            # Accept with probability exp(-(new_score - current_score)/T)
            if T > T_min:
                prob = np.exp(-(new_score - current_score) / T)
                if np.random.random() < prob:
                    current_h = new_h
                    current_score = new_score
        
        T *= cooling
        step_count += 1
        
        # Periodic restart from best if stuck
        if step_count % 5000 == 0 and current_score > best_score * 1.01:
            current_h = best_h.copy()
            current_score = best_score
            T = 0.05
    
    return best_h
