"""Spectral optimization for Erdos minimum overlap."""
import time
import numpy as np

N = 100
DEADLINE = 120.0
SAFETY_MARGIN = 15.0  # seconds before hard limit


def _compute_overlap_max(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    h_sum = np.sum(h)
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    overlap_max = h_sum - np.min(autocorr)
    return overlap_max


def _evaluate(h):
    """Evaluate the objective: maximum overlap of h with shifted (1-h)."""
    if not np.all((h >= -1e-9) & (h <= 1 + 1e-9)):
        return float('inf')
    if abs(np.sum(h) - N / 2) > 1e-6:
        return float('inf')
    return _compute_overlap_max(h)


def _fix_sum(h):
    """Adjust h so that sum is exactly N/2, keeping values in [0,1]."""
    h = h.copy()
    diff = N / 2 - np.sum(h)
    if abs(diff) < 1e-10:
        return h
    # Distribute correction to elements not at boundary
    mask = (h > 1e-6) & (h < 1 - 1e-6)
    n_free = np.sum(mask)
    if n_free == 0:
        # All at boundaries; find one to adjust
        if diff > 0:
            # Need to increase sum; find element closest to 0
            idx = np.argmin(h)
        else:
            # Need to decrease sum; find element closest to 1
            idx = np.argmax(h)
        h[idx] = np.clip(h[idx] + diff, 0, 1)
        # May still not be exact; do iterative
        for _ in range(20):
            diff = N / 2 - np.sum(h)
            if abs(diff) < 1e-10:
                break
            mask = (h > 1e-6) & (h < 1 - 1e-6)
            n_free = np.sum(mask)
            if n_free == 0:
                if diff > 0:
                    idx = np.argmin(h)
                else:
                    idx = np.argmax(h)
                h[idx] = np.clip(h[idx] + diff, 0, 1)
            else:
                h[mask] += diff / n_free
                h = np.clip(h, 0, 1)
    else:
        h[mask] += diff / n_free
        h = np.clip(h, 0, 1)
        # Iterative refinement
        for _ in range(20):
            diff = N / 2 - np.sum(h)
            if abs(diff) < 1e-10:
                break
            mask = (h > 1e-6) & (h < 1 - 1e-6)
            n_free = np.sum(mask)
            if n_free == 0:
                if diff > 0:
                    idx = np.argmin(h)
                else:
                    idx = np.argmax(h)
                h[idx] = np.clip(h[idx] + diff, 0, 1)
            else:
                h[mask] += diff / n_free
                h = np.clip(h, 0, 1)
    return h


def _simulated_annealing(h0, t_start=1.0, t_end=0.01, n_iter=50000):
    """Simulated annealing to minimize max overlap."""
    h = h0.copy()
    h = _fix_sum(h)
    best_h = h.copy()
    best_score = _evaluate(h)
    current_score = best_score
    
    start_time = time.time()
    elapsed = 0.0
    for it in range(n_iter):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        
        # Temperature schedule (geometric cooling)
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Generate neighbor: pick 2 random indices, swap or adjust
        i, j = np.random.choice(N, 2, replace=False)
        delta = np.random.uniform(-0.3, 0.3)
        new_h = h.copy()
        new_h[i] = np.clip(h[i] + delta, 0, 1)
        new_h[j] = np.clip(h[j] - delta, 0, 1)
        new_h = _fix_sum(new_h)
        
        new_score = _evaluate(new_h)
        
        # Acceptance criterion
        if new_score < current_score:
            h = new_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
        else:
            # Accept with probability exp(-delta/T)
            if np.random.random() < np.exp((current_score - new_score) / t):
                h = new_h
                current_score = new_score
    
    return best_h, best_score


def construct():
    start_time = time.time()
    
    # Start with a good initial guess: alternating pattern with slight variation
    best_h = np.full(N, 0.5)
    best_score = _evaluate(best_h)
    
    # Try several diverse starting points
    starts = []
    
    # 1. Uniform
    starts.append(np.full(N, 0.5))
    
    # 2. Alternating 0 and 1
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt = _fix_sum(h_alt)
    starts.append(h_alt)
    
    # 3. Low-frequency sinusoid
    i = np.arange(N)
    h_sin = 0.5 + 0.4 * np.sin(2 * np.pi * i / N)
    h_sin = np.clip(h_sin, 0, 1)
    h_sin = _fix_sum(h_sin)
    starts.append(h_sin)
    
    # 4. High-frequency sinusoid
    h_hf = 0.5 + 0.4 * np.sin(2 * np.pi * 7 * i / N)
    h_hf = np.clip(h_hf, 0, 1)
    h_hf = _fix_sum(h_hf)
    starts.append(h_hf)
    
    # 5. Random
    np.random.seed(42)
    h_rand = np.random.uniform(0, 1, N)
    h_rand = _fix_sum(h_rand)
    starts.append(h_rand)
    
    # Evaluate all starts and pick best
    for h in starts:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        score = _evaluate(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Run simulated annealing from the best start
    if time.time() < DEADLINE - SAFETY_MARGIN:
        best_h, best_score = _simulated_annealing(best_h)
    
    # Final validation and sum fix
    best_h = np.clip(best_h, 0, 1)
    best_h = _fix_sum(best_h)
    
    return best_h
