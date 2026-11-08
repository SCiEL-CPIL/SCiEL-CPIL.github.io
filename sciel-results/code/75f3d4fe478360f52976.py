"""Combined: FFT-based autoconv, efficient delta updates, and block-level structural moves."""
import time
import numpy as np

N = 400


def _fft_autoconv(h):
    """Compute linear autoconvolution via FFT (fast)."""
    n = len(h)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    C = np.fft.irfft(H * H, n=fft_size)
    return C[:size]


def _eval(h):
    """Return (max_abs_autoconv, integral_sq, ratio)."""
    c = _fft_autoconv(h)
    max_abs = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    int_sq = s * s
    if int_sq < 1e-12:
        ratio = max_abs
    else:
        ratio = max_abs / int_sq
    return max_abs, int_sq, ratio


def _delta_flip(c, h, idx):
    """
    Compute the change in autoconvolution when flipping h[idx].
    c_new[k] = c[k] + delta[k]
    where delta[k] = -2 * h[idx] * (h[idx + (k - (N-1))] if in range) - 2 * h[idx] * (h[(k - (N-1)) - idx] if in range)
    Actually, let's use the direct formula:
    For lag L = k - (N-1), the terms involving h[idx] are:
    - h[idx] * h[idx + L] (if 0 <= idx+L < N)
    - h[idx - L] * h[idx] (if 0 <= idx-L < N)
    Flipping h[idx] to -h[idx] changes these products from +val to -val, so delta = -2*val.
    """
    n = len(h)
    val = h[idx]
    delta = np.zeros(2 * n - 1)
    
    # h[idx] as first factor: j=idx, j2=idx+lag
    for j2 in range(n):
        lag = j2 - idx
        c_idx = lag + (n - 1)
        if 0 <= c_idx < len(delta):
            delta[c_idx] += -2 * val * h[j2]
    
    # h[idx] as second factor: j2=idx, j=idx-lag
    for j in range(n):
        lag = idx - j
        c_idx = lag + (n - 1)
        if 0 <= c_idx < len(delta):
            delta[c_idx] += -2 * val * h[j]
    
    return delta


def _delta_block_flip(c, h, start, length):
    """
    Compute the change in autoconvolution when flipping a block of signs.
    For efficiency, we compute the delta for each index in the block and sum them.
    However, this is O(length * N) which is fine for small blocks.
    """
    delta = np.zeros(2 * len(h) - 1)
    for idx in range(start, min(start + length, len(h))):
        delta += _delta_flip(c, h, idx)
    return delta


def _hill_climb_fast(h, max_iter=1000, deadline=140.0, block_sizes=[1, 2, 4, 8]):
    """
    Fast local search using delta updates.
    Mixes single flips and small block flips to escape local minima.
    """
    c = _fft_autoconv(h)
    max_abs, int_sq, ratio = _eval(h)
    s = float(np.sum(h))
    n = len(h)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        best_delta = None
        best_new_ratio = ratio
        best_idx = -1
        best_block_size = 1
        best_new_s = s
        
        # Try several random moves: single flips and small block flips
        for _ in range(10):
            # Choose a move type
            if np.random.random() < 0.7:
                # Single flip
                idx = np.random.randint(n)
                delta = _delta_flip(c, h, idx)
                c_new = c + delta
                new_max = float(np.max(np.abs(c_new)))
                new_s = s - 2 * h[idx]
                new_int_sq = new_s * new_s
                if new_int_sq < 1e-12:
                    new_ratio = new_max
                else:
                    new_ratio = new_max / new_int_sq
                
                if new_ratio < best_new_ratio:
                    best_new_ratio = new_ratio
                    best_delta = delta
                    best_idx = idx
                    best_block_size = 1
                    best_new_s = new_s
            else:
                # Block flip
                bs = np.random.choice(block_sizes)
                if bs >= n:
                    continue
                start = np.random.randint(0, n - bs + 1)
                delta = _delta_block_flip(c, h, start, bs)
                c_new = c + delta
                new_max = float(np.max(np.abs(c_new)))
                # Compute new sum
                old_block_sum = np.sum(h[start:start+bs])
                new_s = s - 2 * old_block_sum
                new_int_sq = new_s * new_s
                if new_int_sq < 1e-12:
                    new_ratio = new_max
                else:
                    new_ratio = new_max / new_int_sq
                
                if new_ratio < best_new_ratio:
                    best_new_ratio = new_ratio
                    best_delta = delta
                    best_idx = start
                    best_block_size = bs
                    best_new_s = new_s
        
        if best_delta is not None and best_new_ratio < ratio:
            if best_block_size == 1:
                h[best_idx] = -h[best_idx]
            else:
                h[best_idx:best_idx + best_block_size] = -h[best_idx:best_idx + best_block_size]
            c = c + best_delta
            s = best_new_s
            max_abs, int_sq, ratio = _eval(h)
    
    return h, ratio


def _build_rudin_shapiro(n):
    """Classic Rudin-Shapiro sequence of length >= n, truncated."""
    A = np.array([1.0])
    B = np.array([1.0])
    while len(A) < n:
        A, B = np.concatenate([A, B]), np.concatenate([A, -B])
    return A[:n]


def _build_random_block_sequence(n, block_sizes=[16, 32, 64]):
    """
    Construct a sequence by concatenating random blocks of fixed size.
    This introduces a different structural family than pure random or Rudin-Shapiro.
    """
    h = np.zeros(n)
    pos = 0
    while pos < n:
        bs = np.random.choice(block_sizes)
        if pos + bs > n:
            bs = n - pos
        block = np.random.choice([-1.0, 1.0], size=bs)
        h[pos:pos+bs] = block
        pos += bs
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Strategy 1: Rudin-Shapiro
    h_rs = _build_rudin_shapiro(N)
    _, _, val_rs = _eval(h_rs)
    best_h = h_rs.copy()
    best_val = val_rs
    
    # Strategy 2: Random starts with fast local search (single + block flips)
    np.random.seed(42)
    for seed in range(150):
        if time.time() > deadline:
            break
        
        h_rand = np.random.choice([-1.0, 1.0], size=N)
        h_improved, val = _hill_climb_fast(h_rand, max_iter=300, deadline=deadline)
        
        if val < best_val:
            best_val = val
            best_h = h_improved.copy()
    
    # Strategy 3: Block-structured starts
    for seed in range(50):
        if time.time() > deadline:
            break
        
        h_block = _build_random_block_sequence(N)
        h_improved, val = _hill_climb_fast(h_block, max_iter=300, deadline=deadline)
        
        if val < best_val:
            best_val = val
            best_h = h_improved.copy()
    
    # Strategy 4: Polish the best found with more iterations
    if time.time() < deadline - 5:
        h_final, val_final = _hill_climb_fast(best_h.copy(), max_iter=1000, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
