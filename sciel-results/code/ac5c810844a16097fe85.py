"""
Signed autocorrelation C3-prime task.
Minimizes max|autoconv(x)| / (sum(x))^2 for x in {-1, 1}^N.
Strategy:
1. Fast evaluation via FFT.
2. Smart local search using incremental updates of the autoconvolution.
3. Multi-start with diverse initializations (random, structured, smoothed).
4. Deadline guard to ensure completion within time limit.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _fft_size(n):
    """Return next power of 2 >= n."""
    size = 1
    while size < n:
        size <<= 1
    return size

def _init_autoconv(x):
    """Compute full autoconvolution via FFT."""
    n = len(x)
    size = _fft_size(2 * n - 1)
    F = np.fft.rfft(x, size)
    c = np.fft.irfft(F * np.conj(F), size)
    return c[:2 * n - 1]

def _score(x):
    """Compute C3' = max|autoconv| / (sum(x))^2."""
    s = x.sum()
    if abs(s) < 1e-12:
        return float('inf')
    c = _init_autoconv(x)
    return float(np.max(np.abs(c)) / (s * s))

def _local_search(x, time_limit=10.0, max_iter=50000):
    """
    Local search by flipping signs.
    Uses incremental update of autoconvolution for speed.
    """
    start_time = time.time()
    n = len(x)
    size = _fft_size(2 * n - 1)
    
    # Precompute FFT of x for incremental updates
    F = np.fft.rfft(x, size)
    c = np.fft.irfft(F * np.conj(F), size)[:2 * n - 1]
    
    s = x.sum()
    best_score = float(np.max(np.abs(c)) / (s * s)) if abs(s) > 1e-12 else float('inf')
    best_x = x.copy()
    
    # Precompute indices for convolution updates
    # When flipping x[j] to -x[j], the change in autoconvolution is:
    # delta_c[k] = -2 * x[j] * (x * delta_j)[k]
    # where delta_j is 1 at position j, 0 elsewhere.
    # (x * delta_j)[k] = x[k-j] if 0 <= k-j < n, else 0.
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping each element
        for j in range(n):
            # Update autoconvolution
            # delta_c[k] = -2 * x[j] * x[k-j] for valid k
            # We need to update c in place
            
            old_val = x[j]
            new_val = -old_val
            diff = new_val - old_val  # -2 * old_val
            
            # Update c
            # c[k] += diff * x[k-j] for k where 0 <= k-j < n
            # i.e., k in [j, j+n-1]
            lo = j
            hi = min(j + n, 2 * n - 1)
            # x indices: k - j, so 0 to hi - lo - 1
            c[lo:hi] += diff * x[0:hi - lo]
            
            # Update sum
            s += diff
            
            # Compute score
            if abs(s) > 1e-12:
                score = float(np.max(np.abs(c)) / (s * s))
            else:
                score = float('inf')
            
            if score < best_score:
                best_score = score
                best_x = x.copy()
                x[j] = new_val
                # Update F for future incremental updates? 
                # Actually, we're updating c directly, so F is stale.
                # But we only use c for scoring, not F. So it's fine.
                # However, if we want to continue incremental updates correctly,
                # we need to keep c consistent. It is.
            else:
                # Revert
                c[lo:hi] -= diff * x[0:hi - lo]
                s -= diff
                # x[j] remains old_val (unchanged)
    
    return best_x, best_score

def _construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_x = None
    best_score = float('inf')
    
    # Phase 1: All ones baseline
    x = np.ones(N)
    score = _score(x)
    best_x = x.copy()
    best_score = score
    
    # Phase 2: Random starts with local search
    np.random.seed(42)
    n_trials = 50
    for i in range(n_trials):
        if time.time() > deadline - 5.0:
            break
        
        # Diverse initialization
        if i % 3 == 0:
            # Pure random
            x = np.random.choice([-1.0, 1.0], size=N)
        elif i % 3 == 1:
            # Block pattern
            block_size = 2 ** (i % 6)
            if N % block_size != 0:
                block_size = 2
            n_blocks = N // block_size
            pattern = np.random.choice([-1.0, 1.0], size=n_blocks)
            x = np.repeat(pattern, block_size)
        else:
            # Smoothed random
            x = np.random.choice([-1.0, 1.0], size=N)
            # Simple smoothing: majority vote of neighbors
            x_smooth = x.copy()
            for k in range(1, N - 1):
                if x[k-1] == x[k+1]:
                    x_smooth[k] = x[k-1]
            x = x_smooth
        
        score = _score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Local search
        if time.time() < deadline - 10.0:
            x_ls, score_ls = _local_search(x, time_limit=min(5.0, deadline - time.time() - 5.0))
            if score_ls < best_score:
                best_score = score_ls
                best_x = x_ls
    
    # Phase 3: Final intensive local search from best
    if time.time() < deadline - 2.0 and best_x is not None:
        x_final, score_final = _local_search(best_x, time_limit=min(10.0, deadline - time.time() - 1.0))
        if score_final < best_score:
            best_x = x_final
            best_score = score_final
    
    return best_x

def construct():
    """Construct a signed step function minimizing the maximum absolute autoconvolution 
    divided by squared integral."""
    return _construct()

# EVOLVE-BLOCK-END
