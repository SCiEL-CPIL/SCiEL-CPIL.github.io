"""Signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400


def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the full autoconvolution of x."""
    # Full autoconvolution via FFT
    n = len(x)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    X = np.fft.rfft(x, fft_size)
    conv = np.fft.irfft(X * np.conj(X), fft_size)[:size]
    return np.max(np.abs(conv))


def _score(x):
    """Score = max_abs_autoconv / (sum(x)^2). Lower is better."""
    s = np.sum(x)
    if s == 0:
        return np.inf
    return _autoconv_max_abs(x) / (s * s)


def _barker_pattern(n):
    """Generate a Barker-like pattern of length n by tiling/reflecting Barker sequences.
    
    Barker sequences are known for low peak sidelobe levels in autocorrelation.
    We use the length-13 Barker sequence as a base and extend to length n.
    """
    # Barker sequence of length 13 (one of the longest known)
    barker13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    
    # Tile and trim to length n
    pattern = np.tile(barker13, n // len(barker13) + 1)[:n]
    return pattern.astype(np.float64)


def _golay_pair(n):
    """Generate a Golay complementary sequence pair of length n (padded if needed).
    
    Golay sequences have the property that the sum of their autocorrelations
    is zero for all non-zero lags. We use one member as our initial pattern.
    """
    # Start with a known Golay pair of length 4: [1,1,1,1] and [1,1,-1,-1]
    # Build up to desired length via Kronecker products
    a = np.array([1.0, 1.0])
    b = np.array([1.0, -1.0])
    
    while len(a) < n:
        a = np.concatenate([np.concatenate([a, a]), np.concatenate([a, -a])])
        # This doubles the length each time; adjust b similarly
        b = np.concatenate([np.concatenate([b, b]), np.concatenate([b, -b])])
    
    a = a[:n]
    return a


def _random_binary(n, seed=None):
    """Random binary sequence of length n."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.choice([-1.0, 1.0], size=n)


def _sign_flip_local_search(x, time_limit=8.0):
    """Local search by single sign flips to reduce max abs autoconvolution."""
    x = x.copy()
    current_score = _score(x)
    start = time.time()
    
    # Precompute FFT size
    n = len(x)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    
    # We'll iterate: for each position, check if flipping improves score
    # To be efficient, we do multiple passes until no improvement or time limit
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Shuffle order to avoid bias
        order = np.random.permutation(n)
        for i in order:
            if (time.time() - start) > time_limit - 0.5:
                break
            # Try flipping position i
            old_val = x[i]
            x[i] = -old_val
            new_score = _score(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                x[i] = old_val
    
    return x, current_score


def _pair_flip_local_search(x, time_limit=8.0):
    """Local search by pair sign flips (flip two positions at once)."""
    x = x.copy()
    current_score = _score(x)
    start = time.time()
    n = len(x)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try random pairs
        for _ in range(2000):
            if (time.time() - start) > time_limit - 0.5:
                break
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            old_i, old_j = x[i], x[j]
            x[i] = -old_i
            x[j] = -old_j
            new_score = _score(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                x[i] = old_i
                x[j] = old_j
    
    return x, current_score


def construct():
    """Construct a signed step sequence minimizing max abs autoconvolution / (sum)^2.
    
    Uses Barker/Golay-inspired initial patterns and local sign-flip optimization.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    np.random.seed(42)
    
    # Generate diverse initial patterns
    initial_patterns = [
        _barker_pattern(N),
        _golay_pair(N),
        _random_binary(N, seed=1),
        _random_binary(N, seed=2),
        _random_binary(N, seed=3),
        # Sparse pattern: mostly +1 with scattered -1
        np.ones(N),
    ]
    # Add some structured patterns
    # Alternating blocks
    block_pattern = np.ones(N)
    block_size = 8
    for i in range(0, N, block_size * 2):
        block_pattern[i:i+block_size] = -1.0
    initial_patterns.append(block_pattern)
    
    # Tapered pattern
    taper = np.ones(N)
    taper[:N//4] = -1.0
    taper[3*N//4:] = -1.0
    initial_patterns.append(taper)
    
    best_x = None
    best_score = np.inf
    
    for idx, init in enumerate(initial_patterns):
        if time.time() > deadline - 20.0:
            break
        
        # First pass: single-flip local search
        x, sc = _sign_flip_local_search(init, time_limit=min(8.0, max(1.0, deadline - time.time() - 10.0)))
        
        # Second pass: pair-flip local search
        if time.time() < deadline - 10.0:
            x, sc = _pair_flip_local_search(x, time_limit=min(6.0, max(1.0, deadline - time.time() - 5.0)))
        
        # Final single-flip polish
        if time.time() < deadline - 5.0:
            x, sc = _sign_flip_local_search(x, time_limit=min(4.0, max(0.5, deadline - time.time() - 2.0)))
        
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
