"""
Rewritten program: Spectral-domain optimization for signed step functions.

Task: minimize max_k |c_k| / (sum x_i)^2 where c = autoconvolution of x.

Key structural change from previous attempts:
- Previous: time-domain gradient descent with soft-max surrogate
- New: spectral-domain optimization using the relationship between
  autoconvolution and power spectrum, combined with coordinate descent
  and local search in the binary space.

The autoconvolution c_k relates to the DFT: if X = DFT(x), then the
autoconvolution's DFT is |X|^2. So max|c_k| is related to the peak of
the inverse DFT of |X|^2.

For binary sequences (±1), we can use:
- Coordinate descent: flip one bit at a time, accept if it reduces max|c_k|
- Spectral initialization: start from sequences with flat power spectrum
- Local search with simulated annealing for escaping local optima

The constant sequence (all 1s) gives ratio 1/N. For N=400, that's 0.0025.
The theoretical lower bound for binary sequences is related to the
minimum possible peak sidelobe level.
"""
import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _autoconv(x):
    """Compute full autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2 * N - 1]
    return C


def _compute_ratio(x):
    """Compute max|c_k| / (sum x_i)^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s ** 2)


def _max_abs_c(x):
    """Return max|c_k| for autoconvolution of x."""
    c = _autoconv(x)
    return np.max(np.abs(c))


def _local_search_binary(x, deadline, max_iters=100000):
    """
    Coordinate descent / local search on binary sequence.
    Flip one bit at a time; accept if it reduces max|c_k|.
    Use simulated annealing to escape local optima.
    """
    x = x.copy().astype(np.float64)
    # Ensure binary
    x = np.sign(x)
    x[x == 0] = 1.0
    
    current_val = _max_abs_c(x)
    best_val = current_val
    best_x = x.copy()
    
    T = 1.0  # temperature
    T_min = 0.01
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick random position to flip
        i = np.random.randint(N)
        old_val = x[i]
        x[i] = -x[i]
        
        new_val = _max_abs_c(x)
        
        if new_val < current_val or np.random.random() < np.exp((current_val - new_val) / max(T, 1e-10)):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()
        else:
            x[i] = old_val
        
        # Cool down
        if it % 100 == 0:
            T = max(T_min, T * cooling)
    
    return best_x, best_val


def _spectral_init(deadline):
    """
    Generate initial binary sequences with good spectral properties.
    Use the fact that we want flat |X|^2 in frequency domain.
    """
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 3. Random binary
    np.random.seed(42)
    for _ in range(5):
        candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 4. M-sequence-like (using a simple LFSR)
    # Generate a pseudo-random binary sequence with good autocorrelation properties
    def lfsr_sequence(n, poly=0x59):
        """Generate LFSR sequence. poly is the polynomial in hex."""
        seq = []
        state = 0xFFFF
        for _ in range(n):
            # Output bit
            bit = state & 1
            seq.append(1.0 if bit else -1.0)
            # Shift and feedback
            state >>= 1
            if bit:
                state ^= poly
        return np.array(seq[:n])
    
    # Try a few LFSR configurations
    for poly in [0x59, 0x69, 0x79, 0x159, 0x169]:
        try:
            seq = lfsr_sequence(N, poly)
            candidates.append(seq)
        except:
            pass
    
    return candidates


def _refine_continuous(x0, deadline, max_iters=2000):
    """
    Refine a binary sequence by allowing small continuous perturbations,
    then round back to binary. This can help escape local optima.
    """
    x = x0.copy().astype(np.float64)
    current_val = _max_abs_c(x)
    best_val = current_val
    best_x = x.copy()
    
    T = 0.5
    lr = 0.1
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Add small noise
        noise = np.random.randn(N) * 0.1
        x_new = x + noise
        
        # Keep values in [-1, 1] range (soft constraint)
        x_new = np.tanh(x_new)
        
        new_val = _max_abs_c(x_new)
        
        if new_val < current_val or np.random.random() < np.exp((current_val - new_val) / max(T, 1e-10)):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_x = x_new.copy()
            x = x_new
        else:
            # Don't update x
            pass

        # Cool and reduce learning rate
        if it % 50 == 0:
            T = max(0.01, T * 0.98)
            lr = max(0.01, lr * 0.99)
    
    # Round to binary
    x_rounded = np.sign(best_x)
    x_rounded[x_rounded == 0] = 1.0
    return x_rounded, _max_abs_c(x_rounded)


def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Baseline: all ones
    x_baseline = np.ones(N)
    val_baseline = _max_abs_c(x_baseline)
    best_x = x_baseline.copy()
    best_val = val_baseline
    
    # Get spectral initializations
    candidates = _spectral_init(deadline)
    
    # Evaluate each candidate
    for x_cand in candidates:
        if time.time() > deadline:
            break
        val = _max_abs_c(x_cand)
        if val < best_val:
            best_val = val
            best_x = x_cand.copy()
    
    # Run local search from best candidate and a few others
    np.random.seed(42)
    
    # Local search from best
    if time.time() < deadline:
        x_ls, val_ls = _local_search_binary(best_x, deadline, max_iters=50000)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls.copy()
    
    # Local search from random binary
    if time.time() < deadline:
        x_rand = np.random.choice([-1.0, 1.0], size=N)
        x_ls, val_ls = _local_search_binary(x_rand, deadline, max_iters=30000)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls.copy()
    
    # Local search from alternating
    if time.time() < deadline:
        x_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
        x_ls, val_ls = _local_search_binary(x_alt, deadline, max_iters=30000)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls.copy()
    
    # Try continuous refinement on best
    if time.time() < deadline:
        x_ref, val_ref = _refine_continuous(best_x, deadline, max_iters=1000)
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref.copy()
    
    # Final local search polish
    if time.time() < deadline:
        x_ls, val_ls = _local_search_binary(best_x, deadline, max_iters=20000)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls.copy()
    
    # Ensure binary output
    best_x = np.sign(best_x)
    best_x[best_x == 0] = 1.0
    
    return best_x


# EVOLVE-BLOCK-END
