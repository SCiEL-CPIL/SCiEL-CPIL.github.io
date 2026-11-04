"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak(h):
    """Compute max absolute autocorrelation at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[k] for k in 1..n-1 gives positive lags
    # For real signals, acf is symmetric, so max over 1..n-1 suffices
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _construct_m_sequence(polynomial, length):
    """
    Generate an m-sequence (maximal length sequence) of given length
    using a primitive polynomial over GF(2).
    
    polynomial: integer representing the primitive polynomial
                e.g., 0x1A4F12 for x^20 + x^17 + x^15 + x^10 + x^4 + x^3 + x^2 + x + 1
    length: desired sequence length (should be 2^k - 1 for some k)
    
    Returns a numpy array of 0s and 1s of the given length.
    """
    # For N=400, we need to find a suitable m-sequence length
    # 2^9 - 1 = 511, which is close to 400
    # We'll generate a longer sequence and take a window of 400
    
    # Find the smallest 2^k - 1 >= length
    k = 1
    while (2**k - 1) < length:
        k += 1
    
    m_seq_length = 2**k - 1
    
    # Use a known primitive polynomial for the given k
    # For k=9, use x^9 + x^4 + 1 (0x211 in binary: 1 0010 0001)
    # Actually, let's use a standard primitive polynomial
    # For k=9: x^9 + x^4 + 1 -> coefficients at positions 9, 4, 0
    # Represented as bits: bit 9, bit 4, bit 0 are set
    # 2^9 + 2^4 + 2^0 = 512 + 16 + 1 = 529
    
    # Let's use a simpler approach: generate a pseudo-random sequence
    # with good autocorrelation properties using a linear congruential generator
    # or use the Welch bound construction
    
    # Alternative: use a Zadoff-Chu-like sequence or Barker sequence
    # For now, let's use a simple deterministic construction
    
    # Generate using a linear feedback shift register
    # For k=9, use polynomial x^9 + x^5 + x^4 + x^3 + 1 (0x22B)
    # Actually, let's use a well-known primitive polynomial for degree 9
    # x^9 + x^4 + 1 is primitive
    
    # Initialize LFSR state
    state = 1  # Non-zero initial state
    
    # Extract the feedback taps from the polynomial
    # For x^9 + x^4 + 1, taps are at positions 9, 4, 0
    # In a Fibonacci LFSR, we need to know which bits to XOR
    
    # Let's use a different approach: generate a random binary sequence
    # and optimize it, but use the m-sequence as an initial guess
    
    # For now, return a simple pattern
    # Actually, let's just generate a random binary sequence and let the
    # local search optimize it
    
    return None

def _greedy_local_search(h, max_iters=5000, time_limit=15.0):
    """
    Fast greedy local search with multiple move types:
    1. Flip individual bits
    2. Zero out bits
    3. Activate zero bits
    4. Pairwise swaps
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    def _fast_score(h):
        """Fast score computation using precomputed FFT size."""
        H = np.fft.rfft(h, n=fft_size)
        acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
        return float(np.max(np.abs(acf[1:n])))
    
    best_score = _fast_score(best_h)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        improved = False
        
        # Strategy 1: Flip signs of non-zero entries
        nonzero_idx = np.nonzero(best_h)[0]
        if len(nonzero_idx) > 0:
            # Try a random subset of flips
            n_tries = min(20, len(nonzero_idx))
            trial_indices = np.random.choice(nonzero_idx, size=n_tries, replace=False)
            for i in trial_indices:
                trial = best_h.copy()
                trial[i] = -trial[i]
                s = _fast_score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
        
        # Strategy 2: Zero out entries
        nonzero_idx = np.nonzero(best_h)[0]
        if len(nonzero_idx) > 0 and not improved:
            n_tries = min(10, len(nonzero_idx))
            trial_indices = np.random.choice(nonzero_idx, size=n_tries, replace=False)
            for i in trial_indices:
                trial = best_h.copy()
                trial[i] = 0.0
                s = _fast_score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
        
        # Strategy 3: Activate zero entries
        zero_idx = np.nonzero(best_h == 0)[0]
        if len(zero_idx) > 0 and not improved:
            n_tries = min(10, len(zero_idx))
            trial_indices = np.random.choice(zero_idx, size=n_tries, replace=False)
            for i in trial_indices:
                for sign in [1.0, -1.0]:
                    trial = best_h.copy()
                    trial[i] = sign
                    s = _fast_score(trial)
                    if s < best_score - 1e-12:
                        best_h = trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
        
        # Strategy 4: Pairwise sign flips
        if len(nonzero_idx) >= 2 and not improved:
            i, j = np.random.choice(nonzero_idx, size=2, replace=False)
            if i != j:
                trial = best_h.copy()
                trial[i] = -trial[i]
                trial[j] = -trial[j]
                s = _fast_score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
        
        # Strategy 5: Simulated annealing-like random perturbation
        if not improved and it % 50 == 0:
            n_perturb = np.random.randint(1, 5)
            perturbed = best_h.copy()
            if len(nonzero_idx) > 0:
                indices = np.random.choice(nonzero_idx, size=min(n_perturb, len(nonzero_idx)), replace=False)
                for i in indices:
                    perturbed[i] = -perturbed[i]
            s = _fast_score(perturbed)
            # Accept with probability based on "temperature"
            temp = 0.5 * (time_limit - (time.time() - start)) / time_limit
            if s < best_score or np.random.random() < np.exp(-(s - best_score) / (temp + 1e-10)):
                best_h = perturbed
                best_score = s
                improved = True
        
        if not improved and it > 100:
            # Check for stagnation
            if it % 100 == 0:
                break
    
    return best_h, best_score

def _construct_barker_like():
    """
    Construct a Barker-like sequence with good autocorrelation properties.
    Barker sequences are known to have optimal autocorrelation for short lengths.
    For longer sequences, we use a recursive construction.
    """
    # Barker sequence of length 13: + + + - + - - + - + - + -
    barker_13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1])
    
    # Repeat and modify to get length 400
    h = np.zeros(N)
    for i in range(N):
        h[i] = barker_13[i % 13]
    
    # Add some randomness to break periodicity
    np.random.seed(42)
    for _ in range(10):
        idx = np.random.randint(0, N)
        h[idx] = -h[idx]
    
    return h

def _construct_random_binary():
    """Generate a random binary sequence."""
    return np.random.choice([-1.0, 1.0], size=N)

def _construct_sparse(k, seed=None):
    """Generate a sparse sequence with k non-zero entries."""
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(N, size=k, replace=False)
    h = np.zeros(N)
    h[indices] = np.random.choice([-1.0, 1.0], size=k)
    return h

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound (max absolute autocorrelation at non-zero lags).
    
    Strategy:
    1. Try multiple initial constructions (Barker-like, random binary, sparse)
    2. Optimize each via fast greedy local search
    3. Return the best solution found within time budget
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try different initial constructions
    initializations = [
        ("barker", _construct_barker_like()),
        ("random_binary", _construct_random_binary()),
    ]
    
    # Try different sparse densities
    for k in [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300]:
        initializations.append((f"sparse_{k}", _construct_sparse(k)))
    
    # Optimize each initialization
    for name, h_init in initializations:
        if time.time() > deadline - 15:
            break
        
        h_opt, score = _greedy_local_search(h_init, max_iters=3000, time_limit=8.0)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Final polish on best solution
    if time.time() < deadline - 10:
        best_h, best_score = _greedy_local_search(best_h, max_iters=5000, time_limit=15.0)
    
    return best_h
# EVOLVE-BLOCK-END
