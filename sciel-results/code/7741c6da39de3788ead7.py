"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_bound(s):
    """
    Compute the upper bound C3 for a signed step function s of length N.
    The autoconvolution c[k] = sum_i s[i]*s[i+k] for k in 0..N-1.
    C3 is typically related to the maximum absolute value of the autoconvolution
    excluding the trivial peak at k=0, or a specific norm.
    
    Based on the task description "autoconvolution upper bound" and "lower is better",
    and the fact that all-ones gives 1.0 (seed), let's deduce the metric.
    
    For s = ones(N):
    Autoconvolution c[k] = N - |k| for k in 0..N-1.
    Max |c[k]| = N = 400.
    If the score is 1.0 for the seed, and the score is seed/value, then value for seed must be such that seed/value = 1.0?
    No, "score is the upper bound on C3... The evaluator measures the raw value (lower is better). The search score is seed/value".
    Wait, "score is seed/value, so 1.0 equals the seed program".
    This implies: Score = Value_seed / Value_candidate.
    If Score = 1.0, then Value_candidate = Value_seed.
    
    What is Value_seed?
    If the task asks for an "upper bound on C3", usually C3 refers to the third moment or a specific autocorrelation bound in discrepancy theory or additive combinatorics.
    However, often in these "signed step function" problems, the objective is to minimize the L-infinity norm of the autocorrelation (excluding lag 0) or the sum of squares of the autocorrelation.
    
    Let's look at the standard "Third autocorrelation inequality" context.
    This likely refers to the problem of minimizing the maximum autocorrelation sidelobe.
    Let's assume the "value" is the maximum absolute autocorrelation sidelobe (lag != 0).
    For s = ones(N):
    c[0] = N
    c[k] = N - |k|
    Max sidelobe = N-1 = 399.
    
    Is it possible the value is normalized?
    If value = max_{k!=0} |c[k]| / N, then for seed, value = 399/400 = 0.9975.
    Then Score = 0.9975 / 0.9975 = 1.0.
    
    Alternatively, maybe the value is just the max sidelobe?
    If value = 399, and seed value = 399, Score = 1.0.
    
    Let's check the "seed/value" phrasing.
    "The search score is seed/value".
    If I find a sequence with max sidelobe 100, and seed has 399, Score = 399/100 = 3.99. This is an improvement.
    
    Let's assume the metric is:
    Value(s) = max_{k != 0} |sum_{i} s[i] s[i+k]|
    
    We want to minimize this value.
    
    Let's verify with the "Zadoff-Chu" inspiration.
    Zadoff-Chu sequences have constant amplitude and zero autocorrelation (cyclic) or low autocorrelation (linear).
    For a Zadoff-Chu sequence of length N (if N is prime or suitable), the cyclic autocorrelation is 0 for all non-zero shifts.
    The linear autocorrelation sidelobes are small.
    
    So the strategy is to construct a sequence with low linear autocorrelation sidelobes.
    
    The prompt mentions: "Formulate the signed step function as the sign of the imaginary part of the product of two independent quadratic phase chirps...".
    This is a specific construction.
    
    Let's implement a search algorithm that minimizes the max linear autocorrelation sidelobe.
    
    Algorithm:
    1. Start with a good initial guess (e.g., random +/- 1, or a chirp).
    2. Use local search (flipping signs) to reduce the max sidelobe.
    3. Use a deadline guard.
    
    To speed up the evaluation of the objective function:
    Use FFT to compute the autocorrelation.
    c = ifft(fft(s) * conj(fft(s)))
    This gives the circular autocorrelation.
    We need linear autocorrelation.
    Zero-pad s to length 2N.
    c = ifft(fft(s, 2N) * conj(fft(s, 2N)))
    The first N elements of c are the linear autocorrelation for lags 0 to N-1.
    We ignore c[0] (which is N) and look at c[1:N].
    Value = max(abs(c[1:N])).
    
    Local Search:
    Iterate over indices i.
    Flip s[i].
    Compute new Value.
    If Value < best_Value, keep flip.
    Else, revert.
    
    This is O(N^2 log N) per full pass if we recompute FFT every time.
    N=400. FFT is fast.
    But flipping one bit changes the entire FFT.
    Can we update the autocorrelation efficiently?
    c[k] = sum s[i] s[i+k].
    Changing s[i] from 1 to -1 changes c[k] by:
    -2 * s[i+k] for k such that i+k is in range (forward lags)
    -2 * s[i-k] for k such that i-k is in range (backward lags)
    Actually, it's easier to just recompute the FFT for N=400.
    400 points FFT is very fast.
    Let's estimate time.
    400 flips * 400 steps = 160,000 evaluations.
    Each evaluation: FFT(400) -> ~microseconds.
    Total time < 1 second.
    So we can do many passes.
    
    Let's refine the local search:
    - Start with a random sequence or a chirp.
    - Perform multiple passes of flipping bits that reduce the max sidelobe.
    - If no improvement in a pass, try a different initialization or a steeper descent (e.g., flip the bit that causes the largest reduction in the peak).
    
    Better heuristic:
    Identify the lag k* where the max sidelobe occurs.
    Find the index i that, if flipped, most reduces |c[k*]|.
    Flip it.
    Repeat.
    
    This is "greedy descent on the peak".
    
    Let's implement this.
    
    Steps:
    1. Initialize s.
    2. Compute c.
    3. Find k_peak = argmax_{k!=0} |c[k]|.
    4. For each i in 0..N-1:
       - Calculate delta_c for lag k_peak if s[i] is flipped.
       - delta_c[k] = -2 * s[i+k] (if i+k < N) - 2 * s[i-k] (if i-k >= 0) ... wait, signs.
       c_new[k] = c[k] - 2*s[i+k]*s[i] - 2*s[i-k]*s[i] ?
       Let's derive:
       c[k] = sum_j s[j] s[j+k]
       If we flip s[i] to -s[i]:
       Terms involving s[i] are:
       1. j=i: s[i] s[i+k] becomes -s[i] s[i+k]. Change: -2 s[i] s[i+k].
       2. j=i-k: s[i-k] s[i] becomes s[i-k] (-s[i]). Change: -2 s[i-k] s[i].
       So delta = -2 * s[i] * (s[i+k] + s[i-k]) (with boundary checks).
       
       We want to minimize |c[k_peak] + delta|.
       Choose i that minimizes this.
       Flip that i.
       Update c.
       Repeat until no improvement or time limit.
       
    This is O(N) per step, and we do O(N) steps per "epoch".
    Total O(N^2) per epoch.
    N=400, N^2=160,000 operations. Very fast.
    We can do many epochs.
    
    Let's add a deadline check.
    
    Initializations to try:
    1. All ones (seed) - just to be safe.
    2. Random.
    3. Chirp (Zadoff-Chu like).
    
    The prompt suggests: "sign of the imaginary part of the product of two independent quadratic phase chirps".
    Let's implement that as a specific initialization.
    
    Chirp 1: exp(2j * pi * n^2 / p1)
    Chirp 2: exp(2j * pi * n^2 / p2)
    Product: exp(2j * pi * n^2 * (1/p1 + 1/p2))
    Imaginary part sign.
    
    Let's pick primes p1, p2 > N.
    
    Let's code this up.
"""

import numpy as np
import time

N = 400

def _compute_c3_bound(s):
    """Compute the max absolute linear autocorrelation sidelobe."""
    # Zero pad to 2N for linear convolution via FFT
    f = np.fft.fft(s, 2*N)
    c = np.fft.ifft(f * np.conj(f))[:N].real
    # c[0] is N, ignore it
    return np.max(np.abs(c[1:]))

def _construct_chirp_product(p1, p2):
    """Construct sequence based on product of two chirps."""
    n = np.arange(N)
    phase1 = 2 * np.pi * (n**2) / p1
    phase2 = 2 * np.pi * (n**2) / p2
    z = np.exp(1j * (phase1 + phase2))
    s = np.sign(z.imag)
    # Ensure no zeros (if any, set to 1)
    s[s == 0] = 1
    return s.astype(np.float64)

def _local_search_greedy(s, time_limit=10.0):
    """
    Greedy local search flipping bits to reduce the peak autocorrelation.
    """
    start_time = time.time()
    s = s.copy()
    
    # Precompute FFT? No, we update c incrementally in the inner loop logic?
    # Actually, calculating delta is O(1) per i if we have c.
    # But we need c to find the peak.
    # So:
    # 1. Compute c from s.
    # 2. Find peak lag k*.
    # 3. For each i, compute delta for c[k*].
    # 4. Pick best i, flip, update c.
    
    c = np.fft.ifft(np.fft.fft(s, 2*N) * np.conj(np.fft.fft(s, 2*N)))[:N].real
    best_val = np.max(np.abs(c[1:]))
    
    # To speed up, we can cache the FFT? No, s changes.
    # But we can update c directly?
    # c[k] = sum s[j] s[j+k]
    # If we flip s[i], c changes.
    # Let's just recompute c after each flip?
    # N=400. FFT is ~10us. 400 flips = 4ms.
    # 10 seconds allows ~2500 full passes.
    # This is plenty.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Compute current peak
        c = np.fft.ifft(np.fft.fft(s, 2*N) * np.conj(np.fft.fft(s, 2*N)))[:N].real
        vals = np.abs(c[1:])
        if len(vals) == 0:
            break
        k_peak = np.argmax(vals) + 1
        peak_val = vals[k_peak-1]
        
        # Find best flip
        best_i = -1
        best_new_val = peak_val
        
        for i in range(N):
            # Calculate change in c[k_peak]
            # delta = -2 * s[i] * (s[i+k_peak] + s[i-k_peak])
            # with bounds
            term1 = 0.0
            term2 = 0.0
            if i + k_peak < N:
                term1 = s[i + k_peak]
            if i - k_peak >= 0:
                term2 = s[i - k_peak]
            
            delta = -2.0 * s[i] * (term1 + term2)
            new_val = abs(c[k_peak] + delta)
            
            if new_val < best_new_val:
                best_new_val = new_val
                best_i = i
        
        if best_i != -1 and best_new_val < peak_val:
            s[best_i] *= -1
            improved = True
            # We could update c incrementally here, but recomputing is safer and fast enough
            # Let's recompute at the start of the next loop iteration
    
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_s = None
    best_score = np.inf
    
    # Strategy 1: Chirp Product Initializations
    primes = [401, 409, 419, 421, 431, 433, 439, 443, 449, 457]
    for p1 in primes[:5]:
        for p2 in primes[5:10]:
            if (time.time() - start_time) > deadline:
                break
            s_init = _construct_chirp_product(p1, p2)
            s_opt = _local_search_greedy(s_init, time_limit=min(10.0, deadline - time.time()))
            score = _compute_c3_bound(s_opt)
            if score < best_score:
                best_score = score
                best_s = s_opt
        
        if (time.time() - start_time) > deadline:
            break
    
    # Strategy 2: Random Initializations
    np.random.seed(42)
    for _ in range(10):
        if (time.time() - start_time) > deadline:
            break
        s_init = np.random.choice([-1.0, 1.0], size=N)
        s_opt = _local_search_greedy(s_init, time_limit=min(10.0, deadline - time.time()))
        score = _compute_c3_bound(s_opt)
        if score < best_score:
            best_score = score
            best_s = s_opt
            
    # Strategy 3: All Ones (Seed) - just in case
    if (time.time() - start_time) < deadline:
        s_seed = np.ones(N)
        s_opt = _local_search_greedy(s_seed, time_limit=min(10.0, deadline - time.time()))
        score = _compute_c3_bound(s_opt)
        if score < best_score:
            best_score = score
            best_s = s_opt

    if best_s is None:
        return np.ones(N)
        
    return best_s

# EVOLVE-BLOCK-END
