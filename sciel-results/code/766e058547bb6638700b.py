"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _autocorr(x):
    """Linear autocorrelation via FFT. Returns array of length 2N-1."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize <<= 1
    fx = np.fft.rfft(x, fsize)
    ac = np.fft.irfft(fx * np.conj(fx), fsize)[:size]
    return ac

def _score(x):
    """Compute max|AC| / (sum)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autocorr(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _delta_ac_flip(x, i, old_val):
    """Compute the change in autocorrelation when flipping x[i] from old_val to -old_val.
    Returns the delta array of length 2N-1."""
    n = len(x)
    delta = np.zeros(2 * n - 1)
    new_val = -old_val
    diff = new_val - old_val  # = -2*old_val
    for k in range(2 * n - 1):
        # AC[k] = sum_j x[j]*x[j+k] for valid j
        # Flipping x[i] affects terms where j=i or j+k=i
        # Term where j=i: x[i]*x[i+k] changes by diff*x[i+k]
        # Term where j+k=i: x[i-k]*x[i] changes by x[i-k]*diff
        val = 0.0
        if 0 <= i + k < n:
            val += diff * x[i + k]
        if 0 <= i - k < n:
            val += x[i - k] * diff
        delta[k] = val
    return delta

def _legendre_sequence(n):
    """Generate a Legendre symbol sequence for a prime p, padded/truncated to n."""
    # Find smallest prime >= n
    def is_prime(m):
        if m < 2:
            return False
        if m == 2:
            return True
        if m % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(m)) + 1, 2):
            if m % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Legendre symbol: (a/p) = a^((p-1)/2) mod p
    x = np.ones(n)
    for i in range(1, n):
        # Legendre symbol of i mod p
        ls = pow(i, (p - 1) // 2, p)
        if ls == p - 1:
            x[i] = -1
    return x

def _local_search_fast(x, time_limit, rng_seed=123):
    """Local search with delta autoconv updates for speed."""
    start_time = time.time()
    current_x = x.copy().astype(np.float64)
    current_ac = _autocorr(current_x)
    current_max_ac = np.max(np.abs(current_ac))
    current_sum = np.sum(current_x)
    current_score = current_max_ac / (current_sum * current_sum)
    
    best_x = current_x.copy()
    best_score = current_score
    best_ac = current_ac.copy()
    best_sum = current_sum
    
    n = len(current_x)
    rng = np.random.RandomState(rng_seed)
    
    # Precompute which lags are affected by flipping position i
    # For flip at i, delta at lag k is: -2*x[i]*(x[i+k] + x[i-k]) where valid
    # We'll compute delta on the fly but use the formula
    
    # Track which lags have the maximum to quickly check if max changes
    max_lags = set(np.where(np.abs(current_ac) == current_max_ac)[0])
    
    max_iters = 200000
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(n)
        old_val = current_x[i]
        diff = -2.0 * old_val  # new - old
        
        # Compute delta AC
        delta = np.zeros(2 * n - 1)
        for k in range(2 * n - 1):
            val = 0.0
            if 0 <= i + k < n:
                val += diff * current_x[i + k]
            if 0 <= i - k < n:
                val += current_x[i - k] * diff
            delta[k] = val
        
        new_ac = current_ac + delta
        new_sum = current_sum + diff
        new_max_ac = np.max(np.abs(new_ac))
        
        if abs(new_sum) < 1e-12:
            new_score = np.inf
        else:
            new_score = new_max_ac / (new_sum * new_sum)
        
        if new_score < current_score:
            # Accept
            current_x[i] = -old_val
            current_ac = new_ac
            current_sum = new_sum
            current_max_ac = new_max_ac
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
                best_ac = new_ac.copy()
                best_sum = new_sum
        else:
            # Occasionally accept worse (simulated annealing)
            if rng.random() < 0.03:
                current_x[i] = -old_val
                current_ac = new_ac
                current_sum = new_sum
                current_max_ac = new_max_ac
                current_score = new_score
    
    return best_x, best_score

def _greedy_construct(n, time_limit):
    """Greedy construction: build sequence position by position."""
    start_time = time.time()
    x = np.zeros(n)
    
    # Start with a balanced base
    for i in range(n):
        if time.time() - start_time > time_limit:
            break
        # Try both signs, pick the one that gives lower score
        x[i] = 1.0
        s = np.sum(x)
        if abs(s) > 1e-12:
            ac = _autocorr(x)
            score_plus = np.max(np.abs(ac)) / (s * s)
        else:
            score_plus = np.inf
        
        x[i] = -1.0
        s = np.sum(x)
        if abs(s) > 1e-12:
            ac = _autocorr(x)
            score_minus = np.max(np.abs(ac)) / (s * s)
        else:
            score_minus = np.inf
        
        if score_plus <= score_minus:
            x[i] = 1.0
        else:
            x[i] = -1.0
    
    return x

def construct():
    deadline = time.time() + 130  # 130s safety margin (150s kill)
    
    best_x = None
    best_score = np.inf
    
    # Candidate 1: Legendre-based
    x1 = _legendre_sequence(N)
    s1 = _score(x1)
    if s1 < best_score:
        best_score = s1
        best_x = x1.copy()
    
    # Candidate 2: All ones (baseline)
    x2 = np.ones(N)
    s2 = _score(x2)
    if s2 < best_score:
        best_score = s2
        best_x = x2.copy()
    
    # Candidate 3: Balanced random
    rng = np.random.RandomState(42)
    x3 = rng.choice([1, -1], size=N)
    s3 = _score(x3)
    if s3 < best_score:
        best_score = s3
        best_x = x3.copy()
    
    # Candidate 4: Block pattern
    x4 = np.ones(N)
    block = 8
    for b in range(0, N, 2 * block):
        end = min(b + block, N)
        x4[b:end] = -1
    s4 = _score(x4)
    if s4 < best_score:
        best_score = s4
        best_x = x4.copy()
    
    # Candidate 5: Chirp sign
    x5 = np.ones(N)
    for i in range(N):
        if (i * i) % N < N // 4:
            x5[i] = -1
    s5 = _score(x5)
    if s5 < best_score:
        best_score = s5
        best_x = x5.copy()
    
    # Run local search on best candidate
    remaining = deadline - time.time()
    if remaining > 10:
        best_x, best_score = _local_search_fast(best_x, remaining - 10, rng_seed=123)
    
    # Try local search on other promising candidates
    remaining = deadline - time.time()
    if remaining > 20:
        # Try Legendre-based local search
        x_leg = _legendre_sequence(N)
        x_leg, s_leg = _local_search_fast(x_leg, min(remaining - 10, 50), rng_seed=456)
        if s_leg < best_score:
            best_score = s_leg
            best_x = x_leg.copy()
    
    # Another restart with different random seed
    remaining = deadline - time.time()
    if remaining > 15:
        rng2 = np.random.RandomState(789)
        x_rand = rng2.choice([1, -1], size=N)
        # Bias toward +1 to increase sum
        # Keep roughly 60% +1
        flip_idx = rng2.choice(N, size=int(N * 0.2), replace=False)
        x_rand[flip_idx] = -1
        x_rand, s_rand = _local_search_fast(x_rand, min(remaining - 5, 40), rng_seed=789)
        if s_rand < best_score:
            best_score = s_rand
            best_x = x_rand.copy()
    
    # Final polish on best
    remaining = deadline - time.time()
    if remaining > 5:
        best_x, best_score = _local_search_fast(best_x, remaining - 3, rng_seed=321)
    
    return best_x
# EVOLVE-BLOCK-END
