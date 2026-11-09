"""
Third autocorrelation inequality: minimize max |R[k]| for k>=1
over signed step functions h of length N=400 (values in {-1, +1}).

Strategy:
1. Generate diverse initial candidates via:
   a) Random sequences
   b) Sinusoidal sign patterns with random frequencies/phases
   c) Walsh-Hadamard based patterns
2. Polish each candidate with:
   a) Fast greedy single-flip local search (vectorized delta computation)
   b) Multi-flip (pair) local search for escaping local minima
   c) Simulated annealing with adaptive temperature
3. Return the best sequence found within the time budget.
"""
import numpy as np
import time
import sys

N = 400
DEADLINE = 145.0  # safety margin: evaluator kills at 150s


def _autocorr(h):
    """Compute autocorrelation R[0..N-1] where R[k] = sum_n h[n]*h[n+k]."""
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return acf


def _max_sidelobe(R):
    """Max |R[k]| for k >= 1."""
    return np.max(np.abs(R[1:]))


def _flip_delta(h, i):
    """
    Compute the change in autocorrelation when flipping h[i].
    Returns delta array of length N such that new_R = R - delta.
    """
    delta = np.zeros(N, dtype=np.float64)
    # Flipping h[i] changes R[k] for k where the pair (i, i+k) or (i-k, i) is involved
    # R[k] = sum_n h[n]*h[n+k]
    # After flip: h[i] -> -h[i]
    # Change in R[k]: -2*h[i]*h[i+k] for k>0 (pair i, i+k)
    #                 -2*h[i]*h[i-k] for k>0 (pair i-k, i)  [when i-k >= 0]
    #                 -2*h[i]*h[i+k] for k>0 [when i+k < N]
    
    # For k in 1..N-1:
    # R_new[k] = R[k] - 2*h[i]*h[i+k] (if i+k < N) - 2*h[i]*h[i-k] (if i-k >= 0)
    # But note: when k is such that both i+k and i-k are valid, both terms contribute.
    # Actually R[k] = sum_{n=0}^{N-1-k} h[n]*h[n+k]
    # The term involving h[i] appears when n=i (if i <= N-1-k) or n+k=i (if k <= i <= N-1)
    # So: contribution = h[i]*h[i+k] (if i+k < N) + h[i-k]*h[i] (if i-k >= 0)
    # After flip: -h[i]*h[i+k] + h[i-k]*(-h[i]) = -2*h[i]*h[i+k] - 2*h[i]*h[i-k] ... wait
    # Let me redo:
    # R[k] = sum_n h[n]*h[n+k]
    # Terms with h[i]: n=i gives h[i]*h[i+k] (valid if i+k<N)
    #                n+k=i gives h[i-k]*h[i] (valid if i-k>=0, i.e., k<=i)
    # After flipping h[i] to -h[i]:
    #   n=i: (-h[i])*h[i+k] = -h[i]*h[i+k]
    #   n+k=i: h[i-k]*(-h[i]) = -h[i-k]*h[i]
    # Change = (-h[i]*h[i+k] - h[i-k]*h[i]) - (h[i]*h[i+k] + h[i-k]*h[i])
    #        = -2*h[i]*h[i+k] - 2*h[i-k]*h[i]
    # So delta[k] = 2*h[i]*(h[i+k] + h[i-k]) for valid indices, 0 otherwise.
    
    # Vectorized:
    # For k=1..N-1:
    #   term1 = h[i+k] if i+k < N else 0
    #   term2 = h[i-k] if i-k >= 0 else 0
    #   delta[k] = 2*h[i]*(term1 + term2)
    
    k = np.arange(1, N)
    # term1: h[i+k] for k where i+k < N
    valid1 = i + k < N
    term1 = np.zeros(N-1)
    term1[valid1] = h[i + k[valid1]]
    
    # term2: h[i-k] for k where i-k >= 0
    valid2 = i - k >= 0
    term2 = np.zeros(N-1)
    term2[valid2] = h[i - k[valid2]]
    
    delta[1:] = 2.0 * h[i] * (term1 + term2)
    return delta


def _greedy_local_search(h, R, max_sidelobe, time_limit, max_rounds=50):
    """
    Greedy single-flip local search.
    Returns (h, R, max_sidelobe).
    """
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < time_limit:
        improved = False
        rounds += 1
        # Compute all deltas at once for speed
        # For each i, delta_i[k] = 2*h[i]*(h[i+k]+h[i-k])
        # new_R_i = R - delta_i
        # We want to find i that minimizes max_sidelobe(R - delta_i)
        
        # Vectorized approach: compute delta matrix D of shape (N, N)
        # D[i, k] = 2*h[i]*(h[i+k]+h[i-k]) for k=1..N-1
        # This is expensive for N=400 (400x400 = 160K entries), but doable.
        
        # Actually, let's do it more efficiently:
        # For each i, the delta only affects R[1:N]
        # new_sidelobe_max_i = max_k |R[k] - D[i,k]|
        
        # Build D efficiently using the structure:
        # D[i, k] = 2*h[i]*(h[i+k] + h[i-k])
        # This is related to the autocorrelation of h with a single flip.
        
        # Let's just iterate but with early termination
        best_i = -1
        best_new_max = max_sidelobe
        R_sidelobe = R[1:]
        
        for i in range(N):
            if time.time() > time_limit:
                break
            
            # Compute delta for this i
            k = np.arange(1, N)
            valid1 = i + k < N
            valid2 = i - k >= 0
            
            term = np.zeros(N-1)
            term[valid1] += h[i + k[valid1]]
            term[valid2] += h[i - k[valid2]]
            
            delta = 2.0 * h[i] * term
            new_sidelobe = R_sidelobe - delta
            new_max = np.max(np.abs(new_sidelobe))
            
            if new_max < best_new_max - 1e-12:
                best_new_max = new_max
                best_i = i
        
        if best_i >= 0:
            h[best_i] *= -1.0
            # Update R
            k = np.arange(1, N)
            valid1 = best_i + k < N
            valid2 = best_i - k >= 0
            term = np.zeros(N-1)
            term[valid1] += h[best_i + k[valid1]]
            term[valid2] += h[best_i - k[valid2]]
            # Note: h[best_i] has already been flipped, so we need to use the old value
            # Actually we should compute delta BEFORE flipping
            # Let me redo: compute delta using old h, then flip
            # But we already flipped. Let's recompute R from scratch for safety? No, too slow.
            # Better: recompute delta properly.
            
            # Actually, let me restructure: compute delta before flipping.
            # I'll fix this by computing delta first.
            
            # For now, recompute R (N=400, FFT is fast ~microseconds)
            R = _autocorr(h)
            max_sidelobe = _max_sidelobe(R)
            improved = True
    
    return h, R, max_sidelobe


def _pair_local_search(h, R, max_sidelobe, time_limit):
    """
    Try flipping pairs of positions to escape local minima.
    """
    improved = True
    R_sidelobe = R[1:]
    
    while improved and time.time() < time_limit:
        improved = False
        # Sample a few random pairs to try
        for _ in range(200):
            if time.time() > time_limit:
                break
            i, j = np.random.choice(N, size=2, replace=False)
            if i > j:
                i, j = j, i
            
            # Compute delta for flipping both i and j
            # delta_ij = delta_i + delta_j - correction (since flipping i changes h[i] which affects delta_j)
            # Simpler: compute new R by flipping both
            h[j] *= -1.0
            h[i] *= -1.0
            new_R = _autocorr(h)
            new_max = _max_sidelobe(new_R)
            
            if new_max < max_sidelobe - 1e-12:
                R = new_R
                max_sidelobe = new_max
                improved = True
            else:
                # Undo
                h[j] *= -1.0
                h[i] *= -1.0
    
    return h, R, max_sidelobe


def _simulated_annealing(h, R, max_sidelobe, time_limit, T0=1.0, T1=0.01, alpha=0.999):
    """
    Simulated annealing with single-flip moves.
    """
    T = T0
    R_sidelobe = R[1:]
    
    while T > T1 and time.time() < time_limit:
        i = np.random.randint(N)
        
        # Compute delta
        k = np.arange(1, N)
        valid1 = i + k < N
        valid2 = i - k >= 0
        term = np.zeros(N-1)
        term[valid1] += h[i + k[valid1]]
        term[valid2] += h[i - k[valid2]]
        delta = 2.0 * h[i] * term
        new_sidelobe = R_sidelobe - delta
        new_max = np.max(np.abs(new_sidelobe))
        delta_obj = new_max - max_sidelobe
        
        if delta_obj < 0 or np.random.random() < np.exp(-delta_obj / T):
            h[i] *= -1.0
            R = _autocorr(h)
            max_sidelobe = new_max
            R_sidelobe = R[1:]
        
        T *= alpha
    
    return h, R, max_sidelobe


def _random_candidate(seed):
    """Generate a random signed sequence."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)


def _sinusoidal_candidate(seed, K=8):
    """Generate a candidate from sign of sum of K sinusoids."""
    rng = np.random.RandomState(seed)
    n = np.arange(N)
    x = np.zeros(N)
    for k in range(K):
        freq = rng.randint(1, N // 2)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.5)
        x += amp * np.cos(2 * np.pi * freq * n / N + phase)
    h = np.sign(x)
    h[h == 0] = 1.0
    return h


def _hadamard_candidate(seed):
    """Generate a candidate from a Walsh-Hadamard-like pattern."""
    rng = np.random.RandomState(seed)
    # Random projection onto low-frequency Walsh functions
    # Simple approach: use a random linear combination of first M Walsh functions
    M = 16
    n = np.arange(N)
    coeffs = rng.randn(M)
    x = np.zeros(N)
    for m in range(M):
        # Rademacher function: (-1)^(floor(n/2^m) mod 2)
        x += coeffs[m] * np.where((n >> m) % 2 == 0, 1.0, -1.0)
    h = np.sign(x)
    h[h == 0] = 1.0
    return h


def construct():
    """
    Construct a signed step function with N=400 intervals to minimize
    the maximum autocorrelation sidelobe (upper bound on C3).
    """
    start_time = time.time()
    best_h = None
    best_val = np.inf
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for s in range(5):
        candidates.append(_random_candidate(s))
    
    # Sinusoidal with different K
    for K in [4, 8, 16, 32]:
        for s in range(3):
            candidates.append(_sinusoidal_candidate(s + 100 * K, K))
    
    # Hadamard-based
    for s in range(5):
        candidates.append(_hadamard_candidate(s + 500))
    
    # Evaluate and polish each candidate
    time_per_candidate = 3.0  # seconds per candidate
    
    for idx, h in enumerate(candidates):
        if time.time() > DEADLINE - 10.0:
            break
        
        R = _autocorr(h)
        cur_max = _max_sidelobe(R)
        
        if cur_max < best_val:
            best_val = cur_max
            best_h = h.copy()
        
        # Greedy local search
        t_limit = min(time.time() + time_per_candidate * 0.5, DEADLINE - 15.0)
        h, R, cur_max = _greedy_local_search(h, R, cur_max, t_limit, max_rounds=30)
        
        if cur_max < best_val:
            best_val = cur_max
            best_h = h.copy()
        
        # Pair local search
        t_limit = min(time.time() + time_per_candidate * 0.25, DEADLINE - 15.0)
        h, R, cur_max = _pair_local_search(h, R, cur_max, t_limit)
        
        if cur_max < best_val:
            best_val = cur_max
            best_h = h.copy()
        
        # Simulated annealing
        t_limit = min(time.time() + time_per_candidate * 0.25, DEADLINE - 15.0)
        h, R, cur_max = _simulated_annealing(h, R, cur_max, t_limit, T0=0.5, T1=0.01, alpha=0.9995)
        
        if cur_max < best_val:
            best_val = cur_max
            best_h = h.copy()
        
        # Final greedy polish
        t_limit = min(time.time() + 1.0, DEADLINE - 10.0)
        h, R, cur_max = _greedy_local_search(h, R, cur_max, t_limit, max_rounds=20)
        
        if cur_max < best_val:
            best_val = cur_max
            best_h = h.copy()
    
    # If time remains, do more rounds of SA + greedy on the best solution
    if best_h is not None and time.time() < DEADLINE - 5.0:
        R = _autocorr(best_h)
        cur_max = _max_sidelobe(R)
        
        while time.time() < DEADLINE - 5.0:
            # SA round
            t_limit = min(time.time() + 5.0, DEADLINE - 10.0)
            best_h, R, cur_max = _simulated_annealing(
                best_h.copy(), R.copy(), cur_max, t_limit, T0=0.3, T1=0.01, alpha=0.9998
            )
            
            # Greedy polish
            t_limit = min(time.time() + 3.0, DEADLINE - 5.0)
            best_h, R, cur_max = _greedy_local_search(best_h, R, cur_max, t_limit, max_rounds=30)
            
            # Pair search
            t_limit = min(time.time() + 2.0, DEADLINE - 5.0)
            best_h, R, cur_max = _pair_local_search(best_h, R, cur_max, t_limit)
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h


if __name__ == "__main__":
    h = construct()
    R = _autocorr(h)
    print(f"Max sidelobe: {_max_sidelobe(R)}")
    print(f"Length: {len(h)}")
    print(f"Values in {-1,1}: {set(h.tolist()) <= {-1.0, 1.0}}")
