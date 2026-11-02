import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1] using FFT."""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=_FFT_SIZE)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_FFT_SIZE)
    return np.real(acf[:N])

def _hard_score(s):
    """Calculate the hard peak sidelobe magnitude."""
    A = _calc_acf(s)
    return float(np.max(np.abs(A[1:])))

def _soft_score(s, beta=10.0):
    """Calculate a smooth surrogate for the peak sidelobe using log-sum-exp."""
    A = _calc_acf(s)
    abs_A = np.abs(A[1:])
    if len(abs_A) == 0:
        return 0.0
    v = abs_A ** 2
    v_max = np.max(v)
    if not np.isfinite(v_max):
        return 0.0
    shifted = v - v_max
    shifted = np.clip(shifted, -50, 50)
    exp_sum = np.sum(np.exp(beta * shifted))
    if exp_sum <= 0:
        return np.sqrt(v_max)
    soft_v = v_max + (1.0 / beta) * np.log(exp_sum)
    return float(np.sqrt(soft_v))

def _vdc_init(seed=0):
    """Generate a +-1 sequence based on the van der Corput sequence."""
    vdc_vals = np.zeros(N)
    for i in range(N):
        val = 0.0
        div = 1
        n = i
        while n > 0:
            div *= 2
            if n % 2 == 1:
                val += 1.0 / div
            n //= 2
        vdc_vals[i] = val
    shift = seed % N
    vdc_vals = np.roll(vdc_vals, shift)
    h = np.where(vdc_vals < 0.5, -1.0, 1.0)
    return h

def _sine_init(freq, phase, seed):
    """Generate a +-1 sequence from zero crossings of a sine wave with noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi * freq, N)
    signal = np.sin(x + phase)
    noise = rng.normal(0, 0.1, size=N)
    h = np.sign(signal + noise)
    h[h == 0] = 1.0
    return h

def _m_sequence_init(p, seed=0):
    """Generate an m-sequence like pattern (not strictly m-seq for N=400, but structured)."""
    rng = np.random.default_rng(seed)
    # Use a pseudo-random number generator with a specific period to create structure
    # This is a placeholder for more structured inits if needed
    return rng.choice([-1, 1], size=N)

def _simulated_annealing(s_init, time_budget, rng):
    """SA with adaptive block sizes and soft objective."""
    s = s_init.copy()
    current_score = _soft_score(s, beta=10.0)
    best_s = s.copy()
    best_score = current_score
    
    T = 100.0
    T_min = 0.01
    cooling = 0.998
    start_time = time.time()
    
    block_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    while time.time() < start_time + time_budget and T > T_min:
        # Adaptive block size: larger blocks at higher T
        if T > 10.0:
            block_size = block_sizes[rng.randint(2, len(block_sizes))]
        elif T > 1.0:
            block_size = block_sizes[rng.randint(0, 4)]
        else:
            block_size = 1
            
        start_idx = rng.randint(0, N - block_size + 1)
        
        old_block = s[start_idx:start_idx + block_size].copy()
        s[start_idx:start_idx + block_size] = -old_block
        
        new_score = _soft_score(s, beta=10.0)
        delta = new_score - current_score
        
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_s = s.copy()
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                s[start_idx:start_idx + block_size] = old_block
        
        T *= cooling
        
    return best_s

def _greedy_polish(s, time_budget, rng):
    """Aggressive greedy local search with hard objective and multiple passes."""
    s = s.copy()
    current_score = _hard_score(s)
    start_time = time.time()
    
    # Phase 1: Single bit flips, multiple passes
    improved = True
    passes = 0
    while improved and time.time() < start_time + time_budget:
        improved = False
        passes += 1
        # Shuffle order to avoid bias
        indices = rng.permutation(N)
        for i in indices:
            if time.time() > start_time + time_budget:
                break
            s[i] = -s[i]
            new_score = _hard_score(s)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                s[i] = -s[i] # revert
                
        # If no improvement in a full pass, stop
        if not improved:
            break

    # Phase 2: Small block flips if time permits
    if time.time() < start_time + time_budget - 1.0:
        for block_size in [2, 4, 8]:
            # Try a few random block flips
            for _ in range(50):
                if time.time() > start_time + time_budget:
                    break
                start_idx = rng.randint(0, N - block_size + 1)
                old_block = s[start_idx:start_idx + block_size].copy()
                s[start_idx:start_idx + block_size] = -old_block
                new_score = _hard_score(s)
                if new_score < current_score:
                    current_score = new_score
                else:
                    s[start_idx:start_idx + block_size] = old_block
                    
    return s, current_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    rng = np.random.default_rng(42)
    
    # Initialization strategies
    inits = []
    # 1. Random
    for _ in range(5):
        inits.append(rng.choice([-1, 1], size=N))
        
    # 2. Van der Corput
    for shift in range(0, N, 50):
        inits.append(_vdc_init(shift))
        
    # 3. Sine waves
    for freq in [2, 3, 5, 7, 10, 15]:
        for phase in [0, np.pi/2, np.pi]:
            inits.append(_sine_init(freq, phase, seed=int(freq*10+phase*10)))
            
    # Evaluate initial scores to prioritize
    scored_inits = []
    for s in inits:
        score = _soft_score(s, beta=10.0)
        scored_inits.append((score, s))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Strategy:
    # 1. Quick SA on top 5 inits
    # 2. Longer SA on the best result from step 1
    # 3. Greedy polish on the best result from step 2
    
    top_k = 5
    time_per_quick_sa = 5.0
    
    candidates = []
    
    for i in range(min(top_k, len(scored_inits))):
        if time.time() > deadline - 20.0:
            break
        s_init = scored_inits[i][1]
        s_sa = _simulated_annealing(s_init, time_per_quick_sa, rng)
        candidates.append(s_sa)
        
    # Select best candidate from quick SAs based on hard score
    best_candidate = None
    best_candidate_score = float('inf')
    for s in candidates:
        score = _hard_score(s)
        if score < best_candidate_score:
            best_candidate_score = score
            best_candidate = s
            
    if best_candidate is not None:
        if best_candidate_score < best_val:
            best_val = best_candidate_score
            best_seq = best_candidate.copy()
            
    # Long SA on the best candidate
    if time.time() < deadline - 15.0:
        long_sa_budget = min(15.0, deadline - 15.0 - time.time())
        s_long = _simulated_annealing(best_seq, long_sa_budget, rng)
        long_score = _hard_score(s_long)
        if long_score < best_val:
            best_val = long_score
            best_seq = s_long.copy()
            
    # Final Greedy Polish
    if time.time() < deadline - 2.0:
        polish_budget = deadline - 2.0 - time.time()
        best_seq, best_val = _greedy_polish(best_seq, polish_budget, rng)
        
    return best_seq
