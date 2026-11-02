"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time

N = 100
DEADLINE = 145.0  # safety margin before 150s kill

def _overlaps(h):
    """Compute overlap at every circular shift k: sum_i h[i]*(1-h[(i+k)%N]).
    Use FFT: overlap(k) = sum_i h[i]*h[i+k] - sum_i h[i] = autocorr(k) - N/2.
    Actually overlap(k) = sum_i h[i]*(1-h[(i+k)%N]) = N/2 - sum_i h[i]*h[(i+k)%N].
    So we want to maximize min_k autocorr(k) to minimize max overlap.
    max_overlap = N/2 - min_k autocorr(k).
    """
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # Use FFT: ifft(fft(h) * conj(fft(h))) gives circular autocorrelation
    H = np.fft.fft(h.astype(np.float64))
    ac = np.real(np.fft.ifft(H * np.conj(H)))
    ac = np.round(ac).astype(np.int64)  # should be integers for binary
    # ac[k] = sum_i h[i] * h[(i+k)%N] for k=0..N-1
    # overlap at shift k = N/2 - ac[k]  (since sum h = N/2)
    # But careful: ac[0] = sum h[i]^2 = N/2 for binary
    # overlap(0) = sum h[i]*(1-h[i]) = 0 for binary
    # Let's verify: overlap(k) = sum_i h[i]*(1-h[(i+k)%N])
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%N] = N/2 - ac[k]
    # For k=0: N/2 - N/2 = 0. Correct.
    ov = N / 2.0 - ac
    return ov

def _max_overlap(h):
    ov = _overlaps(h)
    return np.max(ov)

def construct():
    t0 = time.time()
    
    # Start with a good initial binary pattern
    # Use a known good difference-set-like pattern or random
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Try a few initial patterns
    # Pattern 1: alternating
    h = np.zeros(N)
    h[::2] = 1.0
    score = _max_overlap(h)
    best_h = h.copy()
    best_score = score
    
    # Pattern 2: random with exactly N/2 ones
    for _ in range(20):
        idx = np.random.permutation(N)[:N//2]
        h = np.zeros(N)
        h[idx] = 1.0
        score = _max_overlap(h)
        if score < best_score:
            best_h = h.copy()
            best_score = score
    
    # Simulated annealing
    h = best_h.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    stuck = 0
    
    while time.time() - t0 < DEADLINE:
        # Pick a random position to flip
        i = np.random.randint(N)
        # Try flipping
        h[i] = 1.0 - h[i]
        new_score = _max_overlap(h)
        
        # Accept if better or with probability exp(-(new-current)/T)
        if new_score <= current_score:
            current_score = new_score
            stuck = 0
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            delta = new_score - current_score
            if np.random.random() < np.exp(-delta / max(T, 1e-6)):
                current_score = new_score
                stuck += 1
            else:
                h[i] = 1.0 - h[i]  # revert
                stuck += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # If stuck too long, restart with perturbation
        if stuck > 200:
            stuck = 0
            T = max(T, 0.5)
            # Perturb: flip a few random positions
            for _ in range(3):
                j = np.random.randint(N)
                h[j] = 1.0 - h[j]
            # Rebalance if needed
            s = np.sum(h)
            if s > N/2:
                excess = int(s - N/2)
                ones = np.where(h == 1.0)[0]
                to_flip = np.random.choice(ones, excess, replace=False)
                h[to_flip] = 0.0
            elif s < N/2:
                deficit = int(N/2 - s)
                zeros = np.where(h == 0.0)[0]
                to_flip = np.random.choice(zeros, deficit, replace=False)
                h[to_flip] = 1.0
    
    # Ensure exact sum N/2
    s = np.sum(best_h)
    if s > N/2:
        excess = int(s - N/2)
        ones = np.where(best_h == 1.0)[0]
        to_flip = np.random.choice(ones, excess, replace=False)
        best_h[to_flip] = 0.0
    elif s < N/2:
        deficit = int(N/2 - s)
        zeros = np.where(best_h == 0.0)[0]
        to_flip = np.random.choice(zeros, deficit, replace=False)
        best_h[to_flip] = 1.0
    
    return best_h
