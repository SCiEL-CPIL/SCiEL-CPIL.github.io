import numpy as np
import time

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    The overlap for shift k is: C(k) = sum_i h[i] * (1 - h[(i+k) % N])
    
    We want to minimize max_k C(k) subject to:
      - sum(h) = N/2
      - 0 <= h[i] <= 1
    
    Key insight: The constant h[i] = 0.5 gives C(k) = N/4 for all k.
    The lower bound is also N/4 (by averaging argument), so this is optimal.
    
    However, let's verify this more carefully. The sum over all k of C(k) is:
      sum_k sum_i h[i]*(1-h[(i+k)%N]) = sum_i h[i] * sum_k (1-h[(i+k)%N])
    = sum_i h[i] * (N - sum_j h[j]) = (N/2)*(N - N/2) = N^2/4
    
    So the average of C(k) over all k is N^2/4 / N = N/4.
    Therefore max_k C(k) >= N/4.
    
    The constant solution achieves exactly N/4, so it's optimal.
    
    But wait - the problem says "score is the upper bound on C5" and "the evaluator
    measures the raw value (lower is better)". Let me re-read.
    
    Actually, looking at the task more carefully: "minimize the maximum overlap of h 
    with shifted 1-h". The score is the raw value of this maximum overlap.
    
    For the constant solution h[i]=0.5, the max overlap is N/4 = 25 when N=100.
    
    Can we do better? The lower bound argument shows max_k C(k) >= N/4, so no.
    
    But wait - maybe I'm computing the overlap wrong. Let me reconsider.
    
    "overlap of h with shifted 1-h" - this could mean:
      C(k) = sum_i h[i] * (1 - h[(i+k) % N])
    
    OR it could mean the overlap in a different sense. Let me think about what "C5" means.
    
    The task mentions "score is the upper bound on C5". This might refer to a specific
    combinatorial quantity. Let me just implement the optimization and see.
    
    Actually, since the constant solution achieves the theoretical lower bound, it's
    provably optimal. But let me try a few different N values and also try to see if
    a different interpretation gives a different answer.
    
    Let me try: maybe the overlap is computed differently, e.g., only for shifts 1..N-1
    (excluding k=0), or maybe it's a circular vs linear shift.
    
    For now, I'll stick with the constant solution since it's provably optimal under
    the natural interpretation. But let me also try a local search to confirm.
    """
    N = 100
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with the known optimal solution
    h = np.full(N, 0.5)
    
    def max_overlap(h):
        """Compute max_k sum_i h[i] * (1 - h[(i+k)%N]) for k=1..N-1"""
        # Use FFT for efficiency
        # C(k) = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
        # = sum(h) - autocorrelation at lag k
        # autocorrelation via FFT
        h_pad = np.concatenate([h, np.zeros(N)])
        F = np.fft.fft(h_pad, n=2*N)
        ac = np.real(np.fft.ifft(F * np.conj(F)))[:N]
        # ac[k] = sum_i h[i]*h[(i+k)%N] for k=0..N-1
        # C(k) = sum(h) - ac[k]
        C = h.sum() - ac[1:]  # exclude k=0
        return np.max(C)
    
    best_h = h.copy()
    best_val = max_overlap(h)
    
    # Try local search with random perturbations
    rng = np.random.default_rng(42)
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Create a perturbed version
        h_trial = h.copy()
        # Pick a random pair of indices and transfer mass
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        delta = rng.uniform(-0.1, 0.1)
        if 0 <= h_trial[i] + delta <= 1 and 0 <= h_trial[j] - delta <= 1:
            h_trial[i] += delta
            h_trial[j] -= delta
            val = max_overlap(h_trial)
            if val < best_val:
                best_val = val
                best_h = h_trial.copy()
                h = h_trial.copy()
    
    # Return the best solution found
    h = best_h
    assert np.isclose(h.sum(), N/2), f"Sum {h.sum()} != {N/2}"
    assert np.all(h >= -1e-10) and np.all(h <= 1 + 1e-10)
    # Clip to [0,1] for safety
    h = np.clip(h, 0, 1)
    
    return h
