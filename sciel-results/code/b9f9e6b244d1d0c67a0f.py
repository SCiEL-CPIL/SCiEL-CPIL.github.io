import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Pre-compute FFT size for efficiency
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2

    def get_autocorr(seq):
        """Compute non-zero lag autocorrelations."""
        f = np.zeros(fft_size, dtype=np.complex128)
        f[:N] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac = np.fft.ifft(F * np.conj(F)).real[:N]
        return ac[1:]

    def score(seq):
        """Max absolute autocorrelation."""
        ac = get_autocorr(seq)
        return np.max(np.abs(ac))

    def lfsr_sequence(n, poly=0x199, init=1 << 8, length=511):
        """Generate LFSR m-sequence."""
        lfsr = init
        seq = []
        for _ in range(length):
            seq.append(1 if (lfsr & 1) else -1)
            if lfsr & 1:
                lfsr = (lfsr >> 1) ^ poly
            else:
                lfsr >>= 1
        return np.array(seq[:n], dtype=np.float64)

    def local_search(h, deadline, max_passes=50):
        """Greedy local search with single bit flips."""
        h = h.copy()
        current_score = score(h)
        
        # Pre-compute autocorr for faster delta calculation? 
        # Actually, for N=400, FFT is fast enough (~0.1ms). 
        # 400 flips * 50 passes = 20,000 evaluations. 20,000 * 0.1ms = 2s. 
        # This is very fast. We can afford many restarts.
        
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            # Random order to avoid bias
            perm = np.random.permutation(N)
            for idx in perm:
                if time.time() > deadline:
                    return h, current_score
                
                h[idx] *= -1
                new_score = score(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[idx] *= -1
            passes += 1
            
        return h, current_score

    best_h = None
    best_score = float('inf')
    rng = np.random.default_rng(42)

    # Strategy 1: LFSR based initialization
    # Generate base LFSR sequence
    base_lfsr = lfsr_sequence(N)
    
    # Try the raw truncated LFSR
    h_candidate = base_lfsr.copy()
    h_candidate, s_candidate = local_search(h_candidate, deadline, max_passes=20)
    if s_candidate < best_score:
        best_score = s_candidate
        best_h = h_candidate.copy()

    # Try LFSR with small random perturbations to escape local minima
    for i in range(10):
        if time.time() > deadline - 10: # Keep 10s buffer
            break
        
        h_candidate = base_lfsr.copy()
        # Flip 1-5 random bits
        n_flips = rng.integers(1, 6)
        flip_indices = rng.choice(N, size=n_flips, replace=False)
        h_candidate[flip_indices] *= -1
        
        h_candidate, s_candidate = local_search(h_candidate, deadline, max_passes=20)
        if s_candidate < best_score:
            best_score = s_candidate
            best_h = h_candidate.copy()

    # Strategy 2: Random initialization (fallback / diversity)
    if time.time() < deadline - 20:
        h_rand = rng.choice([-1, 1], size=N).astype(np.float64)
        h_rand, s_rand = local_search(h_rand, deadline, max_passes=10)
        if s_rand < best_score:
            best_score = s_rand
            best_h = h_rand.copy()

    # Final polish on the best candidate found
    if best_h is not None and time.time() < deadline - 5:
        # Run a few more passes of local search on the best one
        best_h, best_score = local_search(best_h, deadline, max_passes=10)

    return best_h
