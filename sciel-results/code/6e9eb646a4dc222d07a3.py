import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    def get_peak_sidelobe(x):
        """Compute peak sidelobe magnitude of aperiodic autocorrelation."""
        n_fft = 2 * N
        F = np.fft.rfft(x, n=n_fft)
        P = F * np.conj(F)
        R = np.fft.irfft(P, n=n_fft)
        return np.max(np.abs(R[1:N]))
    
    def greedy_construct(rng):
        """Construct initial sequence greedily: place each sign to minimize peak sidelobe so far."""
        x = np.zeros(N)
        for i in range(N):
            # Try both signs
            best_sign = 1.0
            best_score = float('inf')
            for sign in [1.0, -1.0]:
                x[i] = sign
                # Only compute autocorrelation for the partial sequence
                # For efficiency, just evaluate the current peak
                if i >= 1:
                    # Quick check: compute autocorrelation of partial sequence
                    # Pad to handle aperiodic
                    n_partial = i + 1
                    n_fft_p = 2 * n_partial
                    F_p = np.fft.rfft(x[:n_partial], n=n_fft_p)
                    P_p = F_p * np.conj(F_p)
                    R_p = np.fft.irfft(P_p, n=n_fft_p)
                    # For partial sequence, lags 1..i
                    peak = np.max(np.abs(R_p[1:n_partial])) if n_partial > 1 else 0.0
                else:
                    peak = 0.0
                if peak < best_score:
                    best_score = peak
                    best_sign = sign
            x[i] = best_sign
        return x
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_score = float('inf')
    
    # Phase 1: Generate several diverse initial sequences
    initial_sequences = []
    
    # Greedy construction
    initial_sequences.append(greedy_construct(rng))
    
    # Random sequences
    for _ in range(3):
        initial_sequences.append(rng.choice([-1.0, 1.0], size=N))
    
    # One more: alternating pattern with random perturbations
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    # Randomly flip 10% of positions
    flip_mask = rng.random(N) < 0.1
    alt[flip_mask] *= -1.0
    initial_sequences.append(alt)
    
    # Phase 2: Refine each initial sequence with block-flip SA
    sa_deadline = deadline - 15.0  # Leave 15s for final polish
    
    for init_x in initial_sequences:
        if time.time() > sa_deadline:
            break
        
        x = init_x.copy()
        current_score = get_peak_sidelobe(x)
        run_best_x = x.copy()
        run_best_score = current_score
        
        # Block-flip SA with adaptive block size
        T = 8.0
        T_min = 0.005
        cooling_rate = 0.997
        
        while T > T_min and time.time() < sa_deadline:
            # Adaptive block size: large early, small late
            # Block size ranges from ~N/4 down to 1
            min_block = 1
            max_block = max(1, int(N * T / 10.0))
            block_size = rng.integers(min_block, max_block + 1)
            
            # Pick random start position
            start = rng.integers(0, N - block_size + 1)
            end = start + block_size
            
            # Flip the block
            old_block = x[start:end].copy()
            x[start:end] = -x[start:end]
            
            new_score = get_peak_sidelobe(x)
            delta = new_score - current_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < run_best_score:
                    run_best_score = new_score
                    run_best_x = x.copy()
            else:
                # Revert
                x[start:end] = old_block
            
            T *= cooling_rate
        
        # Update global best
        if run_best_score < best_score:
            best_score = run_best_score
            best_x = run_best_x.copy()
        
        # Early termination if we've found a very good solution
        if best_score < 22.0:
            break
    
    # Phase 3: Final polish with single-bit greedy local search
    if best_x is not None and time.time() < deadline - 2.0:
        x = best_x.copy()
        current_score = best_score
        polish_deadline = deadline - 2.0
        
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(N):
                if time.time() > polish_deadline:
                    break
                
                old_val = x[i]
                x[i] = -old_val
                new_score = get_peak_sidelobe(x)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                else:
                    x[i] = old_val
    
    return best_x
