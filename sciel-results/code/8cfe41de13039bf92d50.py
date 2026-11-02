import numpy as np
import time

N = 400

def _score(s):
    """
    Compute the upper bound on C3, which is the max absolute off-peak autocorrelation.
    s: numpy array of +/-1 of length N
    """
    # Use FFT to compute autocorrelation
    # n_fft should be at least 2*N to avoid circular convolution artifacts for lags up to N-1
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    s_f = s.astype(np.float64)
    F = np.fft.fft(s_f, n=n_fft)
    # Autocorrelation is inverse FFT of |F|^2
    acf = np.fft.ifft(F * np.conj(F)).real[:N]
    
    # Normalize by N to get average power? 
    # The task says "upper bound on C3". 
    # In the context of the third autocorrelation inequality, C3 is often related to the 
    # sum of cubes of autocorrelations or similar. However, the prompt says 
    # "score is the upper bound on C3" and "evaluator measures the raw value (lower is better)".
    # The seed program used `acf /= N` and returned `max(abs(acf[1:]))`.
    # Let's stick to the normalization used in the seed to ensure comparability, 
    # although the absolute scale doesn't change the optimization landscape if consistent.
    # Actually, the seed program score was ~0.014. 
    # If we don't normalize, the max autocorr for random sequence is ~sqrt(N) ~ 20.
    # If we normalize by N, it is ~sqrt(N)/N = 1/sqrt(N) ~ 0.05.
    # The seed score 0.0142 is lower than 0.05, indicating a good sequence.
    # Let's use the same normalization as the seed: divide by N.
    
    acf /= N
    return float(np.max(np.abs(acf[1:])))

def construct():
    """
    Construct a signed step function (sequence of +/-1) of length 400 
    that minimizes the max absolute off-peak autocorrelation.
    """
    deadline = time.time() + 110  # Safety margin
    rng = np.random.default_rng(42)
    
    # Initialize with a random sequence
    seq = rng.choice([-1, 1], size=N).astype(np.int8)
    current_score = _score(seq)
    best_seq = seq.copy()
    best_score = current_score
    
    # Iterated Local Search
    # We will perform local search (best improvement) until a local minimum is reached,
    # then perturb and restart.
    
    perturbation_size = 10  # Number of bits to flip when perturbing
    
    while time.time() < deadline:
        improved = True
        # Local Search Phase: Best Improvement Hill Climbing
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_idx = -1
            
            # Evaluate all possible single-bit flips
            for i in range(N):
                # Flip bit i
                seq[i] *= -1
                new_score = _score(seq)
                delta = new_score - current_score
                
                # Revert immediately to keep seq consistent for next trial
                seq[i] *= -1
                
                if delta < best_delta:
                    best_delta = delta
                    best_idx = i
            
            # If an improving flip was found, apply it
            if best_idx != -1:
                seq[best_idx] *= -1
                current_score += best_delta
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_seq = seq.copy()
                    
        # Perturbation Phase
        if time.time() >= deadline:
            break
            
        # Flip 'perturbation_size' random bits to escape local minimum
        idxs = rng.choice(N, size=perturbation_size, replace=False)
        seq[idxs] *= -1
        current_score = _score(seq)
        
        # If the perturbed sequence is better than the current best, update
        if current_score < best_score:
            best_score = current_score
            best_seq = seq.copy()

    return best_seq
