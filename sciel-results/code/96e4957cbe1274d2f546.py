import numpy as np
import time

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    # np.correlate(h, h, 'full') gives lags from -(N-1) to N-1.
    # Index N-1 corresponds to lag 0.
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    best_h = None
    best_score = float('inf')

    # Strategy: Use chaotic deterministic sequences (Logistic Map) to generate
    # candidate +/- 1 sequences. These sequences often have low autocorrelation
    # sidelobes compared to random sequences, and we can tune parameters to find
    # better ones. We also use a simple local search (bit flipping) to refine
    # the best candidates found.

    # Precompute some logistic map trajectories for different parameters
    # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
    # We want sequences that stay in the chaotic regime (r around 3.57 to 4.0)
    
    # First, try some random initializations with logistic map
    num_logistic = 50
    for i in range(num_logistic):
        if time.time() > deadline - 20:
            break
        
        # Choose r in the chaotic regime
        r = 3.57 + 0.43 * (i / num_logistic)
        # Choose initial x0
        x0 = 0.1 + 0.8 * (i / num_logistic)
        
        # Generate the sequence
        x = x0
        h = np.zeros(N, dtype=int)
        for j in range(N):
            x = r * x * (1 - x)
            # Map to +/- 1 based on threshold
            # Threshold at 0.5 is standard, but we can try other thresholds
            # For now, use 0.5
            h[j] = 1 if x > 0.5 else -1
        
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # If we have a good candidate, try to improve it with local search
    if best_h is not None:
        h = best_h.copy()
        score = best_score
        improved = True
        max_iterations = 1000
        
        while improved and time.time() < deadline - 10:
            improved = False
            # Try flipping each bit and see if it improves the score
            for j in range(N):
                if time.time() > deadline - 10:
                    break
                h[j] *= -1
                new_score = _calc_score(h)
                if new_score < score:
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
                else:
                    h[j] *= -1  # Revert
                
                # Early exit if we reach a very low score
                if score < 0.01:
                    break

    # If local search didn't help much, try a few more logistic maps with different thresholds
    if time.time() < deadline - 30:
        for i in range(20):
            if time.time() > deadline - 10:
                break
            
            r = 3.9 + 0.1 * (i / 20)
            x0 = 0.1 + 0.8 * (i / 20)
            
            x = x0
            h = np.zeros(N, dtype=int)
            for j in range(N):
                x = r * x * (1 - x)
                # Try different thresholds
                threshold = 0.4 + 0.2 * (i / 20)
                h[j] = 1 if x > threshold else -1
            
            score = _calc_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()

    # Final fallback: if no good candidate found, use a simple alternating pattern
    if best_h is None:
        best_h = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
        best_score = _calc_score(best_h)

    return best_h
