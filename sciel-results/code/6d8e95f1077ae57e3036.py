import numpy as np
import time

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    # Compute autocorrelation
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
    deadline = start_time + 115.0 # Safety margin for 120s limit
    
    best_h = None
    best_score = float('inf')
    
    # Initial random guesses
    # Use a fixed seed for reproducibility in local testing, but random for actual runs?
    # The task says "Choose a signed step function", implying we can search.
    # Let's try a few random starts.
    num_initial = 20
    for i in range(num_initial):
        if time.time() > deadline - 5:
            break
        h = np.random.choice([-1, 1], size=N)
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    if best_h is None:
        # Fallback
        return np.ones(N, dtype=int)

    h = best_h
    current_score = best_score
    
    # Local search / Hill climbing with simulated annealing-like restarts
    # To make it faster, we can precompute the correlation and update it incrementally?
    # For N=400, np.correlate is very fast (C implementation).
    # Let's estimate: 400*400 = 160k operations. np.correlate does this in < 1ms.
    # So we can do ~1000 iterations per second.
    # In 100 seconds, we can do ~100,000 iterations.
    # Each iteration tries N=400 flips. That's 40M flips. Might be tight but doable.
    # Let's optimize the inner loop.
    
    # Instead of trying all N flips in a full pass, we can do random flips or a subset.
    # Or we can do a full pass but break early if no improvement?
    # Let's try a full pass first.
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 50 # Restart if stuck for 50 full passes
    
    while time.time() < deadline - 5:
        improved = False
        # Shuffle order of indices to avoid bias?
        # For simplicity, sequential is fine for hill climbing.
        for j in range(N):
            if time.time() > deadline - 5:
                break
            
            old_val = h[j]
            h[j] = -h[j]
            
            new_score = _calc_score(h)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                # Accept
            else:
                # Revert
                h[j] = old_val
        
        if improved:
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb and restart climbing
                # Flip a random subset of bits
                num_perturb = np.random.randint(5, 20)
                indices = np.random.choice(N, num_perturb, replace=False)
                h[indices] *= -1
                current_score = _calc_score(h)
                no_improve_count = 0
                
                # If perturbation is much worse, maybe try a fresh random start?
                # Let's keep the best_h found so far.
                # If current_score is very bad, we might want to reset h to best_h?
                # Or just continue from perturbed state.
                # Let's try: if current_score > best_score * 1.5, reset to best_h and perturb less?
                # For now, just continue.
                
    return best_h
