"""Search-based program for Erdos minimum overlap task."""
import numpy as np
import time

N = 100

def _max_overlap_fft(h):
    """
    Compute max over shifts of mean(h[i]*(1-h[(i+s)%N])).
    Uses FFT for speed.
    """
    g = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    corr = np.fft.ifft(H * G.conj()).real
    max_ov = np.max(corr) / N
    return max_ov

def _construct_periodic(base_pattern, N):
    """
    Expand a base pattern of length P to length N by tiling.
    Adjust to sum to N/2.
    """
    P = len(base_pattern)
    reps = N // P + 1
    tiled = np.tile(base_pattern, reps)[:N]
    
    current_sum = tiled.sum()
    target = N / 2.0
    if current_sum != 0:
        tiled = tiled * (target / current_sum)
    else:
        tiled = np.full(N, 0.5)
        
    tiled = np.clip(tiled, 0.0, 1.0)
    s = tiled.sum()
    if s > 0:
        tiled = tiled * (target / s)
        tiled = np.clip(tiled, 0.0, 1.0)
        diff = target - tiled.sum()
        if abs(diff) > 1e-12:
            idx = 0
            tiled[idx] += diff
            tiled = np.clip(tiled, 0.0, 1.0)
            diff2 = target - tiled.sum()
            if abs(diff2) > 1e-12:
                idx2 = 1
                tiled[idx2] += diff2
                tiled = np.clip(tiled, 0.0, 1.0)
    return tiled

def _score(h):
    s = h.sum()
    if abs(s - N/2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-8 or h.max() > 1.0 + 1e-8:
        return float('inf')
    return _max_overlap_fft(h)

def _search_periodic(P, deadline):
    """
    Search for best periodic pattern of period P.
    """
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    rng = np.random.default_rng(42 + P)
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        base = rng.random(P)
        
        for step in range(200):
            if time.time() > deadline:
                return best_h, best_score
            
            h_curr = _construct_periodic(base, N)
            sc_curr = _score(h_curr)
            
            if sc_curr < best_score:
                best_score = sc_curr
                best_h = h_curr.copy()
                
            num_pert = rng.integers(1, 4)
            idxs = rng.choice(P, size=num_pert, replace=False)
            base_trial = base.copy()
            for i in idxs:
                base_trial[i] += rng.uniform(-0.1, 0.1)
                base_trial[i] = np.clip(base_trial[i], 0.0, 1.0)
                
            h_trial = _construct_periodic(base_trial, N)
            sc_trial = _score(h_trial)
            
            if sc_trial < sc_curr:
                base = base_trial
                
    return best_h, best_score

def _local_polish(h, deadline):
    """
    Final local search on the full N-dimensional vector.
    """
    best_h = h.copy()
    best_score = _score(best_h)
    rng = np.random.default_rng(123)
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        h_trial = best_h.copy()
        idxs = rng.choice(N, size=5, replace=False)
        for i in idxs:
            h_trial[i] += rng.uniform(-0.05, 0.05)
            h_trial[i] = np.clip(h_trial[i], 0.0, 1.0)
            
        diff = N/2.0 - h_trial.sum()
        if abs(diff) > 1e-12:
            s = h_trial.sum()
            if s > 0:
                h_trial = h_trial * (N/2.0 / s)
                h_trial = np.clip(h_trial, 0.0, 1.0)
                
        sc_trial = _score(h_trial)
        if sc_trial < best_score:
            best_score = sc_trial
            best_h = h_trial.copy()
            
    return best_h, best_score

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    for P in range(2, 21):
        if time.time() > deadline:
            break
        h_p, sc_p = _search_periodic(P, deadline)
        if sc_p < best_score:
            best_score = sc_p
            best_h = h_p.copy()
            
    best_h, best_score = _local_polish(best_h, deadline)
    
    return best_h
# EVOLVE-BLOCK-END
