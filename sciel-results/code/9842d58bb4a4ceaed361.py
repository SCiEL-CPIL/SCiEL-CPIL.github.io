import numpy as np
import time

N = 50

def _score_fast(h):
    """
    Calculate the score for a given height array.
    h is assumed to be normalized (sum=1) or we normalize here.
    """
    h = np.asarray(h)
    S = np.sum(h)
    if S == 0:
        return 0.0
    h = h / S
    Q = np.sum(h**2)
    # Convolution
    # c_k = sum h_i h_j s.t. i+j=k
    # Using np.convolve
    c = np.convolve(h, h)
    M = np.max(c)
    if M == 0:
        return 0.0
    return Q / M

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_h = np.ones(N) / N
    best_score = _score_fast(best_h)
    
    # Try a few random sparse configurations
    # and optimize them
    
    def optimize_local(h_init, max_iter=1000):
        h = h_init.copy()
        h = h / np.sum(h)
        current_score = _score_fast(h)
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Perturb
            # Move mass from one bin to another
            # Or adjust weights of non-zero bins
            
            # Strategy: Keep support fixed, optimize weights?
            # Or change support?
            
            # Simple hill climbing on weights of non-zero elements
            nz_idx = np.nonzero(h)[0]
            if len(nz_idx) < 2:
                continue
                
            # Pick two indices to swap mass
            i1 = np.random.choice(len(nz_idx))
            i2 = np.random.choice(len(nz_idx))
            idx1 = nz_idx[i1]
            idx2 = nz_idx[i2]
            
            # Try moving some mass
            delta = 0.01 * np.random.randn()
            h_new = h.copy()
            h_new[idx1] += delta
            h_new[idx2] -= delta
            
            # Ensure non-negative
            if h_new[idx1] < 0 or h_new[idx2] < 0:
                continue
                
            h_new = np.maximum(h_new, 0)
            h_new = h_new / np.sum(h_new)
            
            new_score = _score_fast(h_new)
            
            if new_score > current_score:
                h = h_new
                current_score = new_score
            else:
                # Accept with small probability (annealing)
                if np.random.rand() < 0.01:
                    h = h_new
                    current_score = new_score
                    
        return h, current_score

    # Initial guesses
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) / N)
    
    # 2. 2 Spikes
    h2 = np.zeros(N)
    h2[10] = 0.5
    h2[40] = 0.5
    candidates.append(h2)
    
    # 3. 3 Spikes
    h3 = np.zeros(N)
    h3[0] = 1/3
    h3[25] = 1/3
    h3[49] = 1/3
    candidates.append(h3)
    
    # 4. Random sparse
    for _ in range(5):
        hr = np.zeros(N)
        num_spikes = np.random.randint(2, 6)
        idxs = np.random.choice(N, num_spikes, replace=False)
        hr[idxs] = 1.0
        hr /= np.sum(hr)
        candidates.append(hr)

    for h_init in candidates:
        if time.time() > deadline:
            break
        h_opt, s_opt = optimize_local(h_init, max_iter=500)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt

    return list(best_h)
