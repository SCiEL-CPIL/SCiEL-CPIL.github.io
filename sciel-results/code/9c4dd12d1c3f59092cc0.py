import numpy as np
import time

N = 50

def _ratio(h):
    """
    Compute the ratio (sum h^2) / (sum h * max autoconv)
    This is proportional to the continuous ratio.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    # Discrete autoconvolution max
    # c[k] = sum_i h[i] h[i-k]
    # Using correlate
    c = np.correlate(h, h, mode='full')
    ac_max = c.max()
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Try several initializations and optimize each
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Spike at center
    spike = np.zeros(N)
    spike[N//2] = 1.0
    initializations.append(spike)
    
    # 3. Two adjacent spikes
    two_spikes = np.zeros(N)
    two_spikes[N//2] = 1.0
    two_spikes[N//2 + 1] = 1.0
    initializations.append(two_spikes)
    
    # 4. Gaussian centered
    x = np.linspace(0, N-1, N)
    gauss = np.exp(-0.5 * ((x - N//2) / (N/4))**2)
    initializations.append(gauss)
    
    # 5. Random nonnegative
    rng = np.random.RandomState(42)
    for _ in range(10):
        initializations.append(rng.random(N))
    
    # 6. Block patterns
    for block_size in [2, 3, 5, 10]:
        block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            block[i:i+block_size] = 1.0
        initializations.append(block)
    
    # Optimize each initialization with coordinate descent + simulated annealing
    for init in initializations:
        if time.time() > deadline:
            break
        
        h = init.copy()
        h_score = _ratio(h)
        
        if h_score > best_score:
            best_score = h_score
            best_h = h.copy()
        
        # Coordinate descent with simulated annealing
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Random order of coordinates
            order = np.random.permutation(N)
            
            for idx in order:
                if time.time() > deadline:
                    break
                
                old_val = h[idx]
                
                # Try several perturbations
                candidates = []
                # Small perturbations
                for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0]:
                    new_val = old_val + delta
                    if new_val >= 0:
                        candidates.append(new_val)
                
                # Reset to zero
                candidates.append(0.0)
                
                # Large jump
                if old_val > 0:
                    candidates.append(old_val * 0.1)
                    candidates.append(old_val * 2.0)
                
                # Try each candidate
                best_candidate = old_val
                best_candidate_score = h_score
                
                for cand in candidates:
                    h[idx] = cand
                    cand_score = _ratio(h)
                    
                    # Accept if better, or with some probability if worse (annealing)
                    if cand_score > best_candidate_score:
                        best_candidate = cand
                        best_candidate_score = cand_score
                    
                    # Simulated annealing acceptance
                    elif cand_score > 0 and best_candidate_score > 0:
                        T = max(0.001, 0.1 * (1 - iteration / 500))
                        if np.random.rand() < np.exp(-(best_candidate_score - cand_score) / (T * best_candidate_score)):
                            best_candidate = cand
                            best_candidate_score = cand_score
                
                h[idx] = best_candidate
                h_score = best_candidate_score
                
                if h_score > best_score:
                    best_score = h_score
                    best_h = h.copy()
    
    # Final local refinement
    if time.time() < deadline - 1.0:
        h = best_h.copy()
        for iteration in range(200):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(N):
                old_val = h[idx]
                best_val = old_val
                best_score_local = _ratio(h)
                
                for delta in [-0.1, -0.05, 0.05, 0.1, 0.2, 0.5]:
                    new_val = old_val + delta
                    if new_val >= 0:
                        h[idx] = new_val
                        s = _ratio(h)
                        if s > best_score_local:
                            best_score_local = s
                            best_val = new_val
                
                h[idx] = best_val
                if best_val != old_val:
                    improved = True
            
            if not improved:
                break
        
        final_score = _ratio(h)
        if final_score > best_score:
            best_score = final_score
            best_h = h
    
    # Ensure nonnegative and return
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
