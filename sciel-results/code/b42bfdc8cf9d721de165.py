import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # 1. Initialize with a strong candidate (Gaussian-like shape)
    # The seed was a delta (score 1.0). 
    # A spread-out shape (like a Gaussian) usually scores higher because 
    # the L2 norm increases relative to the autoconvolution peak compared to a spike.
    # Let's start with a Gaussian centered in the middle.
    x = np.arange(N) - (N - 1) / 2.0
    h = np.exp(-0.5 * (x / (N / 4.0))**2)
    
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    
    # 2. Refine using coordinate descent / local search
    # We want to maximize the ratio. 
    # Since the ratio is scale-invariant (if we normalize by L1), we can work with 
    # unnormalized h or normalized h. The compute_ratio function normalizes internally.
    # So we can just adjust the shape.
    
    # Adaptive step size
    step_size = 0.1
    improved = True
    while improved and (time.time() < deadline - 10.0):
        improved = False
        for i in range(N):
            if time.time() > deadline - 10.0:
                break
            
            # Try increasing
            h_trial = best_h.copy()
            h_trial[i] += step_size
            # Ensure non-negative
            h_trial[h_trial < 0] = 0.0
            sc = compute_ratio(h_trial)
            if sc > best_score:
                best_score = sc
                best_h = h_trial
                improved = True
                continue # Re-evaluate this index again with same step? Or move on? 
                         # Usually move on or shrink step if stuck. Let's move on for now.
            
            # Try decreasing
            h_trial = best_h.copy()
            h_trial[i] -= step_size
            h_trial[h_trial < 0] = 0.0
            sc = compute_ratio(h_trial)
            if sc > best_score:
                best_score = sc
                best_h = h_trial
                improved = True
        
        # If no improvement, reduce step size to refine
        if not improved:
            step_size *= 0.5
            if step_size < 1e-4:
                break
    
    # 3. Try a few global structural changes if time permits
    # Sometimes local optima can be trapped. 
    # Try different initializations and run a quick local search on them.
    # But we are close to deadline, so only if fast.
    
    if time.time() < deadline - 20.0:
        # Try a triangular shape
        h_tri = np.maximum(0, 1 - np.abs(x) / (N / 2.0))
        h_tri = h_tri / np.sum(h_tri) # Normalize for consistency in search, though compute_ratio handles it
        
        # Quick local search on triangular
        best_h_tri = h_tri.copy()
        best_score_tri = compute_ratio(best_h_tri)
        
        step_tri = 0.05
        for _ in range(10): # Fewer iterations for secondary search
            if time.time() > deadline - 10.0:
                break
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                h_trial = best_h_tri.copy()
                h_trial[i] += step_tri
                h_trial[h_trial < 0] = 0.0
                sc = compute_ratio(h_trial)
                if sc > best_score_tri:
                    best_score_tri = sc
                    best_h_tri = h_trial
                    continue
                h_trial = best_h_tri.copy()
                h_trial[i] -= step_tri
                h_trial[h_trial < 0] = 0.0
                sc = compute_ratio(h_trial)
                if sc > best_score_tri:
                    best_score_tri = sc
                    best_h_tri = h_trial
        
        if best_score_tri > best_score:
            best_h = best_h_tri
            best_score = best_score_tri

    return list(best_h)
