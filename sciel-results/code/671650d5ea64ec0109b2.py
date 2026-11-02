import numpy as np
import time
import sys

N = 50

def compute_score(h):
    """
    Computes the score assuming the evaluator normalizes by L1 norm.
    Score = (||h||_2^2 / ||h||_1) / (||h*h||_infty / ||h||_1) 
           = ||h||_2^2 / ||h*h||_infty  (if normalized to L1=1)
    Wait, the prompt says: "ratio of squared L2 norm to L1 norm times supremum norm"
    R = ||h||_2^2 / (||h||_1 * ||h*h||_infty)
    
    If the evaluator normalizes h such that ||h||_1 = 1, then:
    R_norm = (||h_norm||_2^2) / (1 * ||h_norm*h_norm||_infty)
    
    Let h_raw be our output.
    h_norm = h_raw / ||h_raw||_1
    ||h_norm||_2^2 = ||h_raw||_2^2 / ||h_raw||_1^2
    ||h_norm*h_norm||_infty = ||h_raw*h_raw||_infty / ||h_raw||_1^2
    
    R_norm = (||h_raw||_2^2 / ||h_raw||_1^2) / (||h_raw*h_raw||_infty / ||h_raw||_1^2)
           = ||h_raw||_2^2 / ||h_raw*h_raw||_infty
    
    So the score to maximize is ||h||_2^2 / ||h*h||_infty.
    """
    h = np.array(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    # Compute autoconvolution
    # h_norm is length N. Convolution length is 2N-1.
    conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(conv)
    if max_conv == 0:
        return 0.0
    return l2_sq / max_conv

def generate_sidon_set(n_max, target_size=None):
    """
    Greedy construction of a Sidon set in {0, ..., n_max-1}.
    A Sidon set has distinct pairwise sums (i+j).
    """
    S = [0, 1]
    sums = set([0+1]) # i+j where i<j. We also need to consider i=i? 
    # Standard Sidon: all sums i+j (i<=j) are distinct.
    # Let's check sums for i<=j.
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    
    for x in range(2, n_max):
        # Check if x can be added
        valid = True
        new_sums = set()
        for i in range(len(S)):
            new_sums.add(S[i] + x)
        # Check against existing sums
        # The new sums must not overlap with existing sums, 
        # except potentially if the pair is unique?
        # No, all sums in the set must be distinct.
        # So new_sums must be disjoint from sums.
        if new_sums & sums:
            continue
        
        # If valid, add x
        S.append(x)
        sums.update(new_sums)
        
        if target_size and len(S) >= target_size:
            break
            
    return S

def optimize_amplitudes(support, max_iter=1000, tol=1e-8):
    """
    Optimize amplitudes on a fixed support to maximize ||h||_2^2 / ||h*h||_infty.
    Using gradient ascent or coordinate ascent.
    """
    N_global = N
    h = np.zeros(N_global)
    for idx in support:
        h[idx] = 1.0 / len(support)
    
    best_score = compute_score(h)
    best_h = h.copy()
    
    # Simple coordinate ascent / local search
    # We can try scaling individual weights
    step_size = 0.01
    
    for _ in range(max_iter):
        improved = False
        for idx in support:
            # Try increasing and decreasing h[idx]
            original = h[idx]
            for delta in [-step_size, step_size, -0.5*step_size, 0.5*step_size]:
                new_h = h.copy()
                new_h[idx] = max(0, original + delta)
                # Renormalize? No, the score is scale invariant?
                # Wait, R = ||h||_2^2 / ||h*h||_infty.
                # If we scale h by c, R becomes c^2 ||h||_2^2 / c^2 ||h*h||_infty = R.
                # So the score is scale invariant.
                # We can fix sum(h) = 1 to avoid scaling drift.
                if np.sum(new_h) > 0:
                    new_h = new_h / np.sum(new_h)
                    score = compute_score(new_h)
                    if score > best_score + tol:
                        h = new_h
                        best_score = score
                        best_h = h.copy()
                        improved = True
                        break
            if not improved:
                # Try small random perturbations
                pass
        
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate Sidon set
    # Greedy might not find the largest, but it's a good start.
    # Let's try a few random restarts or different greedy orders.
    best_support = None
    best_score = 0.0
    best_h = None
    
    # Try standard greedy
    S1 = generate_sidon_set(N)
    h1, s1 = optimize_amplitudes(S1)
    if s1 > best_score:
        best_score = s1
        best_h = h1
        best_support = S1
        
    # Try a few random permutations of the greedy selection to find larger sets?
    # Or just use the greedy result. Size is likely 7 or 8.
    
    # If time permits, try to improve support by swapping elements?
    # For now, the structural change (Sidon) is the main driver.
    
    # Ensure output is a list of floats
    return [float(x) for x in best_h]
