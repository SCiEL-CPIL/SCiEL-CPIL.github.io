import numpy as np
import time
import random
from itertools import combinations

N = 50

def _autoconv_peak(h):
    """Compute the maximum value of the autoconvolution of h."""
    h = np.asarray(h, dtype=np.float64)
    # Autoconvolution: (h * h)(k) = sum_i h[i] * h[k-i] for valid indices
    # For discrete signals, use np.convolve with mode='full'
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||autoconv||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    if np.max(h) < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if l1 * peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def construct():
    deadline = time.time() + 140.0  # safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_score = _score(best_h)
    
    # Strategy: Try to find sparse-like patterns that minimize autoconvolution peak
    # Key idea: If we use a pattern where indices are "additively independent"
    # (few collisions in sums), the autoconvolution peak will be lower.
    
    # Approach 1: Try various structured patterns
    # Pattern: geometric spacing or prime-like spacing
    
    rng = random.Random(42)
    
    # Try different structured patterns
    patterns = []
    
    # All ones (baseline)
    patterns.append(np.ones(N))
    
    # Sparse: only a few non-zero entries
    for k in range(1, min(N+1, 15)):
        # Try random subsets of size k
        for trial in range(20):
            idx = set(rng.sample(range(N), k))
            h = np.zeros(N)
            h[list(idx)] = 1.0
            patterns.append(h)
    
    # Try geometric progression-like patterns
    for start in range(0, N, 3):
        for ratio in [2, 3]:
            h = np.zeros(N)
            idx = start
            while idx < N:
                h[idx] = 1.0
                idx += ratio
            if np.sum(h) > 0:
                patterns.append(h)
    
    # Try alternating patterns
    for period in [2, 3, 4, 5]:
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i - offset) % period == 0:
                    h[i] = 1.0
            if np.sum(h) > 0:
                patterns.append(h)
    
    # Try patterns based on quadratic residues or other number-theoretic constructions
    # For now, let's try some more random patterns with optimization
    
    # Evaluate all patterns
    for h in patterns:
        if time.time() > deadline - 10:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization: try to improve by adjusting individual entries
    # Use a simple coordinate ascent / descent approach
    current_h = best_h.copy()
    current_score = best_score
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try flipping or adjusting each entry
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            # Try setting h[i] to various values
            old_val = current_h[i]
            candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
            if old_val > 0:
                candidates.append(old_val * 0.5)
                candidates.append(old_val * 2.0)
            
            best_candidate = None
            best_candidate_score = current_score
            
            for val in candidates:
                if val < 0:
                    continue
                current_h[i] = val
                s = _score(current_h)
                if s > best_candidate_score + 1e-12:
                    best_candidate_score = s
                    best_candidate = val
            
            if best_candidate is not None:
                current_h[i] = best_candidate
                current_score = best_candidate_score
                improved = True
            else:
                current_h[i] = old_val
        
        if not improved:
            break
        
        if current_score > best_score:
            best_score = current_score
            best_h = current_h.copy()
    
    # Try some random restarts with local optimization
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Generate a random starting point
        h = np.zeros(N)
        # Randomly set some entries to random positive values
        n_nonzero = rng.randint(1, N)
        idx = rng.sample(range(N), n_nonzero)
        for i in idx:
            h[i] = rng.uniform(0.1, 2.0)
        
        # Local optimization
        current_h = h.copy()
        current_score = _score(current_h)
        
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                old_val = current_h[i]
                candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
                if old_val > 0:
                    candidates.append(old_val * 0.5)
                    candidates.append(old_val * 2.0)
                
                best_candidate = None
                best_candidate_score = current_score
                
                for val in candidates:
                    if val < 0:
                        continue
                    current_h[i] = val
                    s = _score(current_h)
                    if s > best_candidate_score + 1e-12:
                        best_candidate_score = s
                        best_candidate = val
                
                if best_candidate is not None:
                    current_h[i] = best_candidate
                    current_score = best_candidate_score
                    improved = True
                else:
                    current_h[i] = old_val
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_h = current_h.copy()
    
    return list(best_h)
