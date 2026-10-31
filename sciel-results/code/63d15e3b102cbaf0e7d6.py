"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1] efficiently."""
    h = np.array(h, dtype=float)
    # Iterative projection to handle bounds and sum simultaneously
    # Simple: adjust sum, clip, repeat
    for _ in range(10):
        s = np.sum(h)
        diff = N / 2.0 - s
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final correction
    s = np.sum(h)
    diff = N / 2.0 - s
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _thue_morse_sequence(n):
    """Generate first n terms of Thue-Morse sequence (0 or 1)."""
    seq = [0]
    for i in range(1, n):
        # t(n) = parity of popcount(n)
        seq.append(bin(i).count('1') % 2)
    return np.array(seq)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    # 1. Generate diverse initial guesses
    
    candidates = []
    
    # A: 2-block Gold Mean (Current best base)
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    candidates.append(h0)
    
    # B: Thue-Morse based (Binary structure, low autocorrelation)
    tm = _thue_morse_sequence(N)
    # Map 0,1 to 0,1 and normalize to sum N/2
    # Thue-Morse for N=100 might not sum to exactly 50.
    h_tm = tm.astype(float)
    # Adjust to sum N/2 by scaling? No, keep binary-ish.
    # If sum is not N/2, flip some bits?
    # Simpler: Map to [0,1] and let optimizer fix sum.
    # But SLSQP needs a good start.
    # Let's create a "binary" seed that sums to N/2.
    # Start with random permutation of 50 zeros and 50 ones?
    # Or use Thue-Morse and adjust.
    sum_tm = np.sum(h_tm)
    if sum_tm != N / 2.0:
        # Flip bits to adjust sum
        # If sum > N/2, flip 1s to 0s. If sum < N/2, flip 0s to 1s.
        indices_1 = np.where(h_tm == 1)[0]
        indices_0 = np.where(h_tm == 0)[0]
        if sum_tm > N / 2.0:
            num_to_flip = int(sum_tm - N / 2.0)
            # Flip last num_to_flip ones
            h_tm[indices_1[-num_to_flip:]] = 0
        else:
            num_to_flip = int(N / 2.0 - sum_tm)
            h_tm[indices_0[-num_to_flip:]] = 1
    candidates.append(h_tm)
    
    # C: Multi-block pattern (4 blocks)
    # Try to distribute the "high" and "low" regions to reduce local overlap peaks
    # Pattern: High, Low, High, Low
    # Widths: 25, 25, 25, 25
    # Values: A, B, A, B
    h4 = np.zeros(N)
    h4[:25] = A
    h4[25:50] = B
    h4[50:75] = A
    h4[75:] = B
    candidates.append(h4)
    
    # D: Random smooth
    np.random.seed(42)
    h_rand = 0.5 + 0.2 * np.sin(np.random.uniform(0, 2 * np.pi) + np.arange(N) * 0.1)
    h_rand = np.clip(h_rand, 0, 1)
    h_rand = _normalize(h_rand)
    candidates.append(h_rand)
    
    best_h = candidates[0].copy()
    best_score = _score(best_h)
    
    # 2. Optimization Loop
    # Use SLSQP with smooth max objective
    # Objective: smooth_max(overlap)
    # Constraints: sum(h) = N/2, 0 <= h <= 1
    
    def _objective_smooth(params, alpha):
        h = params
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        return _smooth_max(overlap, alpha=alpha)
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]
    bounds = [(0.0, 1.0)] * N
    
    for i, h_init in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        try:
            # First pass: Coarse optimization
            result = minimize(
                lambda x: _objective_smooth(x, alpha=50.0),
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.fun < 1.0: # Sanity check
                h_opt = _normalize(result.x)
                s_opt = _score(h_opt)
                if s_opt < best_score:
                    best_h = h_opt
                    best_score = s_opt
                    
            # Second pass: Fine polish on the best candidate found so far
            # We only do the fine polish on the current best to save time
            # Or do it on the result of this specific candidate if it improved?
            # Let's accumulate best_h and do a final global polish at the end.
            
        except Exception:
            pass
            
    # 3. Final Global Polish on the best candidate
    if time.time() < deadline - 5.0:
        try:
            # Use a higher alpha for sharper max approximation
            result_polish = minimize(
                lambda x: _objective_smooth(x, alpha=200.0),
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            
            if result_polish.fun < 1.0:
                h_polished = _normalize(result_polish.x)
                s_polished = _score(h_polished)
                if s_polished < best_score:
                    best_h = h_polished
                    best_score = s_polished
                    
        except Exception:
            pass
            
    return best_h
