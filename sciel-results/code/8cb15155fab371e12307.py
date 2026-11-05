"""Low discrepancy on all arithmetic progressions - Greedy Steepest Descent with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    # We iterate over step sizes d.
    # For a given d, the progressions are determined by start index i (0 <= i < d) 
    # and length L.
    # An AP starting at i with step d is s[i], s[i+d], s[i+2d], ...
    # We want max |sum_{k=0}^{L-1} s[i + k*d]| for all valid i, L.
    
    for d in range(1, n + 1):
        for i in range(d):
            if i >= n: break
            seq = s[i:n:d]
            if len(seq) == 0: continue
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
    
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe against 90s kill
    
    n = 256
    
    # Generate a few diverse initial candidates
    # 1. Random
    # 2. Thue-Morse (low discrepancy for small steps)
    # 3. Rademacher sequence (deterministic pseudo-random)
    
    candidates = []
    
    # Candidate 1: Random
    rng = np.random.default_rng(42)
    c1 = rng.choice([-1, 1], size=n).astype(np.int16)
    candidates.append(c1)
    
    # Candidate 2: Thue-Morse
    # t(n) = parity of number of 1s in binary representation of n
    # Map 0 -> 1, 1 -> -1
    c2 = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int16)
    candidates.append(c2)
    
    # Candidate 3: Rademacher (deterministic)
    c3 = np.array([1 if (i % 4 < 2) else -1 for i in range(n)], dtype=np.int16)
    candidates.append(c3)
    
    # Candidate 4: Random with different seed
    rng2 = np.random.default_rng(123)
    c4 = rng2.choice([-1, 1], size=n).astype(np.int16)
    candidates.append(c4)
    
    best_signs = None
    best_disc = float('inf')
    
    for init_signs in candidates:
        if time.time() > deadline:
            break
            
        current_signs = init_signs.copy()
        current_disc = _max_ap_discrepancy(current_signs)
        
        # Greedy Steepest Descent
        # In each iteration, try flipping every bit and pick the one that minimizes discrepancy.
        # If no flip improves, stop (local minimum).
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            best_flip_idx = -1
            best_new_disc = current_disc
            
            # Evaluate all possible single-bit flips
            # To save time, we can break early if we find a flip that reduces discrepancy significantly?
            # No, we need the steepest descent, so we must check all.
            # However, n=256. 256 evaluations per iteration.
            # Each evaluation is ~1-2ms.
            # 256 * 2ms = 0.5s per iteration.
            # This might be too slow for many iterations.
            # Let's estimate: 85s / 0.5s = 170 iterations per start.
            # 4 starts * 170 = 680 iterations.
            # This is reasonable.
            
            for i in range(n):
                # Flip bit i
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                new_disc = _max_ap_discrepancy(current_signs)
                
                if new_disc < best_new_disc:
                    best_new_disc = new_disc
                    best_flip_idx = i
                elif new_disc == best_new_disc and best_flip_idx == -1:
                    # Tie-breaking: keep first found or random?
                    # Let's keep the first one to be deterministic
                    best_flip_idx = i
                    
                # Revert
                current_signs[i] = old_sign
            
            if best_flip_idx != -1 and best_new_disc < current_disc:
                # Apply the best flip
                current_signs[best_flip_idx] = -current_signs[best_flip_idx]
                current_disc = best_new_disc
            else:
                # No improvement found, local minimum
                improved = False
        
        # Check if this local minimum is better than global best
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = current_signs.copy()
            
        # If we have plenty of time left and found a good solution, we could try to restart from a perturbed version of the best solution?
        # For now, let's just run the next candidate.
        
    if best_signs is None:
        # Fallback: return random
        return rng.choice([-1, 1], size=n).astype(int).tolist()
        
    return best_signs.tolist()

# EVOLVE-BLOCK-END
