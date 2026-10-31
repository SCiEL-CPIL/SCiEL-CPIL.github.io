import time
import numpy as np
import math

_start_time = None
_DEADLINE = 330 * 0.95

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # 1. Spectral Score (Randomness/Flatness)
    # Convert to numpy array
    v = np.array(vector, dtype=np.float64)
    
    # DFT
    # For n=1, DFT is trivial.
    if n == 1:
        spectral_score = 0.0
    else:
        F = np.fft.fft(v)
        P = np.abs(F)**2
        # Normalize
        total_power = np.sum(P)
        if total_power > 0:
            P_norm = P / total_power
        else:
            P_norm = np.ones(n) / n
            
        # Entropy
        # Avoid log(0)
        P_nonzero = P_norm[P_norm > 0]
        if len(P_nonzero) > 0:
            entropy = -np.sum(P_nonzero * np.log(P_nonzero))
            max_entropy = math.log(n)
            spectral_score = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            spectral_score = 0.0

    # 2. Symbol Balance Score
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    sq_sum = c0*c0 + c1*c1 + c2*c2
    # Higher is better if spread out
    balance_score = n - sq_sum / n
    
    # 3. Mod-3 Sum Penalty
    coord_sum = c1 + 2*c2
    sum_mod = coord_sum % 3
    # Penalize 0
    if sum_mod == 0:
        sum_score = -100.0
    else:
        sum_score = 0.0
        
    # 4. Hash Tie-breaker
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combine
    # Weights: Spectral is primary
    total = 5000.0 * spectral_score + 100.0 * balance_score + sum_score + hash_score
    return float(total)
