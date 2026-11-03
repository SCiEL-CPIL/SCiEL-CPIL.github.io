"""Optimize over geometric-progressions U = {floor(b^k)} to maximize the
sumset/difference-set exponent lower bound."""
import math
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _score_of_np(U_np):
    """Return the raw lower-bound score for a concrete set U (numpy array) containing 0."""
    if len(U_np) == 0:
        return 0.0
    u = np.unique(U_np)
    # Ensure 0 is included
    if u[0] != 0:
        u = np.concatenate(([0], u))
    n = len(u)
    if n < 2:
        return 0.0
    mx = u[-1]
    if mx <= 0:
        return 0.0
    
    # Compute difference set U - U
    diffs = u[:, None] - u[None, :]
    diffs_flat = diffs.ravel()
    num_diffs = len(np.unique(diffs_flat))
    
    # Compute sum set U + U
    sums = u[:, None] + u[None, :]
    sums_flat = sums.ravel()
    num_sums = len(np.unique(sums_flat))
    
    denom = 2.0 * mx + 1.0
    if denom <= 1.0:
        return 0.0
    ratio = num_diffs / float(num_sums)
    if ratio <= 0.0:
        return 0.0
    return 1.0 + (math.log(ratio) / math.log(denom))

def _build_np(b, m):
    """Construct U = {0} U {floor(b^k) | k=1..m} as a numpy array."""
    if m == 0:
        return np.array([0], dtype=np.int64)
    k = np.arange(1, m + 1, dtype=np.float64)
    vals = np.floor(np.power(b, k)).astype(np.int64)
    # Add 0
    u = np.unique(np.concatenate(([0], vals)))
    return u

def construct():
    deadline = time.time() + 100.0
    best_score = 0.0
    best_set = [0, 1, 2, 4, 8]
    
    # Search over m (number of terms in GP excluding 0? No, k=1..m, so m terms + 0)
    # We found that small m with b~2 is good.
    # Let's search m in [2, 50] and b in [1.2, 5.0]
    
    m_values = list(range(2, 21)) + [30, 40, 50]
    
    for m in m_values:
        if time.time() > deadline - 5.0:
            break
            
        # Coarse grid search for b
        # b range: 1.2 to 5.0
        # 40 points
        b_grid = np.linspace(1.2, 5.0, 40)
        best_b_for_m = 1.2
        best_score_for_m = 0.0
        
        for b in b_grid:
            if time.time() > deadline - 2.0:
                break
            U_np = _build_np(b, m)
            s = _score_of_np(U_np)
            if s > best_score_for_m:
                best_score_for_m = s
                best_b_for_m = b
        
        # Refine with ternary search
        if best_score_for_m > 0:
            # Local search around best_b_for_m
            # Step size 0.1
            search_lo = max(1.05, best_b_for_m - 0.2)
            search_hi = min(6.0, best_b_for_m + 0.2)
            
            # Ternary search
            for _ in range(15):
                if time.time() > deadline - 1.0:
                    break
                m1 = search_lo + (search_hi - search_lo) / 3.0
                m2 = search_hi - (search_hi - search_lo) / 3.0
                
                U1 = _build_np(m1, m)
                s1 = _score_of_np(U1)
                
                U2 = _build_np(m2, m)
                s2 = _score_of_np(U2)
                
                if s1 < s2:
                    search_lo = m1
                else:
                    search_hi = m2
                    
            # Check center
            b_opt = (search_lo + search_hi) / 2.0
            U_opt = _build_np(b_opt, m)
            s_opt = _score_of_np(U_opt)
            
            if s_opt > best_score_for_m:
                best_score_for_m = s_opt
                best_b_for_m = b_opt
            
            if best_score_for_m > best_score:
                best_score = best_score_for_m
                best_set = [int(x) for x in _build_np(best_b_for_m, m)]
                
    return best_set
# EVOLVE-BLOCK-END
