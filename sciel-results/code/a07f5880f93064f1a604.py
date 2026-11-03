"""Optimize over geometric-progressions U = {floor(b^k)} to maximize the
sumset/difference-set exponent lower bound."""
import math
import time
import itertools

# EVOLVE-BLOCK-START
def _score_of(U):
    """Return the raw lower-bound score for a concrete set U (must contain 0)."""
    if not U:
        return 0.0
    u = sorted(set(U))
    if 0 not in u:
        u.append(0)
    n = len(u)
    if n < 2:
        return 0.0
    mx = u[-1]
    if mx <= 0:
        return 0.0
    # difference set U - U
    diffs = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
    # sum set U + U
    sums = set()
    for a in u:
        for b in u:
            sums.add(a + b)
    denom = 2.0 * mx + 1.0
    if denom <= 1.0:
        return 0.0
    ratio = len(diffs) / float(len(sums))
    if ratio <= 0.0:
        return 0.0
    return 1.0 + (math.log(ratio) / math.log(denom))

def _build_geom(b, m):
    """Construct U = {floor(b^k) | k=0..m} as a list containing 0."""
    U = [0]
    v = 1.0
    for k in range(1, m + 1):
        v *= b
        U.append(int(v))
    return U

def construct():
    deadline = time.time() + 110.0
    best_score = 0.0
    best_set = [0, 1]
    
    # Search over geometric progressions U = {floor(b^k) | k=0..m}
    # b in [1.01, 5.0], m in [2, 80]
    
    # Coarse grid search
    b_start = 1.01
    b_end = 5.0
    b_step = 0.05
    
    m_max = 80
    
    current_b = b_start
    while current_b <= b_end:
        if time.time() > deadline - 20: # Leave time for refinement
            break
            
        # For this b, find best m
        best_m_score = 0.0
        best_m_set = None
        
        # m from 2 to m_max
        for m in range(2, m_max + 1):
            U = _build_geom(current_b, m)
            # Optimization: if max U is very small, score might be low?
            # Just compute.
            s = _score_of(U)
            if s > best_m_score:
                best_m_score = s
                best_m_set = U
                
        if best_m_score > best_score:
            best_score = best_m_score
            best_set = best_m_set
            
        current_b += b_step
        
    # Refinement
    # We need to know which b was best. Let's track it.
    # Since I didn't track it in the loop above, I will re-run a fine grid 
    # over the whole range but with smaller step, or just assume the coarse 
    # found a good region. 
    # Better: track best_b in the loop.
    
    # Let's restart the logic to track best_b properly.
    
    best_b = 0.0
    
    current_b = b_start
    while current_b <= b_end:
        if time.time() > deadline - 20:
            break
            
        best_m_score = 0.0
        best_m_set = None
        
        for m in range(2, m_max + 1):
            U = _build_geom(current_b, m)
            s = _score_of(U)
            if s > best_m_score:
                best_m_score = s
                best_m_set = U
                
        if best_m_score > best_score:
            best_score = best_m_score
            best_set = best_m_set
            best_b = current_b
            
        current_b += b_step
        
    # Fine search around best_b
    if time.time() < deadline - 5 and best_b > 0:
        b_start_fine = max(1.01, best_b - 0.1)
        b_end_fine = min(5.0, best_b + 0.1)
        b_step_fine = 0.005
        
        current_b = b_start_fine
        while current_b <= b_end_fine:
            if time.time() > deadline - 2:
                break
            
            best_m_score = 0.0
            best_m_set = None
            
            for m in range(2, m_max + 1):
                U = _build_geom(current_b, m)
                s = _score_of(U)
                if s > best_m_score:
                    best_m_score = s
                    best_m_set = U
                    
            if best_m_score > best_score:
                best_score = best_m_score
                best_set = best_m_set
                best_b = current_b
                
            current_b += b_step_fine

    return best_set
# EVOLVE-BLOCK-END
