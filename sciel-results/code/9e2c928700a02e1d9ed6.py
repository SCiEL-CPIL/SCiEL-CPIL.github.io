"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

_START = time.time()

def _elapsed():
    return time.time() - _START

# Precompute AP indices for N=128
_N = 128
_AP_A = []
_AP_D = []
for a in range(_N):
    max_d = (_N - 1 - a) // 3
    for d in range(1, max_d + 1):
        if a + 3 * d < _N:
            _AP_A.append(a)
            _AP_D.append(d)
_AP_A = np.array(_AP_A)
_AP_D = np.array(_AP_D)
_IDX0 = _AP_A
_IDX1 = _AP_A + _AP_D
_IDX2 = _AP_A + 2 * _AP_D
_IDX3 = _AP_A + 3 * _AP_D

def _count_violations_fast(colors_np):
    """
    Returns the number of 4-term APs with fewer than 3 distinct colors.
    colors_np: np.array of int
    """
    c0 = colors_np[_IDX0]
    c1 = colors_np[_IDX1]
    c2 = colors_np[_IDX2]
    c3 = colors_np[_IDX3]
    
    eq_01 = (c0 == c1).astype(np.int8)
    eq_02 = (c0 == c2).astype(np.int8)
    eq_03 = (c0 == c3).astype(np.int8)
    eq_12 = (c1 == c2).astype(np.int8)
    eq_13 = (c1 == c3).astype(np.int8)
    eq_23 = (c2 == c3).astype(np.int8)
    
    pair_count = eq_01 + eq_02 + eq_03 + eq_12 + eq_13 + eq_23
    
    # Invalid if pair_count >= 2
    violations = np.sum(pair_count >= 2)
    return int(violations)

def _is_valid_fast(colors_np):
    return _count_violations_fast(colors_np) == 0

def _num_colors(colors_np):
    return len(np.unique(colors_np))

def _local_search_fast(colors_np, K, max_iter=50000):
    """
    Local search to reduce violations to 0.
    Returns (colors_np, violations)
    """
    current_violations = _count_violations_fast(colors_np)
    if current_violations == 0:
        return colors_np, 0
    
    best_colors = colors_np.copy()
    best_violations = current_violations
    
    for i in range(max_iter):
        if _elapsed() > 55:
            break
        if best_violations == 0:
            break
        
        # Pick a random position
        pos = random.randint(0, _N - 1)
        old_color = colors_np[pos]
        new_color = random.randint(0, K - 1)
        if new_color == old_color:
            continue
        
        colors_np[pos] = new_color
        new_violations = _count_violations_fast(colors_np)
        
        if new_violations < best_violations:
            best_violations = new_violations
            best_colors = colors_np.copy()
        elif new_violations == best_violations:
            # Accept neutral moves to escape local minima
            if random.random() < 0.1:
                best_violations = new_violations
                best_colors = colors_np.copy()
            else:
                colors_np[pos] = old_color
        else:
            colors_np[pos] = old_color
            
    return best_colors, best_violations

def construct():
    global _START
    _START = time.time()
    n = 128
    
    best_result = None
    best_nc = float('inf')
    
    # Try to find a valid coloring with small K
    for K in range(3, 8):
        if _elapsed() > 50:
            break
            
        # Multiple restarts for each K
        for attempt in range(5):
            if _elapsed() > 50:
                break
                
            colors = np.array([random.randint(0, K - 1) for _ in range(n)], dtype=np.int32)
            colors, violations = _local_search_fast(colors, K, max_iter=20000)
            
            if violations == 0:
                nc = _num_colors(colors)
                if nc < best_nc:
                    best_nc = nc
                    best_result = colors.tolist()
                    # If we found a solution with 3 colors, that's likely optimal or near-optimal
                    if nc == 3:
                        return best_result
                        
    if best_result is not None:
        return best_result
        
    # Fallback: Deterministic construction
    # Try 2-adic valuation
    colors = []
    for x in range(1, n + 1):
        v = 0
        temp = x
        while temp % 2 == 0:
            v += 1
            temp //= 2
        colors.append(v)
    colors_np = np.array(colors, dtype=np.int32)
    if _is_valid_fast(colors_np):
        return colors
        
    # Try modulo 7
    colors = [x % 7 for x in range(1, n + 1)]
    colors_np = np.array(colors, dtype=np.int32)
    if _is_valid_fast(colors_np):
        return colors
        
    # Last resort: Random with larger K, just to return *something* valid
    for K in range(8, 15):
        for _ in range(10):
            if _elapsed() > 58:
                break
            colors = np.array([random.randint(0, K - 1) for _ in range(n)], dtype=np.int32)
            if _is_valid_fast(colors):
                return colors.tolist()
                
    # If all else fails, return a high-color solution (e.g., identity)
    # Identity uses 128 colors, which is valid?
    # If c(i) = i, then c(a), c(a+d), c(a+2d), c(a+3d) are distinct if d!=0.
    # So yes, identity is valid.
    return list(range(n))

# EVOLVE-BLOCK-END
