"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _precompute_aps(n=128):
    """Precompute indices of all 4-term arithmetic progressions."""
    ap_list = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 >= n:
                break
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    return np.array(ap_list, dtype=np.int32)

def _count_bad(colors, aps):
    """Count number of 4-APs with < 3 distinct colors."""
    c0 = colors[aps[:, 0]]
    c1 = colors[aps[:, 1]]
    c2 = colors[aps[:, 2]]
    c3 = colors[aps[:, 3]]
    
    # Pairwise equalities
    e01 = (c0 == c1)
    e02 = (c0 == c2)
    e03 = (c0 == c3)
    e12 = (c1 == c2)
    e13 = (c1 == c3)
    e23 = (c2 == c3)
    
    sum_e = e01 + e02 + e03 + e12 + e13 + e23
    
    # Size < 3 iff sum_e >= 2
    bad = sum_e >= 2
    return np.sum(bad)

def _solve_local_search(k, n=128, aps=None, time_limit=5.0):
    """Try to find a valid k-coloring using local search."""
    if aps is None:
        aps = _precompute_aps(n)
        
    start_time = time.time()
    best_bad = float('inf')
    best_colors = None
    
    # Try a few random restarts
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        colors = np.array([random.randint(0, k - 1) for _ in range(n)], dtype=np.int32)
        current_bad = _count_bad(colors, aps)
        
        if current_bad == 0:
            return colors.tolist()
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.99
        min_T = 0.01
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
                
            if current_bad == 0:
                return colors.tolist()
                
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Try all possible new colors
            best_new_color = old_color
            best_new_bad = current_bad
            
            # Optimization: Only try colors that might help
            # But for small k, trying all is fine
            for nc in range(k):
                if nc == old_color:
                    continue
                colors[pos] = nc
                new_bad = _count_bad(colors, aps)
                if new_bad < best_new_bad:
                    best_new_bad = new_bad
                    best_new_color = nc
            colors[pos] = old_color
            
            if best_new_color != old_color:
                # Accept if better, or with probability exp(-delta/T) if worse
                delta = best_new_bad - current_bad
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[pos] = best_new_color
                    current_bad = best_new_bad
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_colors = colors.copy()
            else:
                # No improvement found in this step, cool down faster?
                pass
                
            T *= cooling
            
    return best_colors.tolist() if best_colors is not None else None

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    n = 128
    
    aps = _precompute_aps(n)
    
    # Try to find a solution with small number of colors
    for k in range(3, 43):
        if time.time() > deadline:
            break
            
        # Allocate time for this k
        # Give more time to smaller k? Or equal time?
        # Let's give 2 seconds per k, or scale based on remaining time.
        time_per_k = 2.0
        
        result = _solve_local_search(k, n, aps, time_limit=time_per_k)
        
        if result is not None:
            # Verify
            if _count_bad(np.array(result, dtype=np.int32), aps) == 0:
                return result
                
    # Fallback to modular 43
    colors = [x % 43 for x in range(1, n + 1)]
    return colors
# EVOLVE-BLOCK-END
