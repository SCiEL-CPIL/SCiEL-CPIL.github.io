"""Optimized B&B solver for Large Reciprocal Weight under LCM Separation."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    n_max = 256
    # Structural insight:
    # 1. All numbers in [129, 256] are pairwise compatible (LCM > 256).
    # 2. No number in [2, 128] can be included if ALL numbers in [129, 256] are included.
    #    Specifically, any a in [2, 128] divides some b in [129, 256].
    #    Thus, we must trade off: include some small numbers (high weight) but exclude their multiples in large set (low weight loss).
    # 3. We restrict small candidates to [2, 64] because [65, 128] are "dominated" or just part of the small set logic.
    #    Actually, we established [65, 128] cannot be in solution if we keep many larges.
    #    But we can just search over [2, 64] as the "small" set, and assume all other larges are included unless excluded.
    
    small = list(range(2, 65))
    large = list(range(129, 257))
    n_small = len(small)
    n_large = len(large)
    
    # Precompute weights
    sw_frac = [Fraction(1, v) for v in small]
    lw_frac = [Fraction(1, v) for v in large]
    sw_float = [float(w) for w in sw_frac]
    lw_float = [float(w) for w in lw_frac]
    
    # Precompute large set sum
    total_large_frac = sum(lw_frac)
    total_large_float = sum(lw_float)
    
    # Precompute conflict matrix for small-small
    # conflict[i] is bitmask of small indices j such that lcm(small[i], small[j]) <= 256
    ss_conflict = [0] * n_small
    for i in range(n_small):
        for j in range(i + 1, n_small):
            if lcm(small[i], small[j]) <= n_max:
                ss_conflict[i] |= (1 << j)
                ss_conflict[j] |= (1 << i)
    
    # Precompute for each small i, the mask of large indices j such that large[j] % small[i] == 0
    sl_conflict = [0] * n_small
    for i, s in enumerate(small):
        mask = 0
        for j, L in enumerate(large):
            if L % s == 0:
                mask |= (1 << j)
        sl_conflict[i] = mask
    
    # Suffix sums of small weights for bounding
    suffix_sw_float = [0.0] * (n_small + 1)
    for i in range(n_small - 1, -1, -1):
        suffix_sw_float[i] = suffix_sw_float[i + 1] + sw_float[i]
    
    # Initial best: All large, no small
    best_score = total_large_frac
    best_small_mask = 0
    best_large_excluded = 0 # 0 means none excluded, so all included
    
    # Greedy initialization to get a better lower bound
    # Try adding small numbers one by one if they improve the score
    def greedy_init():
        current_mask = 0
        current_excluded = 0
        current_score = total_large_frac
        # Sort by weight descending
        indices = sorted(range(n_small), key=lambda i: -sw_float[i])
        for i in indices:
            if (current_mask & ss_conflict[i]) != 0:
                continue
            new_excluded = current_excluded | sl_conflict[i]
            # Calculate new large score
            # We can approximate with float for speed in init
            loss = 0.0
            # To be exact, we'd use Fraction, but for init float is ok
            # We'll just use the float bound for now
            # Actually, let's just do a simple check
            if sw_float[i] > 0.01: # Heuristic: only try if weight is significant
                # Estimate loss
                # This is complex, so we just rely on B&B
                pass
        return best_score, best_small_mask, best_large_excluded
    
    # B&B Solver
    import sys
    sys.setrecursionlimit(10000)
    
    def solve_bnb():
        nonlocal best_score, best_small_mask, best_large_excluded, deadline
        
        def dfs(pos, selected_mask, excluded_large, current_score_frac, current_score_float, current_large_float):
            nonlocal best_score, best_small_mask, best_large_excluded, deadline
            
            if time.time() > deadline:
                return
            
            if pos == n_small:
                # Calculate exact large score
                large_frac = Fraction(0)
                for j in range(n_large):
                    if not (excluded_large >> j) & 1:
                        large_frac += lw_frac[j]
                total = current_score_frac + large_frac
                if total > best_score:
                    best_score = total
                    best_small_mask = selected_mask
                    best_large_excluded = excluded_large
                return
            
            # Upper Bound
            # Current score + max possible gain from remaining smalls + current large score
            # Note: current_large_float is the score of large numbers NOT yet excluded by selected smalls.
            # Adding more smalls will only decrease the large score.
            # So UB = current_score_float + suffix_sw_float[pos] + current_large_float
            ub_float = current_score_float + suffix_sw_float[pos] + current_large_float
            
            if ub_float <= float(best_score) + 1e-12:
                return
            
            # Option 1: Include small[pos] if compatible
            if (selected_mask & ss_conflict[pos]) == 0:
                new_mask = selected_mask | (1 << pos)
                new_excluded = excluded_large | sl_conflict[pos]
                
                # Calculate loss in large score
                # We need sum of lw_float[j] for j in sl_conflict[pos] that are NOT already in excluded_large
                loss = 0.0
                # Optimization: precompute prefix sums? No, mask is sparse.
                # Just iterate over bits in sl_conflict[pos]
                mask_to_check = sl_conflict[pos] & ~excluded_large
                while mask_to_check:
                    lsb = mask_to_check & -mask_to_check
                    idx = (lsb.bit_length() - 1)
                    loss += lw_float[idx]
                    mask_to_check ^= lsb
                
                new_large_float = current_large_float - loss
                new_score_float = current_score_float + sw_float[pos]
                
                dfs(pos + 1, new_mask, new_excluded, current_score_frac + sw_frac[pos], new_score_float, new_large_float)
            
            # Option 2: Exclude small[pos]
            dfs(pos + 1, selected_mask, excluded_large, current_score_frac, current_score_float, current_large_float)
        
        dfs(0, 0, 0, Fraction(0), 0.0, total_large_float)
    
    solve_bnb()
    
    # Build final set
    result = []
    for i in range(n_small):
        if (best_small_mask >> i) & 1:
            result.append(small[i])
    for j in range(n_large):
        if not (best_large_excluded >> j) & 1:
            result.append(large[j])
    
    if not result:
        return [2]
        
    return result
# EVOLVE-BLOCK-END
