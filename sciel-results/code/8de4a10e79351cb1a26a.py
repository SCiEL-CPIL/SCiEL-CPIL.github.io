"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import reduce

_divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_n = len(_divisors)
_max_val = 1000000
_start_time = time.time()
_deadline = _start_time + 85.0  # 5s safety margin before 90s kill

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_lcm_all = reduce(_lcm, _divisors)

def _solve_window(L, W):
    """Try to assign distinct multiples in [L, L+W]. Returns list of ints or None."""
    U = L + W
    if U > _max_val or U < 1:
        return None
    
    candidates = []
    for d in _divisors:
        k_min = max(1, (L + d - 1) // d)
        k_max = U // d
        if k_max < k_min:
            return None
        cands = [k * d for k in range(k_min, k_max + 1)]
        candidates.append(cands)
    
    order = sorted(range(_n), key=lambda i: len(candidates[i]))
    used = set()
    assignment = [None] * _n
    
    def dfs(idx):
        if idx == _n:
            return True
        i = order[idx]
        for v in candidates[i]:
            if v not in used:
                used.add(v)
                assignment[i] = v
                if dfs(idx + 1):
                    return True
                used.discard(v)
                assignment[i] = None
        return False
    
    if dfs(0):
        return [int(assignment[i]) for i in range(_n)]
    return None

def construct():
    best_result = None
    best_raw = float('inf')
    
    # Strategy: exploit periodicity. The divisibility pattern repeats every _lcm_all.
    # For each offset r in [0, _lcm_all), check if there exists a window of width W
    # starting at some L ≡ r (mod _lcm_all) that is feasible.
    # 
    # Key insight: if a window [L, L+W] is feasible, then [L + _lcm_all, L + W + _lcm_all]
    # is also feasible (all multiples shift by _lcm_all). So we only need to check one
    # representative per residue class, and the optimal window will be near the start
    # of the valid range for that residue.
    #
    # For each residue r, the earliest valid L is max(1, r) if r >= 1, or 1 if r == 0.
    # But we need L >= 1 (positive integers). So L_min(r) = r if r >= 1, else 1.
    # Actually L must be >= 1, and L ≡ r (mod _lcm_all).
    # If r == 0, L can be _lcm_all, 2*_lcm_all, ... (since L must be positive)
    # If r >= 1, L can be r, r + _lcm_all, r + 2*_lcm_all, ...
    
    # For each residue, binary search on W starting from a lower bound.
    # Lower bound for W: at least _n - 1 = 9 (need 10 distinct values).
    # Upper bound: _max_val - L_min.
    
    # We'll iterate over all residues r in [0, _lcm_all) and for each, try to find
    # the minimal W such that the window [L, L+W] is feasible, where L is the
    # smallest positive integer ≡ r (mod _lcm_all).
    
    # But _lcm_all = 32730, so we have 32730 residues. For each, we do a binary
    # search on W (log2(1000000) ≈ 20 iterations), and each feasibility check is
    # a DFS. This might be too slow.
    #
    # Optimization: many residues will have the same structure. Also, we can
    # prune: if for a given residue, the minimal feasible W is already >= best_raw,
    # skip it.
    #
    # Better approach: Instead of checking all 32730 residues, note that the
    # "tightness" of the constraint depends on which multiples fall in the window.
    # The critical factor is the spacing of multiples. Let's try a smarter search:
    #
    # 1. Start with a small W (say 9, 10, 11, ...) and for each W, check if there
    #    exists ANY L such that [L, L+W] is feasible.
    # 2. For a fixed W, we can check feasibility by trying all residues r and
    #    seeing if the window starting at the earliest L for that residue works.
    #
    # Actually, let's just do a direct search over small W values first, since
    # the answer is likely small.
    
    # Phase 1: Try small W values directly
    for W in range(9, 200):
        if time.time() > _deadline:
            break
        found = False
        # For this W, try various L values
        # L must be such that L + W <= _max_val and L >= 1
        # Try L values that are "aligned" with divisor patterns
        # Key: L should be chosen so that the window contains good multiples
        
        # Try L = 1, 2, ..., up to some limit, and also L aligned with divisors
        L_tried = set()
        
        # Try small L values
        for L in range(1, min(1000, _max_val - W + 1)):
            if time.time() > _deadline:
                break
            if L in L_tried:
                continue
            L_tried.add(L)
            result = _solve_window(L, W)
            if result is not None:
                raw = max(result) - min(result)
                if raw <= W and raw < best_raw:
                    best_raw = raw
                    best_result = result[:]
                    found = True
        
        # Also try L values that are multiples of small divisors
        for d in _divisors:
            for k in range(1, min(100, (_max_val - W) // d + 1)):
                L = k * d
                if L > _max_val - W or L < 1:
                    continue
                if L in L_tried:
                    continue
                L_tried.add(L)
                if time.time() > _deadline:
                    break
                result = _solve_window(L, W)
                if result is not None:
                    raw = max(result) - min(result)
                    if raw <= W and raw < best_raw:
                        best_raw = raw
                        best_result = result[:]
                        found = True
        
        # If we found a solution with raw == W, we can't do better for this W
        # But we should continue to check smaller W... wait, we're going up in W.
        # Actually we want the minimum W, so if we find a solution at W, we should
        # stop only if we've verified no smaller W works. Since we're iterating W
        # from 9 upward, the first W where we find a solution is optimal.
        if found and best_raw <= W:
            # We found a solution; break if this is the minimal possible
            # But we should verify by checking if best_raw < W (which would mean
            # the actual span is less than W)
            if best_raw == W:
                break
    
    # Phase 2: If phase 1 didn't find a good solution, use periodicity-based search
    if best_raw > 50:
        # For each residue r in [0, _lcm_all), find the minimal W
        # Use the fact that the pattern repeats every _lcm_all
        for r in range(0, _lcm_all):
            if time.time() > _deadline:
                break
            
            # Earliest positive L with L ≡ r (mod _lcm_all)
            if r == 0:
                L_min = _lcm_all
            else:
                L_min = r
            
            if L_min > _max_val:
                continue
            
            # Binary search on W for this residue
            lo = 9
            hi = _max_val - L_min
            if hi < lo:
                continue
            
            # Quick check: is hi feasible?
            result = _solve_window(L_min, hi)
            if result is None:
                continue
            
            best_w_for_r = hi
            while lo < hi:
                if time.time() > _deadline:
                    break
                mid = (lo + hi) // 2
                result = _solve_window(L_min, mid)
                if result is not None:
                    best_w_for_r = mid
                    hi = mid
                else:
                    lo = mid + 1
            
            if best_w_for_r < best_raw:
                result = _solve_window(L_min, best_w_for_r)
                if result is not None:
                    raw = max(result) - min(result)
                    if raw < best_raw:
                        best_raw = raw
                        best_result = result[:]
    
    if best_result is None:
        # Fallback
        return [int(d) for d in _divisors]
    
    return best_result
# EVOLVE-BLOCK-END
