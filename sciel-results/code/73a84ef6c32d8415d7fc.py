"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import reduce

_divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_n = len(_divisors)
_max_val = 1000000
_deadline = time.time() + 85.0  # 5s safety margin before 90s kill

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_lcm_all = reduce(_lcm, _divisors)

def _feasible(L, W):
    """Check if window [L, L+W] contains distinct multiples for all divisors."""
    U = L + W
    if U > _max_val:
        return False
    
    # Build candidate lists for each divisor
    candidates = []
    for d in _divisors:
        k_min = (L + d - 1) // d
        k_max = U // d
        if k_max < k_min:
            return False
        # Store as list of multiples
        cands = [k * d for k in range(k_min, k_max + 1)]
        candidates.append(cands)
    
    # Sort divisors by number of candidates (ascending) for MRV
    order = sorted(range(_n), key=lambda i: len(candidates[i]))
    
    used = set()
    
    def dfs(idx):
        if idx == _n:
            return True
        i = order[idx]
        for v in candidates[i]:
            if v not in used:
                used.add(v)
                if dfs(idx + 1):
                    return True
                used.discard(v)
        return False
    
    return dfs(0)

def _find_min_W(L):
    """Binary search for minimal W such that window [L, L+W] is feasible."""
    # Lower bound: at least _n - 1 (need distinct values)
    # Upper bound: _max_val - L
    lo = 0
    hi = _max_val - L
    if hi < 0:
        return None
    if not _feasible(L, hi):
        return None
    
    best_w = hi
    while lo < hi:
        if time.time() > _deadline:
            break
        mid = (lo + hi) // 2
        if _feasible(L, mid):
            best_w = mid
            hi = mid
        else:
            lo = mid + 1
    return best_w

def construct():
    best_result = None
    best_raw = float('inf')
    
    # Try multiple L values
    # L should be such that L + W <= _max_val
    # Good candidates: multiples of small numbers, and 0
    # The optimal L is likely near 0 or aligned with divisor patterns
    
    # Generate candidate L values
    L_candidates = set()
    L_candidates.add(0)
    # Add multiples of each divisor up to some range
    for d in _divisors:
        for k in range(0, 20):
            L_candidates.add(k * d)
    # Add some random-ish offsets
    for offset in range(0, 100):
        L_candidates.add(offset)
    # Add multiples of LCM (small range)
    for k in range(0, 5):
        L_candidates.add(k * _lcm_all)
    
    # Sort by L (smaller L allows larger W, but we want minimal W)
    L_list = sorted(L_candidates)
    
    for L in L_list:
        if time.time() > _deadline:
            break
        if L > _max_val:
            break
        W = _find_min_W(L)
        if W is not None:
            # Now find the actual assignment for this (L, W)
            U = L + W
            candidates = []
            for d in _divisors:
                k_min = (L + d - 1) // d
                k_max = U // d
                cands = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(cands)
            
            order = sorted(range(_n), key=lambda i: len(candidates[i]))
            used = set()
            assignment = [None] * _n
            
            def dfs_assign(idx):
                if idx == _n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used:
                        used.add(v)
                        assignment[i] = v
                        if dfs_assign(idx + 1):
                            return True
                        used.discard(v)
                        assignment[i] = None
                return False
            
            if dfs_assign(0):
                result = [int(assignment[i]) for i in range(_n)]
                raw = max(result) - min(result)
                if raw < best_raw:
                    best_raw = raw
                    best_result = result[:]
    
    if best_result is None:
        # Fallback: just return the divisors themselves
        return [int(d) for d in _divisors]
    
    return best_result
# EVOLVE-BLOCK-END
