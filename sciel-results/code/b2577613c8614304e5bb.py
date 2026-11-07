"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = 30030
_MAX_VAL = 1000000
_N = 10

# Precompute residues for each divisor
# residues[i] is a list of positions p in [0, LCM) such that p % _DIVISORS[i] == 0
_residues = []
for d in _DIVISORS:
    res = []
    for p in range(0, _LCM, d):
        res.append(p)
    _residues.append(res)

# Sort divisors and their residues by scarcity (largest divisors have fewer residues)
# This helps in backtracking to prune early
_indices = sorted(range(_N), key=lambda i: len(_residues[i]), reverse=True)
# Wait, sorting by scarcity means we want to pick the one with FEWEST options first.
# So sort by length of residue list ASCENDING.
_indices = sorted(range(_N), key=lambda i: len(_residues[i]))
_sorted_divisors = [_DIVISORS[i] for i in _indices]
_sorted_residues = [_residues[i] for i in _indices]
# We need to map back to original indices for the result
_inv_indices = [0] * _N
for new_idx, orig_idx in enumerate(_indices):
    _inv_indices[orig_idx] = new_idx

def _check_window(W):
    """
    Check if there is a window of length W that allows distinct assignment.
    Returns (offset, assignment_list) where assignment_list is in original divisor order.
    """
    # We iterate through all possible offsets S in [0, LCM)
    # Optimization: We can skip offsets that are "equivalent" in terms of coverage?
    # Not easily. But we can make the inner check very fast.
    
    for S in range(_LCM):
        # For each divisor, find candidates in [S, S+W] mod LCM
        # Candidates are p such that (p - S) % LCM <= W
        
        # We build a list of candidates for each sorted divisor
        # To avoid allocating lists every time, we can just iterate and check.
        # But we need to sort candidates? No, we just need to pick one.
        
        # Let's collect candidates for each sorted divisor
        # cand[i] is a list of positions for _sorted_divisors[i]
        # To save time, we can pre-filter: if a divisor has no candidates, fail fast.
        
        # We will use a recursive backtracking function.
        # To make it fast, we use a bitmask for used positions.
        # Since positions are in [0, LCM), we can't use a simple bitmask for positions directly if LCM is large.
        # But the number of candidates is small.
        # We can map the specific candidates in this window to indices 0..K-1.
        
        # Let's just use a set of used positions, but optimize the loop.
        # Actually, since N=10, and candidates are few, a simple list of used positions is fine.
        
        # Better: Precompute for each S? No, S changes.
        
        # Let's try a different approach:
        # For a fixed S, the available positions for divisor i are a subset of _residues[i].
        # We can intersect _residues[i] with the window.
        
        # To speed up, we can note that if the window is small, the number of candidates is small.
        
        # Let's use a global 'used' set and backtrack.
        used = set()
        assignment = [None] * _N
        
        def backtrack(idx):
            if idx == _N:
                return True
            
            d_idx = _indices[idx] # Original index
            d = _DIVISORS[d_idx]
            res_list = _residues[d_idx]
            
            # Find candidates in window [S, S+W]
            # Since res_list is sorted, we can binary search or just iterate.
            # Given small size, iteration is fine.
            # To be faster, we can precompute the start index in res_list for S?
            # (S % d) gives the first multiple >= 0.
            # First multiple >= S:
            # m = (S // d) * d
            # if m < S: m += d
            # Then check m, m+d, ... <= S+W (handling wrap around)
            
            # Let's just iterate res_list and check condition.
            # Optimization: Skip if p not in window.
            
            # To order candidates by "most constrained" is hard here.
            # Just iterate.
            
            for p in res_list:
                # Check if p is in [S, S+W] mod LCM
                # (p - S) % LCM <= W
                if (p - S) % _LCM <= W:
                    if p not in used:
                        used.add(p)
                        assignment[d_idx] = p
                        if backtrack(idx + 1):
                            return True
                        used.remove(p)
                        assignment[d_idx] = None
            return False
        
        if backtrack(0):
            return S, assignment
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_W = None
    best_S = None
    best_assignment = None
    
    lo = 1
    hi = _LCM
    
    # Binary search on W
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        res = _check_window(mid)
        if res is None:
            if time.time() > deadline:
                break
            lo = mid + 1
        else:
            S, assignment = res
            best_W = mid
            best_S = S
            best_assignment = assignment
            hi = mid
    
    if best_W is None:
        return list(_DIVISORS)
    
    # Construct actual integers
    min_p = min(best_assignment)
    max_p = max(best_assignment)
    
    # We need S_global such that S_global + p is in [1, 1000000]
    # S_global must be a multiple of LCM? No.
    # The positions p are residues mod LCM.
    # The actual numbers are X_i = S_global + p_i.
    # X_i must be divisible by d_i.
    # Since p_i is divisible by d_i, S_global must be divisible by d_i?
    # No. X_i = S_global + p_i.
    # X_i % d_i == (S_global + p_i) % d_i == S_global % d_i.
    # So S_global must be divisible by ALL d_i?
    # Wait. p_i is a multiple of d_i.
    # So p_i % d_i == 0.
    # For X_i to be divisible by d_i, we need S_global % d_i == 0.
    # So S_global must be a multiple of LCM.
    
    S_base = 0
    # Find smallest S_base (multiple of LCM) such that S_base + min_p >= 1
    # and S_base + max_p <= 1000000
    
    min_needed = 1 - min_p
    max_allowed = 1000000 - max_p
    
    # S_base must be >= min_needed and <= max_allowed
    # S_base is a multiple of LCM.
    
    if min_needed <= 0:
        S_base = 0
    else:
        S_base = ((min_needed + _LCM - 1) // _LCM) * _LCM
        
    if S_base > max_allowed:
        # Try to find if any multiple works?
        # If S_base > max_allowed, then the next one is larger, so no solution in this range?
        # Wait, if min_p is large, S_base=0 might work if min_p >= 1.
        # If S_base (smallest valid) is too big, then no solution.
        return list(_DIVISORS)
        
    result = [S_base + p for p in best_assignment]
    
    # Verify
    for i, val in enumerate(result):
        if val <= 0 or val > 1000000:
            return list(_DIVISORS)
        if val % _DIVISORS[i] != 0:
            return list(_DIVISORS)
    if len(set(result)) != _N:
        return list(_DIVISORS)
        
    return result

# EVOLVE-BLOCK-END
