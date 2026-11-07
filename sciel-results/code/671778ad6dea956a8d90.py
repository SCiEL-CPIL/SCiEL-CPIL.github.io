"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVISORS)
_MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for d in _DIVISORS:
    _LCM = _lcm(_LCM, d)

def _can_assign_fast(L, R, divisors):
    """Check if we can assign distinct multiples in [L, L+R-1] for each divisor.
    Returns the assignment list or None."""
    n = len(divisors)
    candidates = []
    for i in range(n):
        d = divisors[i]
        k_start = (L + d - 1) // d
        k_end = (L + R - 1) // d
        if k_start > k_end:
            return None
        cand = list(range(k_start * d, k_end * d + 1, d))
        candidates.append(cand)
    
    # MRV heuristic: process divisors with fewest candidates first
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    assigned = [None] * n
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return assigned
    return None

def _get_valid_L_mask(R, divisors, LCM):
    """Create a boolean mask of valid L values in [0, LCM-1] for a given R."""
    # L is 1-indexed in the search loop, but modulo arithmetic is 0-indexed.
    # We check L in [1, LCM]. 
    # valid[L-1] corresponds to L.
    mask = [True] * LCM
    for d in divisors:
        # For a given L, a multiple of d exists in [L, L+R-1]
        # iff there is no "gap" of length R in the multiples of d covering [L, L+R-1].
        # Multiples are at 0, d, 2d, ...
        # The interval [L, L+R-1] contains a multiple iff 
        # ceil(L/d) <= floor((L+R-1)/d)
        # Let's just check each L modulo d.
        # Actually, we can just mark invalid L's.
        # L is invalid if [L, L+R-1] contains no multiple of d.
        # This happens if L is in (k*d, (k+1)*d - R + 1) ?
        # Let multiple be m. m in [L, L+R-1] <=> L <= m <= L+R-1 <=> m-R+1 <= L <= m.
        # So L is VALID if it is in the union of intervals [m-R+1, m] for all multiples m.
        # L is INVALID if it is NOT in any such interval.
        
        # We iterate over the period d.
        # For each residue r in [0, d-1], check if L=r is valid.
        # If L=r, interval is [r, r+R-1].
        # Does it contain a multiple of d?
        # Multiple is 0 or d (since r < d).
        # If r+R-1 >= 0 and r <= d? 
        # Actually, if R >= d, all L are valid.
        if R >= d:
            continue
        
        # If R < d, there are invalid L's.
        # The valid L's are those where [L, L+R-1] intersects {0, d, 2d...}
        # In the range [0, d-1], the only multiple is 0.
        # So [L, L+R-1] must contain 0? No, it's a window in positive integers.
        # Let's use the property: 
        # L is valid iff (L-1) % d < R or (L-1) % d >= d - (R - 1)? 
        # Simpler: 
        # For each L in [1, d], check if valid.
        # Then tile this pattern to [1, LCM].
        
        invalid_residues = set()
        for L_mod in range(1, d + 1):
            # Window [L_mod, L_mod + R - 1]
            # Check if any multiple of d is in this window.
            # Since L_mod <= d, the possible multiples are 0 (if L_mod=0? No L>=1) and d.
            # If L_mod + R - 1 >= d, then d is in the window. Valid.
            # If L_mod + R - 1 < d, then only 0 is possible? But L>=1, so 0 is not in window.
            # So if L_mod + R - 1 < d, NO multiple. Invalid.
            # Wait, what about 2d? No, window size R < d.
            # So if L_mod + R - 1 < d, it's invalid.
            # If L_mod + R - 1 >= d, it's valid.
            
            # This logic holds for L in [1, d].
            if L_mod + R - 1 < d:
                invalid_residues.add(L_mod)
        
        # Now apply this to the full mask
        # L in [1, LCM].
        # L % d gives the residue. Note: in Python, L % d is 0..d-1.
        # Our L_mod was 1..d.
        # Residue 0 corresponds to L_mod d.
        # Residue 1..d-1 corresponds to L_mod 1..d-1.
        
        # Map invalid_residues (1..d) to 0..d-1
        # L_mod d -> res 0
        # L_mod k -> res k-1
        
        invalid_res_0 = set()
        for r in invalid_residues:
            invalid_res_0.add((r - 1) % d)
        
        # Mark invalid in mask
        # Mask index i corresponds to L = i + 1.
        # L = i + 1.
        # Residue of L is (i+1) % d.
        # If (i+1) % d is in invalid_res_0, then mask[i] = False.
        
        for i in range(LCM):
            if mask[i]: # Only check if still valid
                res = (i + 1) % d
                if res in invalid_res_0:
                    mask[i] = False
    
    return mask

def construct():
    deadline = time.time() + 85
    divisors = _DIVISORS
    n = len(divisors)
    
    # Search over R from small to large
    for R in range(1, 1000):
        if time.time() > deadline:
            break
        
        # Get valid L's
        valid_mask = _get_valid_L_mask(R, divisors, _LCM)
        
        # Iterate over valid L's
        for L_idx in range(_LCM):
            if not valid_mask[L_idx]:
                continue
            
            L = L_idx + 1
            if time.time() > deadline:
                break
            
            assigned = _can_assign_fast(L, R, divisors)
            if assigned is not None:
                if max(assigned) <= _MAX_VAL:
                    return [int(assigned[i]) for i in range(n)]
    
    # Fallback
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
