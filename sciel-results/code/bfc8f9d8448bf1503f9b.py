"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    p = 1009
    n = 1008  # residues 1..1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # The interval must be within [2, 1007] to avoid self-inverses 1 and 1008.
    # For x in [2, 1007], inv[x] is also in [2, 1007] and inv[x] != x.
    # We seek the longest subinterval [L, R] of [2, 1007] such that
    # for all x in [L, R], inv[x] is not in [L, R].
    
    # Two-pointer sliding window approach:
    # L is the left boundary, R is the right boundary.
    # We expand R from 2 to 1007.
    # If adding R creates a conflict (i.e., inv[R] is in [L, R-1]),
    # we must move L to inv[R] + 1 to exclude the pair {inv[R], R}.
    # Note: inv[R] < R is not guaranteed, but if inv[R] >= L, then inv[R] is in [L, R-1] 
    # because inv[R] != R and inv[R] <= 1007. 
    # Wait, if inv[R] > R, it cannot be in [L, R-1]. 
    # So conflict only if L <= inv[R] < R? 
    # No, if inv[R] >= L and inv[R] <= R-1, then conflict.
    # If inv[R] > R-1, i.e., inv[R] >= R, then inv[R] is not in [L, R-1].
    # But inv[R] could be R? No, self-inverse excluded.
    # So if inv[R] > R, no conflict with previous elements.
    # If inv[R] < L, no conflict.
    # If L <= inv[R] < R, conflict.
    
    # Let's re-verify the condition.
    # Window [L, R-1] is valid.
    # Add R. New window [L, R].
    # Conflict if exists x in [L, R] s.t. inv[x] in [L, R].
    # Since [L, R-1] was valid, no x in [L, R-1] has inv[x] in [L, R-1].
    # New conflicts must involve R.
    # 1. x=R. inv[R] in [L, R]? Since inv[R]!=R, this means inv[R] in [L, R-1].
    #    Condition: L <= inv[R] <= R-1.
    # 2. x in [L, R-1]. inv[x] in [L, R]? Since inv[x] not in [L, R-1], this means inv[x] = R.
    #    This implies x = inv[R]. So this is the same condition: inv[R] in [L, R-1].
    
    # So, if L <= inv[R] < R, we have a conflict.
    # To resolve, we set L = inv[R] + 1.
    # If inv[R] >= R, no conflict. L stays.
    # If inv[R] < L, no conflict. L stays.
    
    best_len = 0
    best_start = 2
    L = 2
    
    for R in range(2, 1008):
        if time.time() > deadline:
            break
        if L <= inv[R] < R:
            L = inv[R] + 1
        # Now [L, R] is valid
        if L <= R:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
        # If L > R, window is empty, length 0
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
