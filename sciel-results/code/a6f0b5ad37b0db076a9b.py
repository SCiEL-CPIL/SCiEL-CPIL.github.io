"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = 1008
    
    # Deadline guard
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Precompute modular inverses: inv[x] = x^{-1} mod p
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x * x % p == 1 => x = 1 or x = p-1 = 1008
    # These cannot appear in any valid interval
    
    # Two-pointer sliding window to find longest valid interval
    # An interval [L, R] is valid iff for all x in [L,R]:
    #   1. x is not self-inverse (x != 1 and x != 1008)
    #   2. inv[x] is not in [L, R] (unless inv[x] == x, which is case 1)
    
    # Since inv is an involution (inv[inv[x]] = x), the forbidden pairs are
    # disjoint. For a window [L, R], it's invalid iff there exists x in [L,R]
    # such that inv[x] is also in [L,R].
    
    # Strategy: slide R from 1 to n. For each R, the new constraint is that
    # inv[R] must not be in [L, R-1]. If inv[R] >= L, we must push L past inv[R].
    # Also, if R itself is self-inverse, it can't be in any window, so skip.
    
    best_len = 0
    best_start = 1
    L = 1
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # If R is self-inverse, it can't be in any valid interval
        if R == 1 or R == 1008:
            L = R + 1
            continue
        
        invR = inv[R]
        
        # If invR is in the current window [L, R-1], we have a conflict
        if L <= invR < R:
            L = invR + 1
        
        # Now window is [L, R]. Check it's non-empty.
        # By construction, no x in [L, R] has its inverse also in [L, R]:
        # - For R: invR < L (we just ensured this)
        # - For any x in [L, R-1]: if inv[x] were in [L, R], since inv[x] != x
        #   (x is not self-inverse), inv[x] must be either < L or > R.
        #   But wait - we need to be more careful.
        
        # Actually, the two-pointer approach works because:
        # When we add R, the only new potential conflict is between R and inv[R].
        # If inv[R] was in [L_old, R-1], we move L to inv[R]+1.
        # After this, for all x in [L_new, R-1], their inverses were already
        # checked when they were added (and were outside the window at that time).
        # Moving L forward only removes elements, so it can't create new conflicts.
        
        length = R - L + 1
        if length > 0 and length > best_len:
            best_len = length
            best_start = L
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
