"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find the longest interval [start, start+length-1] within [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    Approach:
    1. Precompute the modular inverse of each residue in 1..1008 modulo 1009.
    2. For each residue x, its inverse is inv(x). The pair (x, inv(x)) is forbidden.
       Note: x=1 has inv=1, x=1008 has inv=1008 (since 1008^2 = (-1)^2 = 1 mod 1009).
       So 1 and 1008 cannot be in any valid interval.
    3. For a given interval [L, R], it is valid iff for all x in [L, R], inv(x) is NOT in [L, R].
       Equivalently, for all x in [L, R], inv(x) < L or inv(x) > R.
    4. We want to maximize R - L + 1.
    
    Efficient approach:
    For each possible left boundary L, find the maximum R such that the interval [L, R] is valid.
    The constraint is: for all x in [L, R], inv(x) is not in [L, R].
    
    Let's think about this differently. For a fixed L, as we extend R, we need to ensure that
    no x in [L, R] has its inverse also in [L, R].
    
    Define for each position x (1..1008), the "conflict" with its inverse.
    If we include x in our interval, we must exclude inv(x).
    
    For a fixed L, the maximum valid R can be found by noting that R is limited by:
    - For each x in [L, R], if inv(x) >= L, then we must have R < inv(x) (if inv(x) > x)
      or R < L (impossible if inv(x) < L, which is fine).
    
    Actually, let's use a sweep-line approach:
    For each x in 1..1008, the pair (x, inv(x)) is a conflict.
    For an interval [L, R] to be valid, it must not contain both x and inv(x) for any x.
    
    Key insight: For a fixed L, the maximum R is determined by the minimum over all x >= L of:
    - If inv(x) >= L, then R must be < min(inv(x), x) ... wait, let me reconsider.
    
    Actually, if x is in [L, R] and inv(x) is also in [L, R], that's a conflict.
    So for a fixed L, as R increases, the first conflict occurs when we hit some x such that
    inv(x) is also in [L, R].
    
    Let's precompute for each L the maximum valid R using a two-pointer/sweep approach.
    """
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        # Fermat's little theorem: x^(p-2) mod p
        inv[x] = pow(x, p - 2, p)
    
    # For each L, find max R such that [L, R] is valid
    # Use two-pointer: for each L, maintain R and check validity
    
    best_start = 1
    best_len = 0
    
    # For efficiency, precompute for each x whether inv(x) == x (self-inverse)
    # Self-inverses: x^2 ≡ 1 mod p => x ≡ 1 or x ≡ -1 mod p => x=1 or x=1008
    
    # Two-pointer approach:
    # For L from 1 to n:
    #   Ensure R >= L
    #   While R+1 <= n and [L, R+1] is valid, increment R
    #   Record [L, R] as a candidate
    #   When incrementing L, we might be able to extend R further
    
    # To check if adding R+1 to [L, R] keeps it valid:
    # We need inv(R+1) not in [L, R+1].
    # Since R+1 is in [L, R+1], we need inv(R+1) < L or inv(R+1) > R+1.
    # But also, for all existing x in [L, R], we already ensured inv(x) not in [L, R].
    # When we add R+1, we need to check that inv(R+1) is not in [L, R] (since R+1 is new).
    # And we need to check that R+1 is not the inverse of any existing x in [L, R].
    # i.e., for all x in [L, R], x != inv(R+1), which means inv(R+1) not in [L, R].
    
    # So the condition to extend R to R+1 is: inv(R+1) < L or inv(R+1) > R+1.
    # Wait, inv(R+1) > R+1 is possible? No, inv(R+1) is in 1..1008, and R+1 <= 1008.
    # So inv(R+1) > R+1 means inv(R+1) is in [R+2, 1008], which is outside [L, R+1].
    # And inv(R+1) < L means it's outside [L, R+1].
    
    # So the condition is simply: inv(R+1) not in [L, R+1].
    # i.e., inv(R+1) < L or inv(R+1) > R+1.
    
    L = 1
    R = 0
    for L in range(1, n + 1):
        if R < L:
            R = L - 1
        
        # Try to extend R as far as possible
        while R + 1 <= n:
            next_val = R + 1
            next_inv = inv[next_val]
            # Check if next_inv is in [L, next_val]
            if next_inv < L or next_inv > next_val:
                R = next_val
            else:
                break
        
        # Also need to check that all x in [L, R] have inv(x) not in [L, R]
        # But with the two-pointer approach, when we extended R, we only checked the new element.
        # However, when L increases, some previously valid configurations might become invalid?
        # No: if [L, R] was valid, then [L+1, R] is also valid (removing elements can't create conflicts).
        # So we don't need to re-check.
        
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
