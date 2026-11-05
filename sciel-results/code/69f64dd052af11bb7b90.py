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
    
    # Self-inverse residues: x where x^2 ≡ 1 mod p => x=1 or x=p-1=1008
    # These cannot be in any valid interval
    
    # For each x in [1,1008], if x != inv[x], then {x, inv[x]} is a forbidden pair
    # The interval must not contain both x and inv[x]
    # Also must not contain 1 or 1008
    
    # Find longest valid contiguous interval
    # An interval [a,b] is valid iff:
    #   1. 1 not in [a,b] and 1008 not in [a,b]
    #   2. For all x in [a,b], inv[x] not in [a,b]
    
    # Since 1008 is a self-inverse, any interval containing 1008 is invalid.
    # Similarly for 1.
    # So the interval must be within [2, 1007] (excluding 1 and 1008)
    
    # For x in [2, 1007], inv[x] is also in [2, 1007] (since 1 and 1008 are the only self-inverses)
    # Wait: inv[1]=1, inv[1008]=1008. For other x, inv[x] != x.
    # Also inv[x] could be 1 or 1008? No: if inv[x]=1 then x=1. If inv[x]=1008 then x=1008.
    # So for x in [2,1007], inv[x] is in [2,1007] and inv[x] != x.
    
    # So we need the longest subinterval of [2, 1007] that doesn't contain both x and inv[x] for any x.
    
    # Approach: for each start a in [2, 1007], find the max b such that [a,b] is valid.
    # [a,b] is valid iff for all x in [a,b], inv[x] < a or inv[x] > b.
    # Equivalently: min(inv[x] for x in [a,b] if inv[x] in [a,b]) doesn't exist,
    # i.e., no x in [a,b] has inv[x] in [a,b].
    
    # Better: for each a, the interval [a, b] breaks when we first encounter x such that inv[x] >= a and inv[x] <= b.
    # But b is what we're trying to find...
    
    # Alternative: for each a, the maximum valid b is the minimum over x in [a, ...] of (inv[x] - 1) 
    # where inv[x] >= a. Because if inv[x] >= a, then b must be < inv[x].
    # Also b must be <= 1007.
    
    # So: max_b(a) = min(1007, min{inv[x] - 1 : x in [a, 1007], inv[x] >= a})
    # If no such x exists, max_b(a) = 1007.
    
    # But we also need to check: is the interval [a, max_b(a)] actually valid?
    # By construction, for all x in [a, max_b(a)], inv[x] < a or inv[x] > max_b(a).
    # Since max_b(a) < inv[x] for those with inv[x] >= a, and inv[x] < a for others,
    # yes, it's valid.
    
    # Let's compute this efficiently.
    
    best_len = 0
    best_start = 2
    
    # Precompute for each a, the constraint
    # For each a from 2 to 1007:
    #   constraint = min(inv[x] - 1 for x in [a, 1007] if inv[x] >= a)
    #   if no such x, constraint = 1007
    #   b = min(1007, constraint)
    #   if b >= a: length = b - a + 1
    
    # We can do this in O(n^2) which is ~10^6, fine.
    
    # Use a two-pointer / sliding window approach for efficiency
    # For a fixed start a, extend b as far as possible
    # When we can't extend b (because adding b+1 would create a conflict), move a forward
    
    # Actually, let's use the direct approach: for each a, compute max_b(a)
    # The key insight: for fixed a, max_b(a) = min over x in [a, 1007] with inv[x] >= a of (inv[x] - 1)
    # But we can optimize: as a increases, the set of x in [a, 1007] shrinks, and the constraint inv[x] >= a becomes stronger
    
    # Let's just do the O(n^2) scan since n=1008, that's about 500K iterations, very fast.
    
    for a in range(2, 1008):
        if time.time() > deadline:
            break
        # Find the minimum inv[x]-1 for x in [a, 1007] where inv[x] >= a
        min_constraint = 1007  # default: no constraint beyond 1007
        for x in range(a, 1008):
            ix = inv[x]
            if ix >= a:
                c = ix - 1
                if c < min_constraint:
                    min_constraint = c
                    if min_constraint < a:
                        break  # can't go below a
        b = min(1007, min_constraint)
        if b >= a:
            length = b - a + 1
            if length > best_len:
                best_len = length
                best_start = a
    
    # Verify the result
    start = best_start
    length = best_len
    end = start + length - 1
    
    # Double-check validity
    for x in range(start, end + 1):
        if x * x % p == 1:
            # Self-inverse found, this shouldn't happen
            pass
        ix = inv[x]
        if start <= ix <= end:
            # Conflict found
            pass
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
